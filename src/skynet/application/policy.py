"""The agent's brain -- a pure function from state to intent.

``decide`` contains no I/O, no ``await``, no clock and no randomness. Every
control-flow concern the runtime has (retries, budgets, approvals, completion
detection) is expressed here as a ``Decision``, which means all of it shows up
in the event log, replays deterministically under Temporal, and is testable by
constructing an ``AgentRun`` and calling one function -- no mocks at all.
"""

from __future__ import annotations

from skynet.application import history_compaction as compaction
from skynet.application.context import ContextBuilder
from skynet.application.memory_scoping import recall_scopes, write_scope
from skynet.domain.agent import FinishMode
from skynet.domain.decisions import (
    CallModel,
    CompactHistory,
    Complete,
    ConsolidateMemory,
    Decision,
    Fail,
    InvokeTools,
    RecallMemory,
    RequestApproval,
    ResolvePrompt,
)
from skynet.domain.errors import ErrorInfo
from skynet.domain.memory import ConsolidationMode, MemoryQuery
from skynet.domain.messages import Role, ToolCall
from skynet.domain.models import ModelRequest
from skynet.domain.run import AgentRun
from skynet.domain.task import AgentResult
from skynet.domain.tools import ToolKind
from skynet.ports.tools import ToolCatalog


class AgentPolicy:
    def __init__(self, catalog: ToolCatalog, context_builder: ContextBuilder) -> None:
        self._catalog = catalog
        self._context = context_builder

    def decide(self, run: AgentRun) -> Decision:
        limits = run.spec.limits

        # 1. Bootstrap: the prompt must exist before anything is built from it.
        if run.prompt is None:
            return ResolvePrompt(ref=run.spec.prompt, variables=self._prompt_variables(run))

        # 2. Recall, guarded by "this exact retrieval has already been issued"
        #    rather than by a once-per-run flag.
        if (decision := self._recall_stage(run)) is not None:
            return decision

        # 3. Drain outstanding tool work FIRST.
        #
        # Putting this ahead of everything else is what makes resumption free: a
        # run rebuilt mid-flight (after a crash, a pause, or an approval) simply
        # re-enters here. There is no separate "resume" code path anywhere.
        if run.pending_calls and (decision := self._tool_stage(run)):
            return decision

        # Computed once and reused by stages 4 and 7: a run that is already
        # finished must not spend a model call shrinking a context it will
        # never send.
        finished = self._completion(run)

        # 4. Compaction, before anything that would build a request.
        if finished is None and (decision := self._compaction_stage(run)) is not None:
            return decision

        # 5. Model failures. Retrying is a decision, so it is visible in the log
        #    rather than hidden inside a provider adapter.
        if run.model_failures:
            retryable = run.last_error is not None and run.last_error.retryable
            if retryable and run.model_failures <= limits.max_model_failures:
                return CallModel(request=self._build_request(run))
            return Fail(error=run.last_error or ErrorInfo.of("model_failed", "model call failed"))

        # 6. Unparseable tool syntax from an emulated provider: one corrective turn.
        if run.parse_retries:
            if run.parse_retries > limits.max_parse_retries:
                return Fail(
                    error=ErrorInfo.of("tool_format", "model could not produce a valid tool call")
                )
            return CallModel(request=self._build_request(run))

        # 7. Completion, gated on consolidation so what the run learned is
        #    written down before the run stops existing.
        if finished is not None:
            if (decision := self._consolidation_stage(run)) is not None:
                return decision
            return Complete(result=finished)

        # 8. Budgets.
        if run.iteration >= limits.max_iterations:
            return Fail(error=ErrorInfo.limit("max_iterations", limits.max_iterations))
        if limits.max_total_tokens is not None and run.usage.total >= limits.max_total_tokens:
            return Fail(error=ErrorInfo.limit("max_total_tokens", limits.max_total_tokens))

        # 9. Reflection, before thinking rather than after, so any insight it
        #    produces is in the store before the next request is built.
        if (decision := self._reflection_stage(run)) is not None:
            return decision

        # 10. Otherwise: think.
        return CallModel(request=self._build_request(run))

    # -- stages ---------------------------------------------------------------

    def _recall_stage(self, run: AgentRun) -> Decision | None:
        """Retrieve once per distinct query.

        The watermark is the set of query *signatures*, which is what lets a
        later, differently-scoped retrieval fire while an identical one cannot
        loop. A boolean here would forbid the first and permit neither.
        """
        if not run.spec.memory.auto_recall:
            return None
        query = self._recall_query(run)
        if query.signature() in run.recalled_queries:
            return None
        return RecallMemory(query=query)

    def _compaction_stage(self, run: AgentRun) -> Decision | None:
        """Fold the head of a conversation that has grown too large.

        Three independent guards, and all three are load-bearing:

        * a count budget, so a pathological run cannot summarise indefinitely;
        * a real slice to fold, so the message count strictly decreases;
        * a head that is a meaningful share of the context. Without the last
          one, a run whose *tail alone* exceeds the budget would summarise
          forever without shrinking -- and that case is the trimmer's job.
        """
        limits = run.spec.limits
        if run.compactions >= limits.max_compactions:
            return None

        size = compaction.estimated_chars(run.messages)
        if size <= limits.max_context_chars * compaction.TRIGGER_RATIO:
            return None

        through = compaction.compaction_slice(run.messages)
        if through < 1:
            return None
        if not compaction.worth_folding(run.messages, through, budget=limits.max_context_chars):
            return None

        return CompactHistory(
            request=compaction.compaction_request(
                run.messages,
                through_index=through,
                params=run.spec.memory.model or run.spec.model,
            ),
            through_index=through,
            keep_last=compaction.KEEP_LAST,
            reason=f"context at {size} chars against a budget of {limits.max_context_chars}",
        )

    def _consolidation_stage(self, run: AgentRun) -> Decision | None:
        """Write down what the run learned, exactly once, before it completes.

        Termination rests on ``consolidated_through``, which this sets to
        ``len(run.messages)`` -- and nothing appends to ``messages`` after the
        run is finished, so the guard cannot re-open. That is why this stage
        does *not* also check the count budget: it is already bounded, and a
        budget spent on reflection must not be able to silence the one pass that
        matters most.
        """
        memory = run.spec.memory
        if not memory.consolidate or run.consolidated_through >= len(run.messages):
            return None
        return ConsolidateMemory(
            mode=ConsolidationMode.EXTRACT,
            scope=write_scope(run),
            from_index=run.consolidated_through,
            through_index=len(run.messages),
            retrieval=self._recall_query(run),
            params=memory.model or run.spec.model,
            idempotency_key=(
                f"{run.run_id}:extract:{run.consolidated_through}:{len(run.messages)}"
            ),
        )

    def _reflection_stage(self, run: AgentRun) -> Decision | None:
        """Look across existing memories for something none of them says alone.

        Watermarked on ``reflected_at_iteration`` rather than on
        ``consolidated_through``: reflection reads memories, not messages, so it
        advances no message watermark, and ``iteration`` only moves when the
        model responds. Without an iteration watermark this would re-fire
        forever on the same turn.
        """
        memory = run.spec.memory
        every = memory.reflect_every
        if every <= 0 or run.iteration <= 0 or run.iteration % every:
            return None
        if run.reflected_at_iteration == run.iteration:
            return None
        if run.consolidation_count >= run.spec.limits.max_consolidations:
            return None
        return ConsolidateMemory(
            mode=ConsolidationMode.REFLECT,
            scope=write_scope(run),
            from_index=run.consolidated_through,
            # Reflection consumes no conversation, so it must leave the message
            # watermark exactly where it found it or extraction would starve.
            through_index=run.consolidated_through,
            retrieval=self._recall_query(run),
            params=memory.model or run.spec.model,
            idempotency_key=f"{run.run_id}:reflect:{run.iteration}",
        )

    def _tool_stage(self, run: AgentRun) -> Decision | None:
        unapproved = tuple(
            call
            for call in run.pending_calls
            if self._requires_approval(call) and call.id not in run.approvals
        )
        if unapproved:
            return RequestApproval(
                calls=unapproved, reason="one or more tools require human approval"
            )

        limits = run.spec.limits
        if run.tool_call_count + len(run.pending_calls) > limits.max_tool_calls:
            return Fail(error=ErrorInfo.limit("max_tool_calls", limits.max_tool_calls))

        return InvokeTools(calls=run.pending_calls)

    def _completion(self, run: AgentRun) -> AgentResult | None:
        if not run.messages or run.messages[-1].role is not Role.ASSISTANT:
            return None
        last = run.messages[-1]

        if run.finish_mode is FinishMode.NO_TOOL_CALLS:
            if last.tool_calls:
                return None
            # `text` falls back to the reasoning channel: a thinking model
            # routinely ends with an empty `content`, and reporting that as the
            # answer would silently lose the result.
            return AgentResult(output=last.text, usage=run.usage, iterations=run.iteration)

        # FINAL_ANSWER_TOOL: completion must be explicit, because weak models
        # emit conversational filler instead of stopping.
        for call in last.tool_calls:
            spec = self._catalog.spec(call.name)
            if spec is not None and spec.kind is ToolKind.CONTROL:
                answer = call.arguments.get("answer", "")
                data = call.arguments.get("data")
                return AgentResult(
                    output=str(answer),
                    data=data if isinstance(data, dict) else None,
                    usage=run.usage,
                    iterations=run.iteration,
                )
        return None

    # -- helpers --------------------------------------------------------------

    def _recall_query(self, run: AgentRun) -> MemoryQuery:
        """Built purely, so the signature is the same live and on replay.

        ``as_of`` is deliberately absent: the policy has no clock. The executor
        stamps it, and the event carries it, which is what keeps ranking
        reproducible without making this function impure.
        """
        memory = run.spec.memory
        return MemoryQuery(
            scopes=recall_scopes(run),
            text=run.task.objective,
            kinds=memory.recall_kinds,
            limit=memory.recall_limit,
        )

    def _build_request(self, run: AgentRun) -> ModelRequest:
        # The spec's allowlist is the only source of authority. Delegation is an
        # ordinary tool name in that list; `spec.sub_agents` then constrains
        # which agents it may target.
        tools = self._catalog.specs_for(run.spec.tools, run.context.permissions)
        return self._context.build(run, tools)

    def _requires_approval(self, call: ToolCall) -> bool:
        spec = self._catalog.spec(call.name)
        return spec is not None and spec.requires_approval

    @staticmethod
    def _prompt_variables(run: AgentRun) -> dict[str, str]:
        return {
            "agent_name": run.spec.name,
            "objective": run.task.objective,
            "task_id": str(run.task.id),
            "run_id": str(run.run_id),
            **run.context.variables,
        }


__all__ = ["AgentPolicy"]
