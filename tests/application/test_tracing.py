"""Spans must carry their inputs.

A span with a null input records that something happened but not what was
asked, which is most of the value of a trace. This regressed once already: the
Temporal tool activity opened its span without arguments while the in-process
path passed them, so identical runs traced differently depending on the engine.
"""

from __future__ import annotations

from skynet.application.observability import conversation_input
from skynet.application.runner import InProcessAgentRunner
from skynet.domain.messages import assistant, user
from skynet.domain.models import ModelParams, ModelRequest
from skynet.ports.agents import RunRequest
from skynet.ports.tracing import SpanKind
from skynet.testing.builders import (
    build_services,
    call,
    calls,
    make_context,
    make_spec,
    make_task,
    says,
)
from skynet.testing.fakes import EchoTool


async def traced_run():
    services = build_services(
        responses=[calls(call("echo", "c1", text="hello")), says("done")],
        tools=[EchoTool()],
    )
    await InProcessAgentRunner(services).run(
        RunRequest(spec=make_spec(tools=("echo",)), task=make_task(), context=make_context())
    )
    return services.tracer


def find(tracer, name):
    return next(s for s in tracer.spans if s.name == name)


async def test_the_agent_span_is_typed_as_an_agent():
    tracer = await traced_run()
    assert find(tracer, "agent.demo").kind is SpanKind.AGENT


async def test_model_spans_carry_the_conversation_as_input():
    tracer = await traced_run()
    span = find(tracer, "llm.generate")
    assert span.kind is SpanKind.GENERATION
    assert isinstance(span.input, list) and span.input, "generation input must not be null"
    assert {"role", "content"} <= set(span.input[0])


async def test_tool_spans_carry_their_arguments_as_input():
    tracer = await traced_run()
    span = find(tracer, "tool.echo")
    assert span.kind is SpanKind.TOOL
    assert span.input == {"text": "hello"}, "tool input must not be null"


async def test_tool_spans_record_their_output():
    tracer = await traced_run()
    assert find(tracer, "tool.echo").output == "hello"


async def test_model_spans_record_model_and_token_usage():
    tracer = await traced_run()
    attributes = find(tracer, "llm.generate").attributes
    assert "model" in attributes
    assert attributes["input_tokens"] == 10
    assert attributes["output_tokens"] == 5


def test_conversation_input_flattens_messages_and_tool_calls():
    request = ModelRequest(
        messages=(
            user("do the thing"),
            assistant("working", tool_calls=(call("echo", "c1", text="x"),)),
        ),
        params=ModelParams(model="m"),
    )
    rendered = conversation_input(request)
    assert rendered[0] == {"role": "user", "content": "do the thing"}
    assert rendered[1]["tool_calls"] == ["echo"]
