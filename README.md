# Skynet — a general-purpose agent harness

A small, working agent runtime built so that it can grow into a large one
without rewriting the core.

The prototype runs an agent that receives a task, talks to a model, decides what
to do, executes capabilities in an isolated sandbox, observes the results, loops,
optionally delegates to sub-agents, and produces a final result — durably, on
Temporal, behind an HTTP API.

---

## The one idea worth understanding

Everything below follows from a single decision: **an agent is an event-sourced
state machine driven by a pure decision function.**

```
decide(run) -> Decision          pure: what should happen next
execute(run, decision) -> Events the ONLY impure thing in the system
apply(run, event) -> run         pure: the only way state ever changes
```

`AgentLoop.run()` is the entire driver:

```python
while not (run.status.is_terminal or run.status.is_suspended):
    decision = self._policy.decide(run)                  # pure
    events = await self._effects.execute(run, decision)  # the one seam
    for event in events:
        run = apply(run, event)                          # pure fold
```

That is the whole loop, and **this exact class runs unchanged inside a Temporal
workflow** — only the `EffectExecutor` differs. Consequences that would
otherwise be features, and would otherwise be hard, fall out for free:

| Requirement | How it is satisfied |
|---|---|
| Durable execution | State is a fold over a log, so it replays |
| Crash recovery | `replay(initial, events)` |
| Pause / resume | An event |
| Human approval | An event |
| Audit trail | The log *is* the audit trail |
| Testing without infrastructure | `decide` is a pure function; no mocks needed |

Two architecture tests keep it honest: `skynet.domain` may not import anything
non-deterministic, and the loop must contain **exactly one `await`**.

---

## Quick start

### Run it locally, no infrastructure

```bash
uv sync
uv run pytest                      # 460 tests, offline, ~11s
uv run skynet agents               # what is configured
uv run skynet tools                # what capabilities exist
uv run skynet run "Compute the 20th Fibonacci number, write it to fib.txt, and read it back"
```

Point it at a model by editing `.env` (copy `.env.example`). With Ollama on the
host, nothing else is needed.

### Run the full durable stack

```bash
cp .env.example .env               # replace REPLACE_WITH_* values; match `SKYNET_DB_USER` and `SKYNET_DB_PASSWORD` in `SKYNET_MEMORY__DSN`
# LANGFUSE_ENCRYPTION_KEY must be 64 hex characters; use unique values for every secret
# For host-side Postgres runs, set SKYNET_MEMORY__DSN; Compose builds its internal DSN automatically.
docker compose up -d --build

curl -XPOST localhost:8000/v1/runs -H 'Content-Type: application/json' \
  -d '{"objective":"Compute the factorial of 25 and write it to fact.txt"}'

curl -N localhost:8000/v1/runs/<run_id>/stream    # live server-sent events
```

- API — <http://localhost:8000/docs>
- Temporal UI — <http://localhost:8233>
- MinIO console — <http://localhost:9001> (use the credentials configured in `.env`)

Optional profiles, both off by default:

```bash
docker compose --profile langfuse up -d    # self-hosted Langfuse (+5 containers)
docker compose --profile ollama up -d      # containerised Ollama instead of the host's
```

### Langfuse

The profile provisions its org, project and API keys on first boot, so there is
nothing to click through:

| | |
|---|---|
| UI | <http://localhost:3000> |
| Login and API keys | Configure unique credentials in `.env` before starting the profile |

Then enable it and publish the prompts:

```bash
SKYNET_TRACING__BACKEND=langfuse
SKYNET_PROMPTS__USE_LANGFUSE=true

uv run python scripts/push_prompts.py      # YAML -> Langfuse, labelled "production"
```

Langfuse fills **two unrelated roles**, so it sits behind **two ports**
(`Tracer` and `PromptProvider`) rather than one:

- **Tracing** — a run becomes one named trace grouped by session, with typed
  observations (`agent` / `generation` / `tool`) and native model and token
  usage. In-process, nesting comes from OTel's ambient context; across processes
  the domain's own `TraceContext` is propagated into each Temporal activity, so
  a run stays a single trace either way.
- **Prompts** — `ChainPromptProvider` tries Langfuse first and falls back to the
  version-controlled YAML in `prompts/`. **Langfuse being down is not an
  outage**, and prompts stay reviewable in git.

---

## Layout

Dependencies point one way only, and a test enforces it:

```
domain  <-  ports  <-  application  <-  capabilities  <-  adapters
```

```
src/skynet/
├── domain/         Pure. Pydantic + stdlib. No I/O, no clock, no randomness.
│   ├── run.py        AgentStatus, AgentRun, apply(), replay()   <- the heart
│   ├── events.py     The replayable log (discriminated union)
│   └── decisions.py  What the policy asks the world to do
├── ports/          Protocols only.  effects.py is THE seam.
├── application/    policy (pure brain) · loop · effects · context · runner
├── capabilities/   terminal · code exec · files · memory · delegate · control
├── adapters/       model · prompts · tracing · storage · sandbox · temporal
├── api/            FastAPI. Thin. Depends on ports, never on Temporal.
├── config/         pydantic-settings + the composition root
└── testing/        Fakes and port contract suites (shipped, not test-only)
```

---

## Replaceability, demonstrated rather than asserted

Every mandated technology sits behind a port, and **each port has more than one
working implementation** — an abstraction with a single implementation is
unproven.

| Concern | Port | Implementations |
|---|---|---|
| Model | `ModelProvider` | Ollama · OpenRouter · Gemini · any OpenAI-compatible · emulated-tool-calling decorator |
| Execution | `AgentRunner` | in-process · Temporal |
| Sandbox | `SandboxProvider` | Docker container · local subprocess · fake |
| Object storage | `ObjectStore` | MinIO · filesystem · in-memory |
| Tracing | `Tracer` | Langfuse (typed, nested) · console · no-op |
| Prompts | `PromptProvider` | Langfuse · YAML files · chained with fallback |
| Memory | `MemoryStore` | Postgres + pgvector (hybrid) · in-memory (lexical + trigram) |
| Embeddings | `Embedder` | Ollama · Gemini · deterministic hashing · null |
| Events | `EventSink` / `EventStream` | shared file log · broadcast · logging · composite |

Switching any of them is one environment variable:

```bash
SKYNET_MODEL__PROVIDER=openrouter
SKYNET_EXECUTION__ENGINE=inprocess
SKYNET_SANDBOX__BACKEND=local
SKYNET_TRACING__BACKEND=langfuse
SKYNET_STORAGE__BACKEND=filesystem
SKYNET_MEMORY__BACKEND=postgres
SKYNET_EMBEDDING__BACKEND=ollama
```

The memory store and the object store run the *same* contract suite against
every implementation, which is what turns "replaceable" from a claim into a
test that fails when it stops being true.

`config/container.py` is the only file that knows both sides. Adding a model
provider is a class plus one dict entry; adding a capability is one `register`
call. **Neither touches the agent loop.**

---

## What the design buys you

**Tool problems are observations, never crashes.** An unknown tool, malformed
arguments, a denied permission, a timeout, or an exception inside a tool all
become a `ToolResult` fed back to the model, which then corrects itself. Only
budget exhaustion and non-retryable model failures end a run.

**Argument validation is free and correct.** Tool schemas are generated from
Python type hints by pydantic, and the same model validates incoming arguments.
Schemas cannot drift from implementations.

**Weak models are supported without special cases.** Providers advertise
capabilities; if native tool calling is missing, a decorator emulates it through
prompting and parsing. The core never learns which mode is active.

**Thinking models are handled.** Many reasoning models finish with an empty
`content` and put the answer in a separate `reasoning` channel. `Message.text`
falls back to it, so the result is not silently lost.

**Sub-agents are just a tool.** A `DELEGATION` capability routes to an in-process
spawner or a Temporal child workflow. Hierarchies and parallel children need no
change to the loop. A child's permissions can only ever be a *subset* of its
parent's — `PermissionSet.narrowed` cannot add a grant.

**What a run produces outlives the run.** The sandbox is disposable, so a
`WorkspaceArchiver` copies the workspace into object storage at the moment the
run becomes terminal — before the workspace is released, and on failure too,
where the files are often the most interesting. The references become part of
the result, so they survive replay. Harvesting is bounded (file count, per-file
and total size) because an agent can fill a workspace, and it never raises: a
run that finished must not fail because storage did.

**Isolation is real.** The Docker sandbox runs with a read-only root filesystem,
a tmpfs workspace, no network, and memory/CPU/PID limits. Workspaces are keyed by
run and are adopted back by label after a worker restart.

---

## Testing

```bash
uv run pytest                                  # 460 tests, no Docker, no network
uv run pytest -m integration                   # + Docker sandbox and Postgres contracts
```

There are no mocks. Every port has an in-memory fake in `skynet.testing.fakes`,
and behaviour is asserted against the event log — the runtime's own record of
what happened. `FixedClock` and `SequentialIdGenerator` make that log
byte-stable, so tests assert exact event sequences.

Three kinds of test carry most of the weight:

- **Contract suites** (`skynet.testing.contracts`) — the same assertions run
  against fakes *and* real adapters. This is how "swap MinIO in and nothing
  breaks" becomes checkable. The *same nineteen* memory assertions run against
  the in-memory store and against Postgres + pgvector. They have already caught
  three real bugs: a Docker sandbox that discarded the workspace between tool
  calls, a shell pipeline that masked a missing-file exit code, and pooled
  Postgres connections whose rollback-on-return silently undid every
  `invalidate`.

  One contract assertion had to be *weakened on purpose*: it pinned an exact
  result list for a keyword search, which no store with a semantic arm can
  satisfy, because "volcanoes erupt" has a small non-zero similarity to
  "pangolins". A contract must fix **filtering, which is correctness**, and
  never **ranking, which is quality** — otherwise it silently forbids every
  ranked implementation from ever passing.
- **Replay tests** — run an agent, cut the event log at every point, replay the
  prefix, continue, and assert an identical result. That is the Temporal
  durability contract, verified with zero Temporal.
- **Architecture tests** — layering, determinism of the domain, and the
  single-`await` rule.

---

## Verified end to end

- Agent loop against a live model: model → `python_exec` → `write_file` →
  `read_file` → final answer.
- Sub-agent delegation: a supervisor spawned a researcher child run and used its
  result.
- Three providers through one port: Ollama (`qwen3.5:9b`), OpenRouter free
  models, and Gemini — the core unchanged in each case. Adding Gemini was one
  20-line class and one registry entry.
- Durable execution: a run was started, **the worker was killed mid-flight**,
  restarted, and the run resumed from its next step and completed.
- API restart: a completed run was fully reconstructed — every step and tool
  result — from the event log, with no session state.
- Langfuse: a delegating run appeared as one trace named `agent.supervisor`
  containing the parent's generations, the sub-agent's generations and its tool
  call, with per-call token usage; with Langfuse unreachable, prompts fell back
  to YAML and the run still completed.
- Isolation: read-only rootfs, no DNS, and a 512 MB cgroup limit confirmed
  inside a sandbox container.
- Resource lifecycle: a sandbox container is created for a run and removed when
  the run reaches a terminal state; a restarted worker reaps anything a crashed
  one stranded.
- A complex 7-step task — compute 200 primes, write a CSV, derive statistics,
  commit a fact to memory, **delegate a maths question to a sub-agent**, write a
  markdown report, read it back — completed with every figure matching an
  independently computed ground truth (sum 111,587 · mean 557.93 · 41 twin
  pairs), and both files were retrievable from MinIO afterwards.
- Memory across runs: one run recorded a fact, a later run recalled it before
  its first model call and answered from it **without calling a tool**, and a
  third run under a different agent name could not see it at all.
- Temporal supersession: an agent told "the office moved to Munich" issued an
  `update`, not a second `add`. The old record survived with `valid_to` and
  `superseded_by` set, and `search(as_of=<before>)` still returned Berlin.
- Compaction: a run whose context outgrew its budget folded its own history
  three times — 22 messages into one note — and still answered correctly.
- Hybrid retrieval: a query with **no lexical match at all** was found by the
  vector arm, confirming the two arms are unioned rather than intersected.

---

## Memory

Memory is the one subsystem where an agent's mistakes compound: a wrong fact
written today is reasoned from tomorrow, and nothing in the log explains the
answer it produced. So the design is shaped around being *diagnosable* rather
than around being clever.

**It is bi-temporal.** `created_at` is when a fact was learned; `valid_from` and
`valid_to` are when it held. A contradicted memory is never deleted — it is
stamped and pointed at its replacement. "What did the agent believe last
Tuesday?" stays answerable, which is the only thing that makes a memory bug
diagnosable after the fact.

```
search(as_of=<before>)  ->  "the office is in Berlin"
search(as_of=<after>)   ->  "the office is in Munich"
get(old_id)             ->  still there, valid_to set, superseded_by set
```

**Scopes are values, not strings.** `run:` `agent:` `project:` `user:`, most
specific first. Private by default: an agent reads its own run and agent scopes
unless `agents.yaml` grants it a shared one. `application/memory_scoping.py` is
the only place that decides this, so "which agents can see this fact?" is
answerable by reading one screen.

**Retrieval is hybrid, and ranking is not the store's job.** Postgres composes a
lexical arm and a vector arm over one `live` CTE — full outer join, so it is a
union rather than an intersection — and returns raw component scores. The
application then fuses relevance, recency, importance and scope precedence as a
*weighted sum*: a product would zero out a brand-new memory whose similarity is
merely mediocre, which is exactly the memory just written.

**Everything non-deterministic is still a decision and an event.** Recall,
consolidation, reflection and compaction each appear in the event log with a
watermark that advances *even when the effect produced nothing* — which is the
entire termination argument. `application/loop.py` did not change to add any of
them, and still contains exactly one `await`.

| Step | Decision | Event | Watermark |
|---|---|---|---|
| Retrieve | `RecallMemory` | `MemoryRecalled` | query signature |
| Write down what was learned | `ConsolidateMemory(EXTRACT)` | `MemoryConsolidated` | `consolidated_through` |
| Look across memories | `ConsolidateMemory(REFLECT)` | `MemoryConsolidated` | `reflected_at_iteration` |
| Shrink the conversation | `CompactHistory` | `HistoryCompacted` | `compactions` |

**Memory can never fail a run.** A dead store, a model returning garbage, an
unreadable answer — all degrade to an event with an empty payload that still
advances the watermark. The one exception is deliberate: an embedding failure
stores the record *without* a vector, still lexically searchable, because
telling an agent it remembered something it did not is worse than telling it the
store is down.

Per-agent policy lives in `agents.yaml`:

```yaml
supervisor:
  memory:
    auto_recall: true       # retrieve before the first model call
    consolidate: true       # write down what the run learned, before it ends
    reflect_every: 6        # look across memories every N iterations (0 = off)
    share_project: true     # read a scope shared with other agents
    project: skynet
    write_level: project    # run | agent | project | user
```

---

## Deliberately not built

Named here because the architecture reserves a seam for each, not because they
are hidden:

| Not built | Where it plugs in |
|---|---|
| Advanced planning | A `Decision` variant plus a policy stage |
| Token-level streaming | `StreamingModelProvider`; the loop consumes only final responses |
| Approval UI | `ApprovalGate` — the full pause/approve/resume path already works |
| Browser, git, database tools | A `Tool` and one `register` call |
| Distributed / persistent runs | `RunStore`; `replay()` is already the load path |

### Known limitations

- **Replay is guaranteed within a schema version.** There is no cross-version
  migration path: changing a persisted domain model means clearing the event
  log. New fields must therefore always arrive with defaults.
- **Consolidation costs a model call per run**, and reflection one per interval.
  Point `memory.model` at a cheaper model than the agent's own; both are off by
  default.
- **The offline embedder is not semantic.** `hashing` is the default so the
  stack works with nothing pulled, and it buys partial credit for word variants
  — but `"car"` will not find `"vehicle"`. Switch to `ollama` for that.
- **Sandbox workspaces are tmpfs**, so they do not survive the *container* being
  removed — only the worker restarting, where the container is adopted back by
  label. Persistent sandboxes are a provider change.
- **The shared event log is a file per run.** Deliberately unglamorous, and
  correct; Redis Streams or Postgres would replace it behind the same two ports.
- **Small free models wander.** The budget system stops them (`max_iterations`
  is enforced as a `Decision`, so it is visible in the log), but output quality
  tracks the model you point it at.
