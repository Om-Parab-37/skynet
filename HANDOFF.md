# Skynet — session handoff

Everything a fresh session needs to pick up this project and implement the next
phase. Read this, then the plan.

**The plan:** `/home/omp/.claude/plans/custom-agent-harness-lucky-sonnet.md`
— *Advanced Agent Memory*. **All seven phases are implemented and verified.**
There is no outstanding planned work; what follows is the state to build on.

---

## 1. Where things stand

The agent harness is **built and working end to end**:

| | |
|---|---|
| Tests | **462 offline** (`uv run pytest`, ~12s, no network, no Docker) + **29 integration** (`-m integration`, ~70s) |
| Lint | `ruff check` and `ruff format --check` clean |
| VCS | 9 commits on `master`, no remote. One per phase, each green |
| Stack | `docker compose --profile langfuse up -d` — 13 services, all healthy |

Verified working, not just implemented: durable runs on Temporal (worker killed
mid-run resumes and completes), sub-agent delegation as a child workflow, a
7-step complex task whose every figure matched independently computed ground
truth, artifact harvesting into MinIO, Langfuse tracing with nested typed
observations, and three model providers behind one port.

**Memory (phases 1–7) is likewise verified against the running stack**, not just
unit-tested: a fact written by one run was recalled by a later run *after the
worker was restarted* and answered with zero tool calls; a contradiction
invalidated the old row in Postgres while leaving it readable with `valid_to`
set; a long run folded its own history and still answered correctly; and a
query with no lexical match at all was found by the vector arm.

`README.md` is accurate and current — including its "known limitations" section.
Keep it that way; it has been kept honest deliberately.

---

## 2. The architecture in one screen

```
decide(run) -> Decision          pure "brain"
execute(run, decision) -> Events the ONLY impure thing
apply(run, event) -> run         pure reducer
```

`application/loop.py` alternates these in ~25 lines and **runs unchanged inside
the Temporal workflow** — only the `EffectExecutor` differs.

Layering, enforced by `tests/arch/test_layering.py`:

```
domain <- ports <- application <- capabilities <- adapters
```

### Invariants you must not break

These are enforced by tests. If one fails, the design has gone wrong — do not
"fix" the test.

1. **`skynet.domain` imports nothing non-deterministic** — no `asyncio`, `random`,
   `time`, `uuid`, `os`, `pathlib`. This is what makes replay sound.
2. **`application/loop.py` contains exactly one `await`.** Adding a second means
   a side effect leaked out of the executor.
3. **`application/` never imports `adapters/`**, and core code never names a
   provider (`ollama`, `temporal`, `langfuse`, …) — checked against parsed code,
   not prose, so docstrings may discuss them.
4. **Tool problems are observations, not exceptions.** Unknown tool, bad
   arguments, denied permission, timeout, crash — all become a `ToolResult` fed
   back to the model. Only budget exhaustion and non-retryable model failure end
   a run.
5. **Every non-deterministic step is a Decision + Event with a watermark** that
   advances *even when the effect produced nothing*. That is what makes stages
   terminate.

---

## 3. Running it

```bash
uv sync
uv run pytest -q                          # 462 offline
uv run --extra all pytest -q -m integration   # 29; needs Docker + memory-db
uv run skynet agents | tools              # what's configured
uv run skynet run "..." --agent assistant # in-process, no infra

docker compose --profile langfuse up -d --build
```

| Service | URL | Credentials |
|---|---|---|
| Memory DB | `localhost:5433/skynet_memory` | Configured in `.env` |
| API docs | http://localhost:8000/docs | — |
| Temporal UI | http://localhost:8233 | — |
| MinIO console | http://localhost:9001 | Configured in `.env` |
| Langfuse | http://localhost:3000 | Configured in `.env` |

Langfuse API keys are provisioned headlessly on first boot via `LANGFUSE_INIT_*`
in `docker-compose.yml`; configure unique values in `.env` before starting it.

```bash
# submit a run and watch it
RUN=$(curl -s -XPOST localhost:8000/v1/runs -H 'Content-Type: application/json' \
  -d '{"objective":"...","agent":"assistant"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['run_id'])")
curl -N localhost:8000/v1/runs/$RUN/stream
curl -s localhost:8000/v1/runs/$RUN/artifacts
```

`agents.yaml` and `prompts/` are **bind-mounted**, so editing an agent needs only
a restart, not a rebuild. Everything else needs `docker compose build api`
(the `worker` reuses that image — only `api` has a `build:` stanza, deliberately;
declaring it twice makes both services race to write the same tag).

---

## 4. Model providers — current state

`.env` is gitignored and holds the keys. `.env.example` is the committed template.

| Provider | State |
|---|---|
| **Ollama** (host) | `qwen3.5:9b` (default), `qwen3.5:2b`, `qwen3.8`. Unlimited, correct, but slow (1–4 min/run). Reached from containers via `host.docker.internal`. |
| **Gemini** | `gemini-flash-lite-latest` and `gemini-flash-latest` work with tools and are **fast** (~11s for a delegating run). `gemini-3-flash-preview` works but its free daily quota is currently exhausted. `gemini-2.5-flash*` are retired. |
| **OpenRouter** | Key present; free tier is 50 requests/day and was exhausted. `nvidia/nemotron-3-super-120b-a12b:free` and several others support tools. |

Switching provider is one env var; no code changes.

**Embeddings — settled.** `ollama pull nomic-embed-text` (768-dim, ~274MB) is
done and **verified working** through `/api/embed`. The earlier claim in this
project that "Ollama has embeddings disabled" was **wrong**: that error comes
from asking a *chat* model to embed. `.env` now uses it. The shipped default in
`.env.example` stays `hashing` so a fresh clone works with nothing pulled.

---

## 5. Conventions that matter

- **pydantic v2 frozen models everywhere.** `ConfigDict(frozen=True)`; tagged
  unions via `Field(discriminator="kind")`. Temporal serialises them with
  `pydantic_data_converter`.
- **Adapters**: a base class holding the wire logic plus thin subclasses plus a
  `PROVIDERS`-style dict registry (`adapters/model/registry.py` is the template).
  Optional SDKs are imported *inside* `create()` and raise `ConfigurationError`,
  so an uninstalled extra never breaks collection.
- **Tools** are built by the `@tool` decorator; JSON Schema comes from type hints
  via pydantic, and the same model validates arguments. Never hand-write a schema.
- **Contract suites** (`skynet/testing/contracts.py`) are the project's core
  claim: the same assertions run against fakes *and* real adapters. They have
  already caught two real bugs. Add to them rather than writing adapter-specific
  tests.
- **No mocks anywhere.** Every port has an in-memory fake in
  `skynet/testing/fakes.py`; behaviour is asserted against the event log.
  `FixedClock` + `SequentialIdGenerator` make that log byte-stable.
- Integration tests are marked `@pytest.mark.integration` and excluded by default
  (`addopts = "-m 'not integration'"`), and they **clean up after themselves** —
  Docker containers outlive the process.

---

## 6. Gotchas found the hard way

Each of these cost real debugging time. Do not rediscover them.

| Gotcha | Detail |
|---|---|
| **Verify a patch actually applied** | A `str.replace`-based edit silently matched nothing after `ruff format` had reflowed the target onto one line. "The command succeeded" ≠ "the change landed." Assert the pattern matched, and check the running container, not just the file. |
| **Temporal activities called by name need `result_type=`** | Otherwise you get raw dicts, not models. Bit me on every activity. |
| **`WorkflowExecutionStatus` is an `IntEnum`** | `str(status)` gives `'2'`, not `'COMPLETED'`. Map on `.name`. |
| **Docker sandbox needs a workspace per *run*, not per acquire** | Each tool acquires independently; a container per acquisition silently discards everything the previous tool wrote. Containers are adopted back by label after a worker restart. |
| **Shell pipelines mask exit codes** | `base64 f \| tr -d '\n'` returns `tr`'s status, so missing files read as empty. |
| **Docker `put_archive`/`get_archive` fail on read-only rootfs and tmpfs mounts** | Both file ops go through `exec` + base64 instead. |
| **Langfuse needs `LANGFUSE_S3_*_REGION`** | Without it the AWS SDK throws "Region is missing", which becomes an unhandled rejection that restarts langfuse-web in a loop and silently drops every trace. |
| **Langfuse v4 SDK ≠ v3** | It's OTel-based: `start_as_current_observation(as_type=...)`, not `start_span`. Ambient OTel context must win over an explicit trace context, or nesting flattens. |
| **Ollama returns answers in a separate `reasoning` field** | `content` is empty for thinking models. `Message.text` falls back to it. |
| **Gemini 3 requires echoing `thought_signature`** | Function calls carry an opaque token that must be returned verbatim or the next turn 400s. Carried in `ToolCall.provider_data`. |
| **Flush the tracer before the process exits** | Batched spans are lost otherwise. |
| **`pkill -f <pattern>` matches its own command line** | It killed the shell mid-script. Kill by port/PID instead. |

---

## 7. Known limitations (accurate, in README)

- **Replay is guaranteed within a schema version.** There is no cross-version
  migration path: changing a persisted domain model means clearing the event
  log. New fields must always arrive with defaults.
- **Consolidation costs a model call per run**, reflection one per interval.
  Point `memory.model` at a cheaper model; both default off, and `agents.yaml`
  enables them for the shipped agents.
- **The default embedder is not semantic.** `hashing` is deterministic and
  offline so the stack works with nothing pulled, but `"car"` will not find
  `"vehicle"`. `SKYNET_EMBEDDING__BACKEND=ollama` fixes that.
- **A failed run does not consolidate.** The gate is on completion, so a run
  that exhausted `max_iterations` writes nothing down. Defensible (a failed run
  may have learned wrong things) but worth knowing.
- Sandbox workspaces are tmpfs; they survive a worker restart (adopted by label)
  but not container removal.
- The shared event log is one JSONL file per run on a Docker volume —
  deliberately unglamorous and correct; Redis Streams or Postgres would replace
  it behind the same two ports.
- Small free models wander; budgets stop them, but output quality tracks the model.

---

## 8. Memory, as built

Read `README.md`'s *Memory* section first; this is the operational supplement.

| | |
|---|---|
| Store | `memory-db` (pgvector/pg16), host port **5433**, database `skynet_memory` |
| Schema | numbered `.sql` under `src/skynet/adapters/memory/migrations/`, applied by the `memory-init` one-shot |
| Embeddings | `SKYNET_EMBEDDING__BACKEND` — `hashing` (default, offline) · `ollama` · `gemini` · `null` |
| Policy | the `memory:` block per agent in `agents.yaml` |

```bash
psql "$SKYNET_MEMORY__DSN"
  SELECT content, kind, valid_to, superseded_by FROM memories ORDER BY created_at;
  SELECT count(*) FROM memory_vectors;      -- rows with an embedding
```

**Gotchas specific to this subsystem**, each of which cost time:

| Gotcha | Detail |
|---|---|
| A generated `tsvector` column must be immutable | `to_tsvector('english'::regconfig, …)` — the bare `'english'` literal picks the STABLE overload, and `array_to_string` is STABLE too, so tags cannot be folded in. They have their own GIN index. |
| Migrations cannot run through the pool | The pool's `configure` calls `register_vector_async`, which needs the `vector` type to already exist. A failure there leaves the connection in a transaction, psycopg discards it, and the pool retries **forever** — it presents as a hang, not an error. Migrations run on a standalone connection first. |
| Pooled connections need autocommit | The pool rolls back on return, so a bare `UPDATE` outside an explicit transaction is silently undone. `invalidate` and `delete` reported success and changed nothing. |
| Compaction's guard is about *usefulness*, not size | It refuses to fold when the tail alone still exceeds the budget — that case is the trimmer's. A run can be well over budget and correctly not compact. |
| Langfuse from the host needs a different hostname | `.env` has `SKYNET_LANGFUSE__HOST=http://langfuse-web:3000`, a Compose-internal name. Host-side runs must override it with `http://localhost:3000` or traces are silently dropped. |

### If you extend it

The seams that are already open: `MemoryMaintenance.purge_expired` exists and
nothing calls it (a CLI or a cron job is the intended home); `RetrievalComponents`
carries the raw arm scores so an alternative fusion is a change to
`memory_ranking` alone; and `ScopeLevel` is an enum plus a bonus table, so a new
scope is two lines and a grant.
