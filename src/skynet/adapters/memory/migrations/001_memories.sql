-- Memory storage: the records themselves.
--
-- `scope_kind` and `scope_key` are stored alongside the rendered `namespace`
-- because a rendered string cannot answer "everything in any project scope",
-- and re-parsing it later is a lie waiting to happen.
--
-- Bi-temporal by construction: `created_at` is when a fact was learned,
-- `valid_from`/`valid_to` when it held. Contradiction stamps `valid_to` and
-- sets `superseded_by`; nothing is ever deleted by the agent.

CREATE TABLE IF NOT EXISTS memories (
    id            text PRIMARY KEY,
    namespace     text        NOT NULL,
    scope_kind    text        NOT NULL,
    scope_key     text        NOT NULL,
    kind          text        NOT NULL DEFAULT 'semantic',
    content       text        NOT NULL,
    tags          text[]      NOT NULL DEFAULT '{}',
    metadata      jsonb       NOT NULL DEFAULT '{}'::jsonb,
    importance    real        NOT NULL DEFAULT 0.5,
    source_run_id text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    valid_from    timestamptz,
    valid_to      timestamptz,
    superseded_by text,
    expires_at    timestamptz,
    -- `'english'::regconfig` explicitly, because the two-argument form is the
    -- immutable one and a generated column may use nothing else. Tags are
    -- deliberately not folded in here: `array_to_string` is only STABLE, and
    -- tags already have their own GIN index and their own `&&` filter.
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english'::regconfig, content)
    ) STORED
);

CREATE INDEX IF NOT EXISTS memories_search_idx ON memories USING gin (search_vector);

-- Equality before range, so the index is usable end to end rather than only
-- for its first column.
CREATE INDEX IF NOT EXISTS memories_scope_idx
    ON memories (namespace, kind, valid_from DESC);

CREATE INDEX IF NOT EXISTS memories_scope_kind_idx ON memories (scope_kind, scope_key);

-- Partial, and therefore tiny: most rows have neither column set.
CREATE INDEX IF NOT EXISTS memories_valid_to_idx
    ON memories (valid_to) WHERE valid_to IS NOT NULL;
CREATE INDEX IF NOT EXISTS memories_expires_idx
    ON memories (expires_at) WHERE expires_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS memories_tags_idx ON memories USING gin (tags);
CREATE INDEX IF NOT EXISTS memories_metadata_idx
    ON memories USING gin (metadata jsonb_path_ops);
