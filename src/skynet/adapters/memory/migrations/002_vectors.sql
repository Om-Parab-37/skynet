-- Embeddings live in a side table, deliberately.
--
-- pgvector's HNSW and IVFFlat indexes cap at 2000 dimensions, and putting
-- `vector(N)` on `memories` would weld the embedding dimension to the row: a
-- change of embedder would mean dropping the column *and the memories with it*.
-- Here, re-embedding is `TRUNCATE memory_vectors` plus a backfill, and no memory
-- is ever lost to it.
--
-- HNSW rather than IVFFlat, deliberately. IVFFlat must be trained on
-- representative data with `lists` tuned to the row count -- but a memory store
-- starts empty and grows monotonically, which is precisely IVFFlat's failure
-- mode: recall silently rots until somebody remembers to REINDEX. HNSW builds
-- incrementally and holds recall as the table grows.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id text PRIMARY KEY REFERENCES memories (id) ON DELETE CASCADE,
    model     text NOT NULL,
    dim       int  NOT NULL,
    embedding vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS memory_vectors_hnsw_idx
    ON memory_vectors USING hnsw (embedding vector_cosine_ops);
