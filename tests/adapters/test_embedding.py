"""The offline embedder, which is what keeps the suite hermetic."""

from __future__ import annotations

import math
import subprocess
import sys

import pytest

from skynet.adapters.embedding.hashing import HashingEmbedder, NullEmbedder
from skynet.adapters.embedding.registry import create_embedder
from skynet.domain.errors import ConfigurationError


def cosine(a, b) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


async def embed(*texts: str, dimensions: int = 768):
    return await HashingEmbedder(dimensions=dimensions).embed(list(texts))


async def test_vectors_have_the_configured_width() -> None:
    """A mismatch here presents later as total recall loss with no error."""
    vectors = await embed("anything", dimensions=256)
    assert len(vectors[0]) == 256


async def test_the_same_text_always_embeds_the_same_way() -> None:
    first, second = await embed("pangolins eat ants", "pangolins eat ants")
    assert first == second


async def test_vectors_are_stable_across_processes() -> None:
    """`blake2b`, not `hash()`.

    `hash()` is salted per process by PYTHONHASHSEED, so hash-based vectors
    would differ between test runs and between the API and worker containers
    reading the same table. That single choice is the difference between
    deterministic and haunted -- so it is asserted in a *separate interpreter*
    with a different seed, which is the only way to catch a regression to
    `hash()`.
    """
    script = (
        "import asyncio;"
        "from skynet.adapters.embedding.hashing import HashingEmbedder;"
        "v=asyncio.run(HashingEmbedder().embed(['pangolins eat ants']));"
        "print(round(sum(v[0]), 12))"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "12345")
    }
    assert len(outputs) == 1, f"vectors depend on PYTHONHASHSEED: {outputs}"


async def test_a_morphological_variant_scores_well_above_an_unrelated_string() -> None:
    """Character trigrams buy the partial credit a token-only hash scores at
    exactly zero -- the one behaviour a hybrid retrieval test needs."""
    pangolins, pangolin, volcanoes = await embed(
        "pangolins eat ants", "pangolin eats an ant", "volcanoes erupt violently"
    )
    assert cosine(pangolins, pangolin) > 4 * cosine(pangolins, volcanoes)


async def test_unrelated_text_is_close_to_orthogonal() -> None:
    a, b = await embed("the deploy key lives in the vault", "volcanoes erupt violently")
    assert abs(cosine(a, b)) < 0.2


async def test_empty_text_does_not_explode() -> None:
    assert len((await embed(""))[0]) == 768


async def test_the_null_embedder_yields_no_vectors() -> None:
    """Not a degenerate case: it is the right choice when keywords suffice."""
    assert await NullEmbedder().embed(["anything"]) == ((),)


def test_the_registry_reports_unknown_names_usefully() -> None:
    with pytest.raises(ConfigurationError, match="hashing"):
        create_embedder("does-not-exist")


def test_ollama_uses_the_native_endpoint() -> None:
    """`/api/embed`, not the OpenAI-compatible surface -- and the base URL
    commonly configured for chat ends in `/v1`, which would 404."""
    embedder = create_embedder("ollama", base_url="http://localhost:11434/v1")
    assert embedder._url() == "http://localhost:11434/api/embed"
