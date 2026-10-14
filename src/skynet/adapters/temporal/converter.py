"""Temporal payload conversion.

Every value crossing the workflow boundary is a pydantic model, so Temporal's
own pydantic converter handles serialisation. Nothing bespoke is needed, and
nothing in the domain learns that Temporal exists.
"""

from __future__ import annotations

from temporalio.contrib.pydantic import pydantic_data_converter

DATA_CONVERTER = pydantic_data_converter

__all__ = ["DATA_CONVERTER"]
