"""Turn a plain async function into a ``Tool``.

The JSON Schema the model sees is derived from the function's type hints by
pydantic, and the same model validates incoming arguments. Nobody hand-writes a
schema, and the schema can never drift from the implementation.

Validation failures are returned as ``ToolResult`` observations rather than
raised, so a model that guesses an argument wrong simply gets told and tries
again -- by far the highest-leverage robustness measure for smaller models.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, get_type_hints

from pydantic import BaseModel, ValidationError, create_model

from skynet.domain.errors import ErrorInfo
from skynet.domain.messages import ToolCall
from skynet.domain.tools import ToolContext, ToolKind, ToolResult, ToolSpec, ToolStatus

ToolFunction = Callable[..., Awaitable[Any]]

_CONTEXT_PARAM = "ctx"


def build_arguments_model(func: ToolFunction, name: str) -> type[BaseModel]:
    """Derive a pydantic model from the function signature, ignoring ``ctx``."""
    signature = inspect.signature(func)
    hints = get_type_hints(func)
    fields: dict[str, Any] = {}

    for param_name, param in signature.parameters.items():
        if param_name in {"self", _CONTEXT_PARAM}:
            continue
        annotation = hints.get(param_name, str)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (annotation, default)

    return create_model(f"{name.title().replace('_', '')}Arguments", **fields)


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


class FunctionTool:
    """Adapts a function to the ``Tool`` port."""

    def __init__(self, func: ToolFunction, spec: ToolSpec, arguments: type[BaseModel]) -> None:
        self._func = func
        self._spec = spec
        self._arguments = arguments
        self._wants_context = _CONTEXT_PARAM in inspect.signature(func).parameters

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        try:
            parsed = self._arguments.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolResult(
                call_id=call.id,
                name=self._spec.name,
                status=ToolStatus.ERROR,
                content=f"ERROR: invalid arguments: {_explain(exc)}",
                error=ErrorInfo.of("invalid_arguments", _explain(exc)),
            )

        kwargs: dict[str, Any] = parsed.model_dump()
        if self._wants_context:
            kwargs[_CONTEXT_PARAM] = ctx

        outcome = await self._func(**kwargs)
        return _as_result(outcome, call, self._spec.name)


def _explain(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or 'argument'}: {err['msg']}"
        for err in exc.errors()
    )


def _as_result(outcome: Any, call: ToolCall, name: str) -> ToolResult:
    """Let tools return a ToolResult, a string, or any JSON-able value."""
    if isinstance(outcome, ToolResult):
        return outcome
    if isinstance(outcome, Mapping):
        content = str(outcome.get("content", outcome))
        return ToolResult(
            call_id=call.id,
            name=name,
            status=ToolStatus.OK,
            content=content,
            data=dict(outcome),
        )
    return ToolResult(call_id=call.id, name=name, status=ToolStatus.OK, content=str(outcome))


def tool(
    name: str | None = None,
    *,
    description: str | None = None,
    kind: ToolKind = ToolKind.STANDARD,
    required_permissions: frozenset[str] | set[str] = frozenset(),
    requires_approval: bool = False,
    timeout_s: float = 60.0,
    idempotent: bool = False,
) -> Callable[[ToolFunction], FunctionTool]:
    """Decorator producing a ``FunctionTool`` with a generated schema."""

    def decorate(func: ToolFunction) -> FunctionTool:
        tool_name = name or func.__name__
        arguments = build_arguments_model(func, tool_name)
        spec = ToolSpec(
            name=tool_name,
            description=description or inspect.getdoc(func) or tool_name,
            parameters=json_schema_for(arguments),
            kind=kind,
            required_permissions=frozenset(required_permissions),
            requires_approval=requires_approval,
            timeout_s=timeout_s,
            idempotent=idempotent,
        )
        return FunctionTool(func, spec, arguments)

    return decorate


__all__ = [
    "FunctionTool",
    "ToolFunction",
    "build_arguments_model",
    "json_schema_for",
    "tool",
]
