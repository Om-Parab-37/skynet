"""Temporal's status enum translated at the boundary.

Regression guard: WorkflowExecutionStatus is an IntEnum, so ``str()`` yields the
number. Mapping on the stringified enum silently reported every finished run as
still running.
"""

from __future__ import annotations

from enum import IntEnum

import pytest

temporalio = pytest.importorskip("temporalio")

from skynet.adapters.temporal.runner import workflow_status  # noqa: E402
from skynet.domain.run import AgentStatus  # noqa: E402


class FakeStatus(IntEnum):
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3
    CANCELED = 4
    TERMINATED = 5
    TIMED_OUT = 6


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (FakeStatus.RUNNING, AgentStatus.RUNNING),
        (FakeStatus.COMPLETED, AgentStatus.COMPLETED),
        (FakeStatus.FAILED, AgentStatus.FAILED),
        (FakeStatus.CANCELED, AgentStatus.CANCELLED),
        (FakeStatus.TERMINATED, AgentStatus.CANCELLED),
        (FakeStatus.TIMED_OUT, AgentStatus.FAILED),
    ],
)
def test_status_translation(raw, expected):
    assert workflow_status(raw) is expected


def test_stringified_int_enum_does_not_leak_through():
    """`str(FakeStatus.COMPLETED)` is '2'; the mapping must not rely on it."""
    assert str(FakeStatus.COMPLETED) in {"2", "FakeStatus.COMPLETED"}
    assert workflow_status(FakeStatus.COMPLETED) is AgentStatus.COMPLETED


def test_unknown_status_defaults_to_running():
    assert workflow_status("SOMETHING_NEW") is AgentStatus.RUNNING


def test_real_temporal_enum_is_covered():
    from temporalio.client import WorkflowExecutionStatus

    for member in WorkflowExecutionStatus:
        assert isinstance(workflow_status(member), AgentStatus)


def test_reasoning_field_variants_are_all_recognised():
    """Providers disagree on the field name; the adapter must accept each."""
    from skynet.adapters.model.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider(base_url="http://unused")
    for field in ("reasoning", "reasoning_content", "thinking"):
        parsed = provider.parse_response(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "", field: "why"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        assert parsed.message.reasoning == "why", field
        assert parsed.message.text == "why"


def test_content_is_preferred_over_reasoning_in_the_adapter():
    from skynet.adapters.model.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider(base_url="http://unused")
    parsed = provider.parse_response(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "answer", "reasoning": "why"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    assert parsed.message.text == "answer"


def test_provider_data_on_tool_calls_round_trips():
    """Some providers reject a conversation that loses their continuation token.

    Gemini 3 attaches a `thought_signature` to each function call and refuses the
    next turn if it is not returned verbatim, so the adapter must carry opaque
    provider data through without interpreting it.
    """
    from skynet.adapters.model.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider(base_url="http://unused")
    signature = {"google": {"thought_signature": "opaque-token"}}

    parsed = provider.parse_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "extra_content": signature,
                                "function": {"name": "calc", "arguments": '{"x":1}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    call = parsed.message.tool_calls[0]
    assert call.provider_data["extra_content"] == signature

    encoded = provider._encode_message(parsed.message)
    assert encoded["tool_calls"][0]["extra_content"] == signature
    assert encoded["tool_calls"][0]["function"]["name"] == "calc"


def test_tool_calls_without_provider_data_encode_cleanly():
    """No stray keys for providers that do not use them."""
    from skynet.adapters.model.openai_compat import OpenAICompatProvider
    from skynet.domain.ids import CallId
    from skynet.domain.messages import ToolCall, assistant

    provider = OpenAICompatProvider(base_url="http://unused")
    message = assistant(tool_calls=(ToolCall(id=CallId("c1"), name="calc", arguments={}),))
    encoded = provider._encode_message(message)
    assert "extra_content" not in encoded["tool_calls"][0]
