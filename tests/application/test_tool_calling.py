"""Emulated tool calling: parsing is where weak models are won or lost."""

from __future__ import annotations

from skynet.application.tool_calling import (
    EmulatedToolCallingProvider,
    JsonBlockCodec,
    wrap_for_tool_calling,
)
from skynet.domain.messages import user
from skynet.domain.models import FinishReason, ModelParams, ModelRequest
from skynet.ports.model import ProviderCapabilities
from skynet.testing.builders import says
from skynet.testing.fakes import EchoTool, ScriptedModelProvider, SequentialIdGenerator

CODEC = JsonBlockCodec()


def parse(text: str):
    return CODEC.parse(text, SequentialIdGenerator())


def test_clean_fenced_block_parses():
    parsed = parse(
        '```json\n{"thought": "use it", "tool": "echo", "arguments": {"text": "hi"}}\n```'
    )
    assert parsed.error is None
    assert parsed.calls[0].name == "echo"
    assert parsed.calls[0].arguments == {"text": "hi"}
    assert parsed.text == "use it"


def test_block_surrounded_by_prose_parses():
    parsed = parse('Sure, let me do that.\n```json\n{"tool": "echo", "arguments": {}}\n```\nDone!')
    assert parsed.error is None
    assert parsed.calls[0].name == "echo"


def test_bare_json_without_a_fence_parses():
    parsed = parse('{"tool": "echo", "arguments": {"text": "x"}}')
    assert parsed.error is None
    assert parsed.calls[0].name == "echo"


def test_missing_tool_field_is_an_error():
    assert parse('```json\n{"arguments": {}}\n```').error is not None


def test_malformed_json_is_an_error():
    assert parse('```json\n{"tool": "echo", ,,,}\n```').error is not None


def test_no_json_at_all_is_an_error():
    assert parse("I think the answer is 42.").error is not None


def test_non_object_arguments_is_an_error():
    assert parse('{"tool": "echo", "arguments": "hi"}').error is not None


def test_augment_moves_tools_into_the_prompt():
    request = ModelRequest(
        messages=(user("go"),), params=ModelParams(model="m"), tools=(EchoTool().spec,)
    )
    augmented = CODEC.augment(request)

    assert augmented.tools == ()
    assert "echo" in augmented.messages[-1].content
    assert "final_answer" in augmented.messages[-1].content


async def test_emulated_provider_reports_native_tools_and_no_parallelism():
    inner = ScriptedModelProvider([], capabilities=ProviderCapabilities(native_tools=False))
    wrapped = EmulatedToolCallingProvider(inner, CODEC, SequentialIdGenerator())

    assert wrapped.capabilities.native_tools is True
    assert wrapped.capabilities.emulated_tools is True
    assert wrapped.capabilities.parallel_tool_calls is False


async def test_emulated_provider_turns_text_into_a_tool_call():
    inner = ScriptedModelProvider(
        [says('```json\n{"tool": "echo", "arguments": {"text": "hi"}}\n```')],
        capabilities=ProviderCapabilities(native_tools=False),
    )
    wrapped = EmulatedToolCallingProvider(inner, CODEC, SequentialIdGenerator())
    request = ModelRequest(
        messages=(user("go"),), params=ModelParams(model="m"), tools=(EchoTool().spec,)
    )
    response = await wrapped.generate(request)

    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.message.tool_calls[0].name == "echo"


async def test_unparseable_output_surfaces_as_a_parse_error():
    inner = ScriptedModelProvider(
        [says("no json here")], capabilities=ProviderCapabilities(native_tools=False)
    )
    wrapped = EmulatedToolCallingProvider(inner, CODEC, SequentialIdGenerator())
    request = ModelRequest(
        messages=(user("go"),), params=ModelParams(model="m"), tools=(EchoTool().spec,)
    )
    response = await wrapped.generate(request)

    assert response.finish_reason is FinishReason.PARSE_ERROR
    assert response.parse_error


def test_native_providers_are_not_wrapped():
    native = ScriptedModelProvider([], capabilities=ProviderCapabilities(native_tools=True))
    assert wrap_for_tool_calling(native, SequentialIdGenerator()) is native


def test_non_native_providers_are_wrapped():
    weak = ScriptedModelProvider([], capabilities=ProviderCapabilities(native_tools=False))
    assert isinstance(
        wrap_for_tool_calling(weak, SequentialIdGenerator()), EmulatedToolCallingProvider
    )
