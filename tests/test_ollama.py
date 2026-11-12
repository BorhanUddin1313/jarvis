import json

import httpx
import pytest

from jarvis.llm.base import LLMError, ToolCall
from jarvis.llm.ollama import OllamaLLM

MESSAGES = [{"role": "user", "content": "hi"}]


def make_llm(handler) -> OllamaLLM:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    return OllamaLLM(model="test-model", client=client)


def ok(message: dict):
    return lambda request: httpx.Response(200, json={"message": message})


def test_returns_message_content():
    reply = make_llm(ok({"role": "assistant", "content": " Hello! "})).chat(MESSAGES)
    assert reply.content == "Hello!"
    assert reply.tool_calls == []


def test_sends_model_tools_and_disables_streaming():
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    tools = [{"type": "function", "function": {"name": "clock"}}]
    make_llm(handler).chat(MESSAGES, tools=tools)
    assert sent["model"] == "test-model"
    assert sent["stream"] is False
    assert sent["tools"] == tools


def test_parses_tool_calls():
    message = {
        "content": "",
        "tool_calls": [{"function": {"name": "clock", "arguments": {"what": "date"}}}],
    }
    reply = make_llm(ok(message)).chat(MESSAGES)
    assert reply.tool_calls == [ToolCall("clock", {"what": "date"})]


def test_server_error_raises_llm_error():
    llm = make_llm(lambda request: httpx.Response(500))
    with pytest.raises(LLMError):
        llm.chat(MESSAGES)