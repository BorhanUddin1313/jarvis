import json

import httpx
import pytest

from jarvis.llm.base import LLMError
from jarvis.llm.ollama import OllamaLLM

MESSAGES = [{"role": "user", "content": "hi"}]


def make_llm(handler) -> OllamaLLM:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    return OllamaLLM(model="test-model", client=client)


def test_returns_message_content():
    def handler(request):
        return httpx.Response(200, json={"message": {"role": "assistant", "content": " Hello! "}})

    assert make_llm(handler).chat(MESSAGES) == "Hello!"


def test_sends_model_and_disables_streaming():
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    make_llm(handler).chat(MESSAGES)
    assert sent["model"] == "test-model"
    assert sent["stream"] is False


def test_server_error_raises_llm_error():
    def handler(request):
        return httpx.Response(500)

    with pytest.raises(LLMError):
        make_llm(handler).chat(MESSAGES)