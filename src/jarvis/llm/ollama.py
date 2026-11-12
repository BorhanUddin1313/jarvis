import json

import httpx

from jarvis.llm.base import LLM, LLMError, Message, Reply, ToolCall


class OllamaLLM(LLM):
    """Talks to a local Ollama server over HTTP."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        client: httpx.Client | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> Reply:
        payload = {"model": self.model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools

        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        message = response.json()["message"]
        calls = [self._parse_call(call) for call in message.get("tool_calls") or []]
        return Reply(content=(message.get("content") or "").strip(), tool_calls=calls)

    @staticmethod
    def _parse_call(call: dict) -> ToolCall:
        function = call["function"]
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        return ToolCall(name=function["name"], arguments=arguments)