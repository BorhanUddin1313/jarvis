import httpx

from jarvis.llm.base import LLM, LLMError, Message


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

    def chat(self, messages: list[Message]) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False}
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        return response.json()["message"]["content"].strip()