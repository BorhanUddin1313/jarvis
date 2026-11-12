from abc import ABC, abstractmethod

Message = dict[str, str]


class LLMError(Exception):
    """Raised when the language model can't produce a reply."""


class LLM(ABC):
    """Anything that can continue a conversation."""

    @abstractmethod
    def chat(self, messages: list[Message]) -> str:
        """Send the conversation so far and return the model's reply."""