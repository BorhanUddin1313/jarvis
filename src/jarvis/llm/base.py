from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]


@dataclass
class ToolCall:
    """The LLM asking JARVIS to run a tool."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reply:
    """What the LLM sent back: text, tool calls, or both."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMError(Exception):
    """Raised when the language model can't produce a reply."""


class LLM(ABC):
    """Anything that can continue a conversation."""

    @abstractmethod
    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> Reply:
        """Send the conversation and available tools, return the model's reply."""