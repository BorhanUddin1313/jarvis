import re
from abc import ABC, abstractmethod


def words(text: str) -> set[str]:
    """Lowercase the text and return its words, ignoring punctuation."""
    return set(re.findall(r"[a-z']+", text.lower()))


class Skill(ABC):
    """A single capability JARVIS has, like telling the time."""

    name: str = "unnamed"
    description: str = ""
    parameters: dict = {"type": "object", "properties": {}}
    direct: bool = False  # True if the tool's output should be the final reply

    @abstractmethod
    def matches(self, text: str) -> bool:
        """Return True if this skill should handle the text (keyword mode)."""

    @abstractmethod
    def handle(self, text: str) -> str:
        """Do the work and return the reply (keyword mode)."""

    def run(self, arguments: dict) -> str:
        """Run the skill as an LLM tool. By default, arguments are ignored."""
        return self.handle("")

    def as_tool(self) -> dict:
        """Describe this skill in the format LLMs expect for tools."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }