import re
from abc import ABC, abstractmethod


def words(text: str) -> set[str]:
    """Lowercase the text and return its words, ignoring punctuation."""
    return set(re.findall(r"[a-z']+", text.lower()))


class Skill(ABC):
    """A single capability JARVIS has, like telling the time."""

    name: str = "unnamed"
    description: str = ""

    @abstractmethod
    def matches(self, text: str) -> bool:
        """Return True if this skill should handle the text."""

    @abstractmethod
    def handle(self, text: str) -> str:
        """Do the work and return the reply."""