import logging

from jarvis.llm.base import LLM, LLMError, Message

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
   "You are JARVIS, a personal assistant running on the user's own computer. "
    "You are a software project, not the character from the Iron Man films, "
    "so never mention Tony Stark or Marvel. "
    "You cannot see, hear, or sense the user's location or surroundings. "
    "If you don't know something, say so plainly instead of guessing. "
    "Answer clearly and briefly, in at most three sentences unless asked for more."
)

OFFLINE_REPLY = "I can't reach my language model right now. Is Ollama running?"


class Brain:
    """Answers free-form questions with an LLM and remembers recent turns."""

    def __init__(self, llm: LLM, max_turns: int = 10) -> None:
        self._llm = llm
        self._max_turns = max_turns
        self._history: list[Message] = []

    def ask(self, text: str) -> str:
        self._history.append({"role": "user", "content": text})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self._history]

        try:
            reply = self._llm.chat(messages)
        except LLMError as exc:
            logger.warning("LLM unavailable: %s", exc)
            self._history.pop()
            return OFFLINE_REPLY

        self._history.append({"role": "assistant", "content": reply})
        self._history = self._history[-self._max_turns * 2 :]
        return reply