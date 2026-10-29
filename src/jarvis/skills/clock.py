from datetime import datetime
from typing import Callable

from jarvis.skills.base import Skill, words

TRIGGERS = {"time", "date", "day"}
DATE_WORDS = {"date", "day"}


class ClockSkill(Skill):
    name = "clock"
    description = "Tell the current time or date"

    def __init__(self, now: Callable[[], datetime] = datetime.now) -> None:
        self._now = now

    def matches(self, text: str) -> bool:
        return bool(words(text) & TRIGGERS)

    def handle(self, text: str) -> str:
        now = self._now()
        if words(text) & DATE_WORDS:
            return f"Today is {now:%A, %B} {now.day}."
        hour = now.hour % 12 or 12
        return f"It's {hour}:{now:%M %p}."