import logging

from jarvis.skills.base import Skill

logger = logging.getLogger(__name__)

FALLBACK = "Sorry, I don't know how to do that yet. Type 'help' to see what I can do."


class Router:
    """Sends each command to the first skill that matches it."""

    def __init__(self, skills: list[Skill]) -> None:
        self.skills = skills

    def route(self, text: str) -> str:
        if text.strip().lower() == "help":
            return self._help()

        for skill in self.skills:
            if skill.matches(text):
                logger.debug("Routing to skill: %s", skill.name)
                return skill.handle(text)

        logger.debug("No skill matched")
        return FALLBACK

    def _help(self) -> str:
        lines = [f"- {skill.name}: {skill.description}" for skill in self.skills]
        return "I can:\n" + "\n".join(lines)