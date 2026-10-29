from jarvis.skills.base import Skill, words

GREETINGS = {"hello", "hi", "hey", "morning", "evening"}


class GreetingSkill(Skill):
    name = "greeting"
    description = "Say hello"

    def matches(self, text: str) -> bool:
        return bool(words(text) & GREETINGS)

    def handle(self, text: str) -> str:
        return "Hello. How can I help?"