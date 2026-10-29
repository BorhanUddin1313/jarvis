from jarvis.router import FALLBACK, Router
from jarvis.skills.base import Skill


class FakeSkill(Skill):
    """A skill that matches one trigger word and returns a fixed reply."""

    def __init__(self, name: str, trigger: str, reply: str) -> None:
        self.name = name
        self.description = f"fake {name}"
        self._trigger = trigger
        self._reply = reply

    def matches(self, text: str) -> bool:
        return self._trigger in text.lower()

    def handle(self, text: str) -> str:
        return self._reply


def test_routes_to_matching_skill():
    router = Router([FakeSkill("a", "apple", "got apple")])
    assert router.route("I want an apple") == "got apple"


def test_first_matching_skill_wins():
    router = Router([
        FakeSkill("first", "fruit", "first reply"),
        FakeSkill("second", "fruit", "second reply"),
    ])
    assert router.route("fruit please") == "first reply"


def test_unknown_input_gets_fallback():
    router = Router([FakeSkill("a", "apple", "got apple")])
    assert router.route("something else") == FALLBACK


def test_help_lists_skills():
    router = Router([FakeSkill("a", "apple", "got apple")])
    assert "a: fake a" in router.route("help")