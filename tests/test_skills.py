from datetime import datetime

from jarvis.skills.clock import ClockSkill
from jarvis.skills.greeting import GreetingSkill


def fixed_now() -> datetime:
    return datetime(2026, 10, 28, 15, 5)  # a Wednesday, 3:05 PM


def test_greeting_matches_hello():
    assert GreetingSkill().matches("Hello Jarvis!")


def test_greeting_ignores_words_containing_hi():
    assert not GreetingSkill().matches("this is a thing")


def test_clock_tells_time():
    assert ClockSkill(now=fixed_now).handle("what time is it") == "It's 3:05 PM."


def test_clock_tells_date():
    reply = ClockSkill(now=fixed_now).handle("what's the date")
    assert reply == "Today is Wednesday, October 28."