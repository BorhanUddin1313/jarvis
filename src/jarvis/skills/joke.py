import random
from typing import Callable

from jarvis.skills.base import Skill, words

TRIGGERS = {"joke", "funny", "laugh", "humor", "pun", "comedy", "jokes", "fun", "comedy"}

JOKES = {
    "Why did the chicken cross the road?": "To get to the other side!",
    "Why don't scientists trust atoms?": "Because they make up everything!",
    "Why do cows wear bells?": "Because their horns don't work!",
    "Do you know what a matter-baby is?": "Nothing baby, what is the matter with you?",}


class JokeSkill(Skill):
    name = "joke"
    description = "Tell a joke"

    def __init__(self, choose: Callable = random.choice) -> None:
        self._choose = choose

    def matches(self, text: str) -> bool:
        return bool(words(text) & TRIGGERS)

    def handle(self, text: str) -> str:
        setup, punchline =self._choose(list(JOKES.items()))
        return f"{setup} {punchline}"