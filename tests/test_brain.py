from jarvis.brain import OFFLINE_REPLY, Brain
from jarvis.llm.base import LLM, LLMError


class FakeLLM(LLM):
    """Returns a fixed reply and records every conversation it was sent."""

    def __init__(self, reply: str = "fake reply", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise LLMError("offline")
        return self.reply


def test_returns_llm_reply():
    assert Brain(FakeLLM("hi there")).ask("hello") == "hi there"


def test_sends_system_prompt_first():
    llm = FakeLLM()
    Brain(llm).ask("hello")
    assert llm.calls[0][0]["role"] == "system"


def test_remembers_previous_turns():
    llm = FakeLLM("answer")
    brain = Brain(llm)
    brain.ask("first")
    brain.ask("second")
    contents = [m["content"] for m in llm.calls[1][1:]]
    assert contents == ["first", "answer", "second"]


def test_history_is_trimmed():
    llm = FakeLLM()
    brain = Brain(llm, max_turns=1)
    brain.ask("a")
    brain.ask("b")
    brain.ask("c")
    contents = [m["content"] for m in llm.calls[-1][1:]]
    assert contents == ["b", "fake reply", "c"]


def test_offline_llm_gives_friendly_reply():
    assert Brain(FakeLLM(fail=True)).ask("hello") == OFFLINE_REPLY