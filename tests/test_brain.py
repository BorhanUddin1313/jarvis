from jarvis.brain import MAX_TOOL_ROUNDS, OFFLINE_REPLY, STUCK_REPLY, Brain
from jarvis.llm.base import LLM, LLMError, Reply, ToolCall
from jarvis.skills.base import Skill
from jarvis.skills.clock import ClockSkill


class FakeLLM(LLM):
    """Returns scripted replies in order (repeating the last) and records each call."""

    def __init__(self, replies=None, fail=False):
        self.replies = list(replies or [Reply("fake reply")])
        self.fail = fail
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        if self.fail:
            raise LLMError("offline")
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class EchoSkill(Skill):
    name = "echo"
    description = "Repeat some text"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def matches(self, text):
        return False

    def handle(self, text):
        return f"echo: {text}"

    def run(self, arguments):
        return self.handle(arguments.get("text", ""))


def contents(call):
    return [m["content"] for m in call["messages"][1:]]


def test_returns_llm_reply():
    assert Brain(FakeLLM([Reply("hi there")])).ask("hello") == "hi there"


def test_sends_system_prompt_first():
    llm = FakeLLM()
    Brain(llm).ask("hello")
    assert llm.calls[0]["messages"][0]["role"] == "system"


def test_remembers_previous_turns():
    llm = FakeLLM([Reply("answer")])
    brain = Brain(llm)
    brain.ask("first")
    brain.ask("second")
    assert contents(llm.calls[1]) == ["first", "answer", "second"]


def test_history_is_trimmed():
    llm = FakeLLM()
    brain = Brain(llm, max_turns=1)
    brain.ask("a")
    brain.ask("b")
    brain.ask("c")
    assert contents(llm.calls[-1]) == ["b", "fake reply", "c"]


def test_offline_llm_gives_friendly_reply():
    assert Brain(FakeLLM(fail=True)).ask("hello") == OFFLINE_REPLY


def test_offline_llm_uses_backup():
    brain = Brain(FakeLLM(fail=True), offline=lambda text: f"backup: {text}")
    assert brain.ask("joke") == "backup: joke"


def test_sends_tool_descriptions():
    llm = FakeLLM()
    Brain(llm, tools=[EchoSkill()]).ask("hello")
    assert llm.calls[0]["tools"][0]["function"]["name"] == "echo"


def test_runs_requested_tool_and_returns_final_answer():
    llm = FakeLLM([
        Reply(tool_calls=[ToolCall("echo", {"text": "hi"})]),
        Reply("done"),
    ])
    reply = Brain(llm, tools=[EchoSkill()]).ask("say hi")
    assert reply == "done"
    tool_message = llm.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["content"] == "echo: hi"


def test_unknown_tool_is_reported_to_llm():
    llm = FakeLLM([Reply(tool_calls=[ToolCall("missing")]), Reply("ok")])
    Brain(llm, tools=[EchoSkill()]).ask("do something")
    assert llm.calls[1]["messages"][-1]["content"].startswith("Error")


def test_gives_up_after_too_many_tool_rounds():
    llm = FakeLLM([Reply(tool_calls=[ToolCall("echo", {"text": "again"})])])
    reply = Brain(llm, tools=[EchoSkill()]).ask("loop forever")
    assert reply == STUCK_REPLY
    assert len(llm.calls) == MAX_TOOL_ROUNDS

def test_shortlist_hides_unrelated_tools():
    llm = FakeLLM()
    Brain(llm, tools=[ClockSkill()], shortlist=True).ask("thanks!")
    assert llm.calls[0]["tools"] is None


def test_shortlist_offers_matching_tools():
    llm = FakeLLM()
    Brain(llm, tools=[ClockSkill()], shortlist=True).ask("what time is it")
    assert llm.calls[0]["tools"][0]["function"]["name"] == "clock"

def test_recovers_tool_call_written_as_text():
    llm = FakeLLM([Reply('{"name": "echo", "parameters": {"text": "hi"}}'), Reply("done")])
    reply = Brain(llm, tools=[EchoSkill()]).ask("say hi")
    assert reply == "done"
    assert llm.calls[1]["messages"][-1]["content"] == "echo: hi"


def test_recovers_tool_call_from_broken_json():
    llm = FakeLLM([Reply('{"name":"echo","parameters":{"_: "oops"}}'), Reply("done")])
    Brain(llm, tools=[EchoSkill()]).ask("say hi")
    assert llm.calls[1]["messages"][-1]["content"] == "echo: "


def test_normal_text_is_not_treated_as_tool_call():
    llm = FakeLLM([Reply("My favourite tool is echo.")])
    assert Brain(llm, tools=[EchoSkill()]).ask("hi") == "My favourite tool is echo."


class DirectEchoSkill(EchoSkill):
    name = "direct_echo"
    direct = True


def test_direct_tool_result_skips_second_llm_call():
    llm = FakeLLM([Reply(tool_calls=[ToolCall("direct_echo", {"text": "hi"})])])
    reply = Brain(llm, tools=[DirectEchoSkill()]).ask("say hi")
    assert reply == "echo: hi"
    assert len(llm.calls) == 1