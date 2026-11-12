import json
import logging
import re
from typing import Callable

from jarvis.llm.base import LLM, LLMError, Message, ToolCall
from jarvis.skills.base import Skill

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are JARVIS, a personal assistant running on the user's own computer. "
    "You share your name with the AI from the Iron Man films, but you are not that character. "
    "You may discuss Iron Man, Tony Stark, or Marvel if asked, but never pretend to be the character. "
    "You cannot see the user's location or surroundings, but you have tools "
    "that give you real, current information such as the time and date. "
    "When a tool returns a result, treat it as correct and include it in your reply, "
    "because the user cannot see tool results. "
    "Only use a tool when the request clearly needs it; otherwise answer directly. "
    "If you don't know something, say so plainly instead of guessing. "
    "Answer clearly and briefly, in at most three sentences unless asked for more."
)

OFFLINE_REPLY = "I can't reach my language model right now. Is Ollama running?"
STUCK_REPLY = "Sorry, I got stuck trying to answer that."
MAX_TOOL_ROUNDS = 3

# Matches text that starts like a JSON tool call, e.g. {"name": "joke", ...}
LEAKED_CALL = re.compile(r'^\s*\{.*?"name"\s*:\s*"(\w+)"', re.DOTALL)


class Brain:
    """Answers with an LLM, lets it call tools, and remembers recent turns."""

    def __init__(
        self,
        llm: LLM,
        tools: list[Skill] | None = None,
        offline: Callable[[str], str] | None = None,
        max_turns: int = 10,
        shortlist: bool = False,
    ) -> None:
        self._llm = llm
        self._tools = {tool.name: tool for tool in tools or []}
        self._offline = offline
        self._max_turns = max_turns
        self._shortlist = shortlist
        self._history: list[Message] = []

    def ask(self, text: str) -> str:
        self._history.append({"role": "user", "content": text})
        try:
            reply = self._answer(text)
        except LLMError as exc:
            logger.warning("LLM unavailable: %s", exc)
            self._history.pop()
            return self._offline(text) if self._offline else OFFLINE_REPLY

        self._history.append({"role": "assistant", "content": reply})
        self._history = self._history[-self._max_turns * 2 :]
        return reply

    def _tool_specs_for(self, text: str) -> list[dict] | None:
        """Pick which tools to offer the LLM for this question."""
        tools = list(self._tools.values())
        if self._shortlist:
            tools = [tool for tool in tools if tool.matches(text)]
        return [tool.as_tool() for tool in tools] or None

    def _answer(self, text: str) -> str:
        messages: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}, *self._history]
        specs = self._tool_specs_for(text)
        logger.debug("Offering tools: %s", [s["function"]["name"] for s in specs or []])

        for _ in range(MAX_TOOL_ROUNDS):
            reply = self._llm.chat(messages, tools=specs)
            calls = reply.tool_calls or self._recover_leaked_call(reply.content)
            if not calls:
                return reply.content

            results = [self._run_tool(call) for call in calls]

            if all(self._is_direct(call) for call in calls):
                logger.debug("Direct tool result, skipping second LLM call")
                return " ".join(results)

            messages.append({
                "role": "assistant",
                "content": reply.content if reply.tool_calls else "",
                "tool_calls": [
                    {"function": {"name": call.name, "arguments": call.arguments}}
                    for call in calls
                ],
            })
            for call, result in zip(calls, results):
                messages.append({"role": "tool", "tool_name": call.name, "content": result})

        logger.warning("Gave up after %d tool rounds", MAX_TOOL_ROUNDS)
        return STUCK_REPLY

    def _is_direct(self, call: ToolCall) -> bool:
        tool = self._tools.get(call.name)
        return tool is not None and tool.direct

    def _recover_leaked_call(self, content: str) -> list[ToolCall]:
        """Small models sometimes write a tool call as text instead of using the tool API."""
        match = LEAKED_CALL.match(content)
        if not match or match.group(1) not in self._tools:
            return []

        try:
            data = json.loads(content)
            arguments = data.get("parameters") or data.get("arguments") or {}
        except (json.JSONDecodeError, AttributeError):
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        logger.debug("Recovered tool call from text: %s(%s)", match.group(1), arguments)
        return [ToolCall(match.group(1), arguments)]

    def _run_tool(self, call: ToolCall) -> str:
        logger.debug("Tool call: %s(%s)", call.name, call.arguments)
        tool = self._tools.get(call.name)
        if tool is None:
            return f"Error: there is no tool named {call.name}."
        try:
            return tool.run(call.arguments)
        except Exception as exc:
            logger.exception("Tool %s failed", call.name)
            return f"Error: the {call.name} tool failed ({exc})."