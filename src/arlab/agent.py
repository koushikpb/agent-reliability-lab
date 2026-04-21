import re
from dataclasses import dataclass, field
from typing import Any

from arlab.llm import Provider, cost_usd
from arlab.sandbox import TOOLS, ToolClient

CITATION = re.compile(r"\[cite:\s*([^\]\s]+)\s*\]")


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    result: str
    is_error: bool


@dataclass
class AgentResult:
    # end_turn, max_tokens, refusal, max_steps, or no_tool_calls (tool_use with no tool blocks)
    stop: str = "max_steps"
    steps: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def cited_ids(self) -> list[str]:
        """Chunk ids cited in ticket notes and the final answer, in first-seen order."""
        notes = [c.args.get("note", "") for c in self.tool_calls if c.name == "update_ticket"]
        found = [cid for text in [*notes, self.final_text] for cid in CITATION.findall(text)]
        return list(dict.fromkeys(found))


def run_agent(
    provider: Provider, tools: ToolClient, system: str, ticket_id: str, max_steps: int
) -> AgentResult:
    result = AgentResult()
    messages: list[dict[str, Any]] = [{"role": "user", "content": f"Resolve ticket {ticket_id}."}]
    while result.steps < max_steps:
        response = provider.complete(system, messages, TOOLS)
        result.steps += 1
        result.input_tokens += response.input_tokens
        result.output_tokens += response.output_tokens
        result.cost_usd += cost_usd(response)
        # The full content goes back, thinking blocks included, unchanged.
        messages.append({"role": "assistant", "content": response.content})
        text = [b["text"] for b in response.content if b["type"] == "text"]
        if text:
            result.final_text = "\n".join(text)
        if response.stop_reason != "tool_use":
            result.stop = response.stop_reason
            return result
        tool_results = []
        for block in response.content:
            if block["type"] != "tool_use":
                continue
            output, is_error = tools.call(block["name"], block["input"])
            result.tool_calls.append(ToolCall(block["name"], block["input"], output, is_error))
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": output,
                    "is_error": is_error,
                }
            )
        if not tool_results:
            result.stop = "no_tool_calls"
            return result
        # All results in one message: splitting them discourages parallel tool calls.
        messages.append({"role": "user", "content": tool_results})
    return result
