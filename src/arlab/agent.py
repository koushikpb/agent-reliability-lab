import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.trace import NoOpTracer, Tracer

from arlab.llm import Provider, cost_usd
from arlab.sandbox import TOOLS, ToolClient
from arlab.tracing import record

CITATION = re.compile(r"\[cite:\s*([^\]\s]+)\s*\]")


def citations(texts: Iterable[str]) -> list[str]:
    found = [cid for text in texts for cid in CITATION.findall(text)]
    return list(dict.fromkeys(found))


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    result: str
    status: int

    @property
    def is_error(self) -> bool:
        return self.status >= 400

    @property
    def is_invalid(self) -> bool:
        """The agent's own mistake: unknown tool, bad arguments, missing ticket or pod."""
        return 400 <= self.status < 500 and self.status != 429

    @property
    def is_tool_error(self) -> bool:
        """A server-side failure the agent did not cause: 5xx, rate limit, or no response (0)."""
        return self.status >= 500 or self.status in (0, 429)


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
        return citations([*notes, self.final_text])


def run_agent(
    provider: Provider,
    tools: ToolClient,
    system: str,
    ticket_id: str,
    max_steps: int,
    tracer: Tracer | None = None,
) -> AgentResult:
    tracer = tracer or NoOpTracer()
    result = AgentResult()
    messages: list[dict[str, Any]] = [{"role": "user", "content": f"Resolve ticket {ticket_id}."}]
    while result.steps < max_steps:
        with tracer.start_as_current_span("model_call") as span:
            response = provider.complete(system, messages, TOOLS)
            record(
                span,
                "model_call",
                {
                    "model": response.model,
                    "stop_reason": response.stop_reason,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": cost_usd(response),
                    "content": response.content,
                },
            )
        result.steps += 1
        result.input_tokens += response.input_tokens
        result.output_tokens += response.output_tokens
        result.cost_usd += cost_usd(response)
        # The full content goes back, thinking blocks included, unchanged.
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "pause_turn":
            # The server paused a long turn; sending the history back resumes it.
            continue
        if response.stop_reason != "tool_use":
            # Only the terminal response is the answer. Narration from earlier tool-use turns
            # would otherwise leak its citations into cited_ids.
            texts = [b["text"] for b in response.content if b["type"] == "text"]
            result.final_text = "\n".join(texts)
            result.stop = response.stop_reason
            return result
        tool_results = []
        for block in response.content:
            if block["type"] != "tool_use":
                continue
            with tracer.start_as_current_span("tool_call") as span:
                tool_response = tools.call(block["name"], block["input"])
                record(
                    span,
                    "tool_call",
                    {
                        "name": block["name"],
                        "args": block["input"],
                        "status": tool_response.status,
                        "result": tool_response.text,
                    },
                )
            call = ToolCall(block["name"], block["input"], tool_response.text, tool_response.status)
            result.tool_calls.append(call)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": call.result,
                    "is_error": call.is_error,
                }
            )
        if not tool_results:
            result.stop = "no_tool_calls"
            return result
        # All results in one message: splitting them discourages parallel tool calls.
        messages.append({"role": "user", "content": tool_results})
    return result
