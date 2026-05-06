import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

# USD per million tokens, from the Anthropic pricing page (2026-10). Thinking tokens bill as output.
PRICES = {
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass(frozen=True)
class ModelResponse:
    content: list[dict[str, Any]]  # raw content blocks, passed back to the API unchanged
    stop_reason: str
    model: str
    input_tokens: int
    output_tokens: int


class Provider(Protocol):
    model: str

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse: ...


def cost_usd(response: ModelResponse) -> float:
    input_price, output_price = PRICES[response.model]
    return (response.input_tokens * input_price + response.output_tokens * output_price) / 1e6


class AnthropicProvider:
    def __init__(self, model: str, effort: str | None, max_tokens: int = 16000) -> None:
        import anthropic

        if model not in PRICES:
            raise ValueError(f"no price for {model}; add it to PRICES so spend stays visible")
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        # No temperature: the API rejects sampling parameters, so runs are not reproducible from
        # settings alone. Recorded responses are what make a suite repeatable.
        # No refusal fallbacks: a silent switch to another model would score that model's answer
        # under this config's hash. A refusal is recorded as its own failure class instead.
        # Thinking is explicit because Opus 4.7 runs without it when the field is omitted, while
        # newer models default to adaptive; omitting it would make the same config mean different
        # things on different models.
        # Plain dicts stand in for the SDK's TypedDict params; the API accepts both.
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
            "tools": tools,
        }
        # No effort means a model without adaptive thinking (Haiku 4.5): neither field is sent.
        if self.effort is not None:
            params["thinking"] = {"type": "adaptive"}
            params["output_config"] = {"effort": self.effort}
        response = self.client.messages.create(**params)
        return ModelResponse(
            content=[block.to_dict() for block in response.content],
            stop_reason=str(response.stop_reason),
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


def request_hash(
    model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> str:
    request = {"model": model, "system": system, "messages": messages, "tools": tools}
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


class ReplayMiss(LookupError):
    """No recording for this request: the prompt, tools, history or model changed."""


class RecordingStore:
    """One JSON file per request hash, so CI can replay with no network and no database."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, key: str) -> Any | None:
        path = self.root / f"{key}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, key: str, value: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        # Written to a temp file and renamed, so a crash never leaves a truncated recording.
        tmp = self.root / f"{key}.json.tmp"
        tmp.write_text(json.dumps(value, indent=1, sort_keys=True))
        os.replace(tmp, self.root / f"{key}.json")


class RecordingProvider:
    def __init__(self, inner: Provider, store: RecordingStore) -> None:
        self.inner = inner
        self.model = inner.model
        self.store = store

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        response = self.inner.complete(system, messages, tools)
        self.store.put(request_hash(self.model, system, messages, tools), asdict(response))
        return response


class ReplayProvider:
    def __init__(self, model: str, store: RecordingStore) -> None:
        self.model = model
        self.store = store

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        key = request_hash(self.model, system, messages, tools)
        recorded = self.store.get(key)
        if recorded is None:
            raise ReplayMiss(key)
        return ModelResponse(**recorded)
