import itertools
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.trace import Span as ApiSpan
from opentelemetry.trace import StatusCode, Tracer


@dataclass(frozen=True)
class Event:
    seq: int
    span_id: str
    parent_span_id: str | None
    kind: str  # scenario, model_call, tool_call, retrieval
    payload: dict[str, Any]
    started_at: datetime
    duration_ms: int


class EventCollector(SpanProcessor):
    """Mirrors finished spans into Event rows. seq follows span start order, which is the order a
    replay needs; spans end child-first, so end order would put a tool call after its retrieval."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self._seq = itertools.count()

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        span.set_attribute("arlab.seq", next(self._seq))

    def on_end(self, span: ReadableSpan) -> None:
        attrs = span.attributes or {}
        if "arlab.kind" not in attrs or span.start_time is None or span.end_time is None:
            return
        payload = json.loads(str(attrs["arlab.payload"]))
        if span.status.status_code == StatusCode.ERROR:
            # The span ended by an exception: keep the event, so a crashed run can be explained.
            payload["error"] = span.status.description or "error"
        self.events.append(
            Event(
                seq=int(attrs["arlab.seq"]),  # type: ignore[arg-type]
                span_id=format(span.context.span_id, "016x"),
                parent_span_id=format(span.parent.span_id, "016x") if span.parent else None,
                kind=str(attrs["arlab.kind"]),
                payload=payload,
                started_at=datetime.fromtimestamp(span.start_time / 1e9, tz=UTC),
                duration_ms=round((span.end_time - span.start_time) / 1e6),
            )
        )

    def drain(self) -> list[Event]:
        events = sorted(self.events, key=lambda e: e.seq)
        self.events = []
        self._seq = itertools.count()
        return events


def make_tracer() -> tuple[Tracer, EventCollector]:
    collector = EventCollector()
    provider = TracerProvider()
    provider.add_span_processor(collector)
    return provider.get_tracer("arlab"), collector


def record(span: ApiSpan, kind: str, payload: dict[str, Any]) -> None:
    # Span attributes only hold primitives, so the payload travels as one JSON string.
    span.set_attribute("arlab.kind", kind)
    span.set_attribute("arlab.payload", json.dumps(payload, sort_keys=True))
