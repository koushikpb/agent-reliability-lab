import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx2
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.testclient import TestClient

# search_docs is the only tool that leaves the fixture: it queries the real index.
SearchFn = Callable[[str], list[dict[str, str]]]

TICKET_STATUSES = ("open", "in_progress", "resolved")

# One fault per run. Each fires on the first call to its tools only (empty_search on every call),
# so a fault the agent can recover from by retrying is recoverable.
FAULTS = ("none", "timeout", "rate_limit", "malformed", "stale", "empty_search")
FAULT_TOOLS = {
    # The write is applied and then the response is lost: the case where a blind retry is unsafe.
    "timeout": ("update_ticket", "escalate"),
    "rate_limit": ("list_pods", "get_pod_logs"),
    "malformed": ("get_pod_logs",),
    # A cached snapshot from before the incident: every pod running and ready.
    "stale": ("list_pods",),
    "empty_search": ("search_docs",),
}
RETRY_AFTER_S = 2
# Pod fields that describe the incident, absent from a snapshot taken before it.
INCIDENT_FIELDS = {"last_state", "message", "events", "reason", "conditions", "deletion_timestamp"}
OnFault = Callable[[dict[str, Any]], None]


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    # strict keeps tool inputs schema-valid, so an invalid call is the agent's wrong choice of
    # arguments rather than malformed JSON.
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


TOOLS = [
    _tool(
        "search_docs",
        "Search the Kubernetes documentation. Returns chunks with ids you can cite.",
        {"query": {"type": "string"}},
    ),
    _tool("get_ticket", "Read a support ticket.", {"id": {"type": "string"}}),
    _tool(
        "list_pods",
        "List pods in a namespace with status, restart count and readiness.",
        {"namespace": {"type": "string"}},
    ),
    _tool("get_pod_logs", "Read recent logs for one pod.", {"pod": {"type": "string"}}),
    _tool(
        "update_ticket",
        "Set a ticket's status and add a note. Use status resolved with a cited explanation.",
        {
            "id": {"type": "string"},
            "status": {"type": "string", "enum": list(TICKET_STATUSES)},
            "note": {"type": "string"},
        },
    ),
    _tool(
        "escalate",
        "Hand the ticket to a human engineer when it cannot be resolved safely.",
        {"id": {"type": "string"}, "reason": {"type": "string"}},
    ),
]


def create_tool_server(
    fixture: dict[str, Any], search: SearchFn, fault: str = "none", on_fault: OnFault | None = None
) -> FastAPI:
    """Tool server over one scenario's cluster fixture. State is a copy, mutated by updates."""
    if fault not in FAULTS:
        raise ValueError(f"unknown fault {fault}; one of {FAULTS}")
    state = copy.deepcopy(fixture)
    app = FastAPI()
    app.state.cluster = state
    fired = False

    @app.middleware("http")
    async def inject_fault(request: Request, call_next: Any) -> Response:
        nonlocal fired
        tool = request.url.path.removeprefix("/tools/")
        if fault == "none" or tool not in FAULT_TOOLS[fault] or (fired and fault != "empty_search"):
            return await call_next(request)
        if (
            fault == "stale"
            and json.loads(await request.body()).get("namespace") not in state["pods"]
        ):
            # An unknown namespace is the agent's mistake; the stale snapshot waits for a real one.
            return await call_next(request)
        fired = True
        event: dict[str, Any] = {"fault": fault, "tool": tool}
        if fault == "rate_limit":
            response: Response = Response(
                json.dumps({"detail": "rate limited"}),
                status_code=429,
                headers={"Retry-After": str(RETRY_AFTER_S)},
                media_type="application/json",
            )
        elif fault == "malformed":
            response = Response('{"pod": "', media_type="application/json")
        elif fault == "stale":
            namespace = json.loads(await request.body())["namespace"]
            pods = [
                {k: v for k, v in pod.items() if k not in INCIDENT_FIELDS}
                | {"status": "Running", "ready": True, "restarts": 0}
                for pod in state["pods"][namespace]
            ]
            response = Response(json.dumps(pods), media_type="application/json")
        elif fault == "empty_search":
            response = Response("[]", media_type="application/json")
        else:
            applied = await call_next(request)
            body = json.loads(b"".join([chunk async for chunk in applied.body_iterator]))
            # The trace has to show that the write landed, or the stored final status could not
            # be rebuilt from events.
            event["applied_status"] = body.get("status") if applied.status_code < 400 else None
        if on_fault is not None:
            on_fault(event)
        if fault == "timeout":
            raise httpx2.ReadTimeout("injected timeout after the write was applied")
        return response

    def arg(args: dict[str, str], name: str) -> str:
        # A missing argument is the caller's mistake (e.g. a client on a drifted schema), not a crash.
        if name not in args:
            raise HTTPException(422, f"missing argument {name!r}")
        return args[name]

    def ticket(ticket_id: str) -> dict[str, Any]:
        if ticket_id not in state["tickets"]:
            raise HTTPException(404, f"ticket {ticket_id} not found")
        return state["tickets"][ticket_id]

    @app.post("/tools/search_docs")
    def search_docs(args: dict[str, str]) -> list[dict[str, str]]:
        return search(arg(args, "query"))

    @app.post("/tools/get_ticket")
    def get_ticket(args: dict[str, str]) -> dict[str, Any]:
        return ticket(arg(args, "id"))

    @app.post("/tools/list_pods")
    def list_pods(args: dict[str, str]) -> list[dict[str, Any]]:
        namespace = arg(args, "namespace")
        if namespace not in state["pods"]:
            raise HTTPException(404, f"namespace {namespace} not found")
        return state["pods"][namespace]

    @app.post("/tools/get_pod_logs")
    def get_pod_logs(args: dict[str, str]) -> dict[str, str]:
        pod = arg(args, "pod")
        if pod not in state["logs"]:
            raise HTTPException(404, f"pod {pod} not found")
        return {"pod": pod, "logs": state["logs"][pod]}

    @app.post("/tools/update_ticket")
    def update_ticket(args: dict[str, str]) -> dict[str, Any]:
        t = ticket(arg(args, "id"))
        status, note = arg(args, "status"), arg(args, "note")
        if t["status"] == "escalated":
            # Escalation hands the ticket to a human; the agent may not quietly undo it.
            raise HTTPException(409, "ticket is escalated and owned by a human engineer")
        if status not in TICKET_STATUSES:
            raise HTTPException(422, f"status must be one of {TICKET_STATUSES}")
        # Not idempotent: a retried call appends the note twice. `arlab faults` measures this.
        t["status"] = status
        t["notes"].append(note)
        return t

    @app.post("/tools/escalate")
    def escalate(args: dict[str, str]) -> dict[str, Any]:
        t = ticket(arg(args, "id"))
        reason = arg(args, "reason")
        t["status"] = "escalated"
        t["notes"].append(f"escalated: {reason}")
        return t

    return app


@dataclass(frozen=True)
class ToolResponse:
    text: str  # what the model sees
    status: int  # HTTP status, kept so agent mistakes (4xx) and server faults (5xx, 429) differ


def _detail(response: httpx2.Response) -> str:
    # Error bodies are not always JSON (a proxy 502, an injected malformed body).
    try:
        return str(response.json().get("detail", response.text))
    except (ValueError, AttributeError):
        return response.text[:500]


class ToolClient:
    """Calls the tool server over HTTP, so faults injected at the HTTP layer reach the agent."""

    def __init__(self, http: httpx2.Client) -> None:
        self.http = http

    @classmethod
    def in_process(cls, app: FastAPI) -> "ToolClient":
        # An exception inside the server is a bug or an infrastructure failure (a replay miss, the
        # database down), not something the agent should see as a tool error, so it propagates.
        # Injected faults are explicit HTTP responses instead.
        return cls(TestClient(app, raise_server_exceptions=True))

    def call(self, name: str, args: dict[str, Any]) -> ToolResponse:
        try:
            response = self.http.post(f"/tools/{name}", json=args)
        except httpx2.TimeoutException:
            return ToolResponse("error: tool call timed out", 0)
        if response.status_code == 404 and _detail(response) == "Not Found":
            return ToolResponse(f"unknown tool {name}", 404)
        if response.is_error:
            detail = _detail(response)
            if "Retry-After" in response.headers:
                detail += f" (retry after {response.headers['Retry-After']}s)"
            return ToolResponse(f"error {response.status_code}: {detail}", response.status_code)
        try:
            body = response.json()
        except ValueError:
            return ToolResponse(f"error: malformed response body: {response.text[:200]}", 0)
        return ToolResponse(json.dumps(body), response.status_code)
