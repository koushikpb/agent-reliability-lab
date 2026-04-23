import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx2
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

# search_docs is the only tool that leaves the fixture: it queries the real index.
SearchFn = Callable[[str], list[dict[str, str]]]

TICKET_STATUSES = ("open", "in_progress", "resolved")


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


def create_tool_server(fixture: dict[str, Any], search: SearchFn) -> FastAPI:
    """Tool server over one scenario's cluster fixture. State is a copy, mutated by updates."""
    state = copy.deepcopy(fixture)
    app = FastAPI()
    app.state.cluster = state

    def ticket(ticket_id: str) -> dict[str, Any]:
        if ticket_id not in state["tickets"]:
            raise HTTPException(404, f"ticket {ticket_id} not found")
        return state["tickets"][ticket_id]

    @app.post("/tools/search_docs")
    def search_docs(args: dict[str, str]) -> list[dict[str, str]]:
        return search(args["query"])

    @app.post("/tools/get_ticket")
    def get_ticket(args: dict[str, str]) -> dict[str, Any]:
        return ticket(args["id"])

    @app.post("/tools/list_pods")
    def list_pods(args: dict[str, str]) -> list[dict[str, Any]]:
        if args["namespace"] not in state["pods"]:
            raise HTTPException(404, f"namespace {args['namespace']} not found")
        return state["pods"][args["namespace"]]

    @app.post("/tools/get_pod_logs")
    def get_pod_logs(args: dict[str, str]) -> dict[str, str]:
        if args["pod"] not in state["logs"]:
            raise HTTPException(404, f"pod {args['pod']} not found")
        return {"pod": args["pod"], "logs": state["logs"][args["pod"]]}

    @app.post("/tools/update_ticket")
    def update_ticket(args: dict[str, str]) -> dict[str, Any]:
        t = ticket(args["id"])
        if t["status"] == "escalated":
            # Escalation hands the ticket to a human; the agent may not quietly undo it.
            raise HTTPException(409, "ticket is escalated and owned by a human engineer")
        if args["status"] not in TICKET_STATUSES:
            raise HTTPException(422, f"status must be one of {TICKET_STATUSES}")
        # Not idempotent: a retried call appends the note twice. Milestone 8 measures this.
        t["status"] = args["status"]
        t["notes"].append(args["note"])
        return t

    @app.post("/tools/escalate")
    def escalate(args: dict[str, str]) -> dict[str, Any]:
        t = ticket(args["id"])
        t["status"] = "escalated"
        t["notes"].append(f"escalated: {args['reason']}")
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
        return cls(TestClient(app, raise_server_exceptions=False))

    def call(self, name: str, args: dict[str, Any]) -> ToolResponse:
        response = self.http.post(f"/tools/{name}", json=args)
        if response.status_code == 404 and _detail(response) == "Not Found":
            return ToolResponse(f"unknown tool {name}", 404)
        if response.is_error:
            return ToolResponse(
                f"error {response.status_code}: {_detail(response)}", response.status_code
            )
        return ToolResponse(json.dumps(response.json()), response.status_code)
