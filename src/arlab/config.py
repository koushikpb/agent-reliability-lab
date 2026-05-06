import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from arlab.sandbox import TOOLS

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


class AgentSettings(BaseModel):
    max_steps: int
    max_tokens: int = 16000


class RetrieverParams(BaseModel):
    k: int


class RunConfig(BaseModel):
    model: str
    # None for models without adaptive thinking or effort (Haiku 4.5): neither is sent.
    effort: str | None
    prompt_version: str
    retriever: str
    retriever_params: RetrieverParams
    agent: AgentSettings
    # A deliberate bug, so the suite and the gate can be tested on whether they catch
    # it. wrong_chunks returns search ranks k+1..2k; schema_drift shows the model a get_pod_logs
    # schema whose argument the server no longer accepts.
    seeded_bug: Literal["wrong_chunks", "schema_drift"] | None = None


def load_config(path: Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(path.read_text()))


def system_prompt(config: RunConfig) -> str:
    return (PROMPTS_DIR / f"{config.prompt_version}.md").read_text(encoding="utf-8")


def tool_schemas(config: RunConfig) -> list[dict[str, Any]]:
    if config.seeded_bug != "schema_drift":
        return TOOLS
    drifted = copy.deepcopy(TOOLS)
    [logs] = [t for t in drifted if t["name"] == "get_pod_logs"]
    logs["input_schema"]["properties"] = {"pod_name": {"type": "string"}}
    logs["input_schema"]["required"] = ["pod_name"]
    return drifted


def resolved_params(config: RunConfig) -> dict[str, Any]:
    """Everything that can change the agent's behavior. prompt_version is only a label, so the
    prompt text itself is included, and so are the tool schemas the model sees."""
    # Left out when unset, so configs from before the field existed keep their hashes.
    params = config.model_dump(exclude={"seeded_bug"} if config.seeded_bug is None else None)
    return {**params, "prompt_text": system_prompt(config), "tools": tool_schemas(config)}


def config_hash(config: RunConfig) -> str:
    canonical = json.dumps(resolved_params(config), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
