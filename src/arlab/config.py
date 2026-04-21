import hashlib
import json
from pathlib import Path
from typing import Any

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
    effort: str
    prompt_version: str
    retriever: str
    retriever_params: RetrieverParams
    agent: AgentSettings


def load_config(path: Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(path.read_text()))


def system_prompt(config: RunConfig) -> str:
    return (PROMPTS_DIR / f"{config.prompt_version}.md").read_text(encoding="utf-8")


def resolved_params(config: RunConfig) -> dict[str, Any]:
    """Everything that can change the agent's behavior. prompt_version is only a label, so the
    prompt text itself is included, and so are the tool schemas the model sees."""
    return {**config.model_dump(), "prompt_text": system_prompt(config), "tools": TOOLS}


def config_hash(config: RunConfig) -> str:
    canonical = json.dumps(resolved_params(config), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
