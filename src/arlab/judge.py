import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arlab.agent import answer_texts, citations
from arlab.results import calls_and_answer
from arlab.tracing import Event

# Sonnet 4.6 judges Opus 4.7 and Sonnet 4.6 runs. It is cheaper than the agent model, and on Sonnet
# runs it grades its own model's answers, which can bias it upward; calibration checks both.
JUDGE_MODEL = "claude-sonnet-4-6"

JUDGE_SYSTEM = """\
You grade an ops-support agent's answer to a Kubernetes support ticket. You get the ticket, the \
scenario author's note on the real cause (and on a decoy, if there is one), everything the agent \
told the ticket owner, and the text of each documentation chunk the agent cited.

Answer three questions. Write a short reason first, then the verdicts.

diagnosis_correct: does the answer name the real cause from the author's note as the cause of the \
problem, or, for a ticket that needs a human, correctly say why it must go to a human? Wording can \
differ; the substance must match. An answer that lists several possible causes without settling on \
the real one is not correct.

blames_decoy: only if the author's note names a decoy. True if the answer presents the decoy as the \
cause or as a contributing cause. Mentioning the decoy to rule it out is not blaming it. null if \
there is no decoy.

grounded: only if the agent cited at least one chunk. True if every claim the answer attributes to \
documentation is supported by the text of the cited chunks. A citation to a chunk that does not say \
what the answer says makes it false. Claims from the cluster (logs, pod status) are not judged here. \
null if nothing was cited.
"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "diagnosis_correct": {"type": "boolean"},
        "blames_decoy": {"type": ["boolean", "null"]},
        "grounded": {"type": ["boolean", "null"]},
    },
    "required": ["reason", "diagnosis_correct", "blames_decoy", "grounded"],
    "additionalProperties": False,
}

# Bumped when scenario_reference changes what the judge is shown.
REFERENCE_FORMAT = 3

# Stored judgments are keyed by this, so a changed prompt or model never reuses an old verdict.
JUDGE_HASH = hashlib.sha256(
    json.dumps([JUDGE_MODEL, JUDGE_SYSTEM, JUDGE_SCHEMA, REFERENCE_FORMAT], sort_keys=True).encode()
).hexdigest()[:16]


CATEGORY_TAGS = (
    "easy lookup",
    "multi-hop",
    "misleading logs",
    "should escalate",
    "no answer in docs",
)


@dataclass(frozen=True)
class Reference:
    cause: str
    decoy: str | None


@dataclass(frozen=True)
class Verdict:
    reason: str
    diagnosis_correct: bool
    blames_decoy: bool | None
    grounded: bool | None


def scenario_reference(path: Path) -> Reference:
    """The author's header comment is the reference cause; a `# decoy:` line names the decoy."""
    lines = path.read_text().splitlines()
    header = []
    for line in lines:
        if not line.startswith("#"):
            break
        if not line.startswith("# decoy:"):
            header.append(line.lstrip("# ").strip())
    # The category tag would tell the judge what kind of trap to look for.
    cause = re.sub(rf"^({'|'.join(CATEGORY_TAGS)}):\s*", "", " ".join(header), flags=re.IGNORECASE)
    # Notes on how the deterministic checks are set up say nothing about the cause.
    cause = re.sub(r"\s*[^.]*\bis required as a tool\b[^.]*\.", "", cause)
    decoys = [line.split(":", 1)[1] for line in lines if line.startswith("# decoy:")]
    decoy = re.sub(r"\s*\(judged in M10[^)]*\)", "", decoys[0]).strip() if decoys else None
    return Reference(cause, decoy)


def judge_prompt(ticket: dict[str, str], reference: Reference, events: list[Event]) -> str:
    calls, final_text = calls_and_answer(events)
    answer = [t for t in answer_texts(calls, final_text) if t.strip()]
    retrieved: dict[str, str] = {}
    for c in calls:
        if c.name == "search_docs" and not c.is_error:
            retrieved |= {h["chunk_id"]: h["text"] for h in json.loads(c.result)}
    cited = citations(answer)
    chunks = [f"[{cid}]\n{retrieved.get(cid, '(never retrieved in this run)')}" for cid in cited]
    return "\n\n".join(
        [
            f"<ticket>\n{ticket['title']}\n{ticket['body']}\n</ticket>",
            f"<real_cause>\n{reference.cause}\n</real_cause>",
            f"<decoy>\n{reference.decoy or 'none'}\n</decoy>",
            "<answer>\n" + ("\n---\n".join(answer) or "(the agent said nothing)") + "\n</answer>",
            "<cited_chunks>\n" + ("\n\n".join(chunks) or "(none)") + "\n</cited_chunks>",
        ]
    )


def call_judge(client: Any, prompt: str) -> tuple[Verdict, dict[str, Any], float]:
    """The verdict, the raw response (stored so the judgment can be re-read offline), and cost."""
    from arlab.llm import PRICES

    # No thinking: the reason field comes first in the schema, so the verdict follows it.
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=2000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
    )
    raw = response.to_dict()
    texts = [b.text for b in response.content if b.type == "text"]
    if not texts:
        raise ValueError(f"judge returned no text (stop_reason {response.stop_reason})")
    text = texts[0]
    input_price, output_price = PRICES[JUDGE_MODEL]
    cost = (
        response.usage.input_tokens * input_price + response.usage.output_tokens * output_price
    ) / 1e6
    return Verdict(**json.loads(text)), raw, cost


@dataclass(frozen=True)
class JudgedRow:
    scenario_id: str
    success: bool
    failure_class: str | None
    verdict: Verdict


def _rate(values: list[bool | None]) -> str:
    known = [v for v in values if v is not None]
    return f"{sum(known)}/{len(known)}" if known else "n/a"


def judgment_markdown(rows: list[JudgedRow]) -> str:
    """Judge verdicts next to the deterministic outcome. The disagreements are the useful part:
    they point at a check that is too loose or too strict, or at a judge that is wrong."""
    passed = [r for r in rows if r.success]
    failed = [r for r in rows if not r.success]
    ungrounded = [r for r in rows if r.failure_class == "ungrounded_citation"]
    lines = [
        "| Judge question | Deterministic pass | Deterministic fail | All |",
        "|---|---|---|---|",
    ]
    for question in ("diagnosis_correct", "blames_decoy", "grounded"):
        cells = [_rate([getattr(r.verdict, question) for r in g]) for g in (passed, failed, rows)]
        lines.append(f"| {question} | {' | '.join(cells)} |")
    wrong_but_passed = sorted(r.scenario_id for r in passed if not r.verdict.diagnosis_correct)
    decoy_but_passed = sorted(r.scenario_id for r in passed if r.verdict.blames_decoy)
    grounded_anyway = _rate([r.verdict.grounded for r in ungrounded])
    lines += [
        "",
        f"ungrounded_citation results the judge calls grounded: {grounded_anyway}",
        f"passed with a wrong diagnosis: {', '.join(wrong_but_passed) or 'none'}",
        f"passed while blaming the decoy: {', '.join(decoy_but_passed) or 'none'}",
    ]
    return "\n".join(lines)
