from typing import Any

import yaml

# Short forms of the questions in judge.JUDGE_SYSTEM, shown to the person labeling.
PROMPTS = {
    "diagnosis_correct": "Does the answer settle on the real cause (or correctly say why a human is needed)?",
    "blames_decoy": "Does the answer present the decoy as a cause? (Ruling it out is not blaming it.)",
    "grounded": "Does every claim attributed to the docs appear in the cited chunk text?",
}


def applicable(item: dict[str, Any]) -> list[str]:
    """The questions that apply to an item, read off what the judge was shown: blames_decoy only
    when there is a decoy, grounded only when something was cited."""
    shown = item["shown_to_judge"]
    questions = ["diagnosis_correct"]
    if "<decoy>\nnone\n</decoy>" not in shown:
        questions.append("blames_decoy")
    if "<cited_chunks>\n(none)\n</cited_chunks>" not in shown:
        questions.append("grounded")
    return questions


def is_labeled(item: dict[str, Any]) -> bool:
    return all(item[q] is not None for q in applicable(item))


def read_header(text: str) -> str:
    lines = []
    for line in text.splitlines(keepends=True):
        if not line.startswith("#"):
            break
        lines.append(line)
    return "".join(lines)


class _BlockDumper(yaml.SafeDumper):
    pass


def _block_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    # Multi-line text as a | block, so a person reads the answer as written, not as one escaped
    # line. PyYAML falls back to quoting when a line has trailing spaces.
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_BlockDumper.add_representer(str, _block_str)


def dump_labels(items: list[dict[str, Any]], header: str) -> str:
    return header + yaml.dump(
        items, Dumper=_BlockDumper, sort_keys=False, width=100, allow_unicode=True
    )
