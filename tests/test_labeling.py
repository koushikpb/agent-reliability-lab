import yaml

from arlab.labeling import applicable, dump_labels, is_labeled, read_header


def item(decoy: str = "none", chunks: str = "(none)") -> dict[str, object]:
    return {
        "run_id": "r1",
        "scenario_id": "s1",
        "fault": "none",
        "shown_to_judge": (
            "<ticket>\nt\n</ticket>\n\n<real_cause>\nc\n</real_cause>\n\n"
            f"<decoy>\n{decoy}\n</decoy>\n\n<answer>\na\n</answer>\n\n"
            f"<cited_chunks>\n{chunks}\n</cited_chunks>"
        ),
        "diagnosis_correct": None,
        "blames_decoy": None,
        "grounded": None,
        "notes": "",
    }


def test_questions_follow_what_the_judge_was_shown() -> None:
    assert applicable(item()) == ["diagnosis_correct"]
    assert applicable(item(decoy="old image tag")) == ["diagnosis_correct", "blames_decoy"]
    assert applicable(item(chunks="[a#b-0]\ntext")) == ["diagnosis_correct", "grounded"]


def test_item_without_decoy_or_citations_is_labeled_by_diagnosis_alone() -> None:
    plain = item()
    plain["diagnosis_correct"] = False
    assert is_labeled(plain)
    cited = item(chunks="[a#b-0]\ntext")
    cited["diagnosis_correct"] = True
    assert not is_labeled(cited)


def test_dump_keeps_header_and_round_trips() -> None:
    header = "# label these\n# carefully\n"
    items = [item(decoy="old image tag")]
    items[0]["blames_decoy"] = True
    text = dump_labels(items, header)
    assert read_header(text) == header
    assert yaml.safe_load(text) == items
    assert "shown_to_judge: |-" in text
