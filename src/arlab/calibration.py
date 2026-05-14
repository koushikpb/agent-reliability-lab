from dataclasses import dataclass

QUESTIONS = ("diagnosis_correct", "blames_decoy", "grounded")


@dataclass(frozen=True)
class Agreement:
    question: str
    n: int
    agreement: float
    kappa: float | None
    # confusion[human][judge], counts of True/False pairs.
    true_true: int
    true_false: int
    false_true: int
    false_false: int

    @property
    def judge_false_positives(self) -> int:
        """Judge said True where the human said False."""
        return self.false_true

    @property
    def judge_false_negatives(self) -> int:
        return self.true_false


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Agreement corrected for chance. None when chance agreement is 1 (both raters gave one
    label to everything), where kappa is undefined."""
    if not pairs:
        raise ValueError("no pairs")
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    p_a = sum(a for a, _ in pairs) / n
    p_b = sum(b for _, b in pairs) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def agreement(question: str, pairs: list[tuple[bool, bool]]) -> Agreement:
    """`pairs` are (human, judge) labels for one question, items where either is null left out."""
    return Agreement(
        question=question,
        n=len(pairs),
        agreement=sum(h == j for h, j in pairs) / len(pairs),
        kappa=cohen_kappa(pairs),
        true_true=sum(h and j for h, j in pairs),
        true_false=sum(h and not j for h, j in pairs),
        false_true=sum(not h and j for h, j in pairs),
        false_false=sum(not h and not j for h, j in pairs),
    )


def markdown(results: list[Agreement]) -> str:
    lines = [
        "| Question | n | Agreement | Kappa | Judge false pos | Judge false neg |",
        "|---|---|---|---|---|---|",
    ]
    for a in results:
        kappa = "undefined" if a.kappa is None else f"{a.kappa:.2f}"
        lines.append(
            f"| {a.question} | {a.n} | {a.agreement:.2f} | {kappa}"
            f" | {a.judge_false_positives} | {a.judge_false_negatives} |"
        )
    return "\n".join(lines)
