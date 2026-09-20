"""Paired correctness statistics and molecular-answer entropy.

Correctness must be computed on the same ordered samples before comparison;
missing predictions should be explicitly scored by the caller, never dropped
or silently coerced here. Repeated variants of one origin are not independent
trials: use one prespecified outcome per independent origin for inferential CIs
and McNemar tests, or treat results over variants as descriptive only.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
import math
from typing import Any


def _count(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def wilson_interval(successes: int, total: int) -> tuple[float, float] | None:
    """Return a two-sided Wilson score 95% interval; no trials yields None."""
    _count(successes, "successes")
    _count(total, "total")
    if successes > total:
        raise ValueError("successes cannot exceed total")
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def mcnemar_exact_p(wrong_to_right: int, right_to_wrong: int) -> float:
    """Exact two-sided binomial McNemar p-value, capped at 1.

    Conditional on the number of discordant pairs, the null distribution is
    Binomial(n, 0.5). Integer arithmetic accumulates the exact lower-tail mass
    before conversion to floating point; extreme p-values may underflow to 0.
    With no discordant pairs, p is 1.
    """
    _count(wrong_to_right, "wrong_to_right")
    _count(right_to_wrong, "right_to_wrong")
    discordant = wrong_to_right + right_to_wrong
    if discordant == 0:
        return 1.0
    smaller = min(wrong_to_right, right_to_wrong)
    mass = sum(math.comb(discordant, count) for count in range(smaller + 1))
    return min(1.0, (2 * mass) / (1 << discordant))


def paired_comparison(
    baseline_correct: Sequence[bool],
    treatment_correct: Sequence[bool],
    *,
    baseline_label: str = "A",
    treatment_label: str = "B",
) -> dict[str, Any]:
    """Compare paired conditions; accuracy difference is treatment − baseline.

    Flip-to-wrong is conditional on baseline correctness, so its denominator is
    the number of correct baseline answers. The Wilson interval describes that
    binomial rate. Accuracy differences have no unpaired/Wald interval attached.
    """
    baseline, treatment = list(baseline_correct), list(treatment_correct)
    if len(baseline) != len(treatment):
        raise ValueError("paired conditions must have equal lengths")
    if any(type(value) is not bool for value in baseline + treatment):
        raise TypeError("correctness values must be explicit booleans")
    n = len(baseline)
    both_correct = sum(a and b for a, b in zip(baseline, treatment))
    both_wrong = sum(not a and not b for a, b in zip(baseline, treatment))
    right_to_wrong = sum(a and not b for a, b in zip(baseline, treatment))
    wrong_to_right = sum(not a and b for a, b in zip(baseline, treatment))
    baseline_successes, treatment_successes = sum(baseline), sum(treatment)
    return {
        "n": n,
        "baseline_label": baseline_label,
        "treatment_label": treatment_label,
        "baseline_accuracy": baseline_successes / n if n else None,
        "treatment_accuracy": treatment_successes / n if n else None,
        "baseline_accuracy_ci95": wilson_interval(baseline_successes, n),
        "treatment_accuracy_ci95": wilson_interval(treatment_successes, n),
        "accuracy_difference": (treatment_successes - baseline_successes) / n if n else None,
        "contingency": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "right_to_wrong": right_to_wrong,
            "wrong_to_right": wrong_to_right,
        },
        "mcnemar_exact_p": mcnemar_exact_p(wrong_to_right, right_to_wrong),
        "flip_to_wrong": {
            "count": right_to_wrong,
            "denominator": baseline_successes,
            "rate": right_to_wrong / baseline_successes if baseline_successes else None,
            "ci95": wilson_interval(right_to_wrong, baseline_successes),
        },
    }


def _canonical_smiles(answer: str) -> str | None:
    # Lazy import keeps the statistics and token-label modules usable without
    # importing RDKit or any model runtime.
    from rdkit import Chem

    parameters = Chem.SmilesParserParams()
    # The answer must be a molecular encoding, not a SMILES prefix followed by
    # prose that RDKit would otherwise silently accept as a molecule name.
    parameters.parseName = False
    molecule = Chem.MolFromSmiles(answer, parameters)
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    # Atom-map identifiers are bookkeeping, not molecular identity.
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def canonical_smiles_entropy(
    answers: Sequence[str | None],
    *,
    canonicalizer: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Shannon entropy in bits over canonical isomeric-SMILES answer classes.

    Every answer contributes to the denominator. Invalid SMILES share one
    ``invalid`` category, and None/blank answers share a separate ``missing``
    category. Valid classes use ``smiles:`` prefixes to prevent key collisions.
    ``coverage`` is valid answers / all answers; empty input yields None for
    entropy and coverage. Canonicalization preserves stereochemistry and all
    components; it does not claim tautomer or protonation equivalence.

    An injected canonicalizer must return canonical text or None for an invalid
    answer. Unexpected canonicalizer exceptions propagate to expose tool faults.
    """
    canonicalize = canonicalizer or _canonical_smiles
    clusters: Counter[str] = Counter()
    valid_count = invalid_count = missing_count = 0
    for answer in answers:
        if answer is None or (isinstance(answer, str) and not answer.strip()):
            missing_count += 1
            clusters["missing"] += 1
            continue
        if not isinstance(answer, str):
            raise TypeError("answers must be SMILES strings or None")
        canonical = canonicalize(answer.strip())
        if canonical is None or canonical == "":
            invalid_count += 1
            clusters["invalid"] += 1
        elif isinstance(canonical, str):
            valid_count += 1
            clusters["smiles:" + canonical] += 1
        else:
            raise TypeError("canonicalizer must return a string or None")
    n = valid_count + invalid_count + missing_count
    entropy = -sum((count / n) * math.log2(count / n) for count in clusters.values()) if n else None
    return {
        "n": n,
        "entropy_bits": entropy,
        "clusters": dict(clusters),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "missing_count": missing_count,
        "coverage": valid_count / n if n else None,
    }
