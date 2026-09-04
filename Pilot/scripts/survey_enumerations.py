"""One-time corpus survey for prose enumeration/total constructions.

Run from ``Pilot/`` with ``python scripts/survey_enumerations.py``.  This is
deliberately a reporting script: the production validator lives in
``modules/text_realization/occurrence_audit.py``.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molhallulens.config.paths import DEFAULT_DATASET_ROOT
from molhallulens.modules.ingestion import ChemCoTMolEditAdapter
from molhallulens.modules.reference import build_reference_dag
from molhallulens.modules.text_realization.occurrence_audit import (
    _contains_explicit_enumeration,
)


NUMBER = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
NUMBER_PATTERN = re.compile(rf"(?<![A-Za-z0-9]){NUMBER}(?![A-Za-z0-9])", re.IGNORECASE)
UNIT_PATTERN = re.compile(r"\b(?:heavy atoms?|rings?)\b", re.IGNORECASE)
TOTAL_FIRST = re.compile(
    rf"\b{NUMBER}\s+(?:heavy atoms?|rings?)\s*\([^)]*\b{NUMBER}\b[^)]*\)",
    re.IGNORECASE,
)
EQUALS = re.compile(
    rf"\b{NUMBER}\b[^.\n]{{0,180}}=\s*{NUMBER}\s+(?:heavy atoms?|rings?)\b",
    re.IGNORECASE,
)
ITEMS_FIRST = re.compile(
    rf"\b{NUMBER}\b[^.\n]{{0,220}}\b(?:total(?:ing)?|giving\s+(?:a\s+)?total\s+of|for\s+a\s+total\s+of)\s+{NUMBER}\s+(?:heavy atoms?|rings?)\b",
    re.IGNORECASE,
)
UNIT_FIRST = re.compile(
    rf"\b(?:heavy atoms?|SSSR\s+rings?|ring\s+count)\b[^.\n]{{0,70}}?:\s*"
    rf"{NUMBER}\s*\(",
    re.IGNORECASE,
)


def main() -> None:
    origins = ChemCoTMolEditAdapter().load(DEFAULT_DATASET_ROOT)
    references = tuple(build_reference_dag(origin) for origin in origins)
    rows: list[tuple[str, int, str, str]] = []
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)

    for reference in references:
        for step in reference.trace_steps:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", step.natural_language):
                sentence = sentence.strip()
                if not sentence or UNIT_PATTERN.search(sentence) is None:
                    continue
                if len(NUMBER_PATTERN.findall(sentence)) < 2:
                    continue
                if not _contains_explicit_enumeration(sentence):
                    continue
                if EQUALS.search(sentence):
                    form = "equals_total"
                elif UNIT_FIRST.search(sentence):
                    form = "unit_first_parenthetical"
                elif TOTAL_FIRST.search(sentence):
                    form = "total_first_parenthetical"
                elif ITEMS_FIRST.search(sentence):
                    form = "items_first_total_marker"
                else:
                    form = "other_multi_number_unit"
                counts[form] += 1
                rows.append(
                    (
                        reference.anonymous_sample_id,
                        step.step_index,
                        form,
                        sentence,
                    )
                )
                if sentence not in examples[form] and len(examples[form]) < 12:
                    examples[form].append(sentence)

    print(f"origins={len(references)} candidate_sentences={len(rows)}")
    for form, count in sorted(counts.items()):
        print(f"\n[{form}] count={count}")
        for example in examples[form]:
            print(f"- {example}")
    print("\n[all rows]")
    for origin_id, step_index, form, sentence in rows:
        print(f"{origin_id}\tstep={step_index}\t{form}\t{sentence}")


if __name__ == "__main__":
    main()
