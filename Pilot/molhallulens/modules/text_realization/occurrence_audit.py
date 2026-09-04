"""High-recall, fail-closed audits for natural-language molecular claims."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


NUMERIC_NODE_IDS = frozenset(
    {
        "anchor_idx",
        "source_heavy",
        "product_heavy",
        "heavy_delta",
        "source_rings",
        "product_rings",
        "ring_delta",
        "fragment_heavy",
        "remove_heavy",
        "add_heavy",
    }
)

_FRAGMENT_COUNT_STEPS = frozenset(
    {
        "FRAGMENT_IDENTIFICATION",
        "ADD_FRAGMENT_SIZE",
        "REMOVE_GROUP_SIZE",
        "GROUP_SIZE_VERIFICATION",
    }
)
_COUNT_DERIVATION_WORDS = re.compile(
    r"\b(?:consists?|compris(?:es|ing)|total|sum(?:ming)?|giving|made\s+up|"
    r"one|two|three|four|five|six|seven|eight|nine)\b",
    flags=re.IGNORECASE,
)
_NUMBER = r"[+-]?\d+(?:\.\d+)?"
_ARITHMETIC = re.compile(
    rf"(?<![:\w])(?P<expression>{_NUMBER}(?:\s*[+-]\s*\d+(?:\.\d+)?)+)"
    rf"\s*=\s*(?P<result>{_NUMBER})(?!\d)"
)


def _value_pattern(node_id: str, value: str) -> str:
    escaped = (
        rf"\+?{re.escape(value[1:])}"
        if node_id in NUMERIC_NODE_IDS and value.startswith("+")
        else re.escape(value)
    )
    if node_id in NUMERIC_NODE_IDS:
        return rf"(?<![\d.]){escaped}(?![\d.])"
    return escaped


def _capture(pattern: str, text: str) -> set[tuple[int, int]]:
    return {
        match.span("value")
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    }


def loose_occurrence_spans(
    node_id: str,
    value: str,
    natural_body: str,
    *,
    step_name: str,
) -> tuple[tuple[int, int], ...]:
    """Find conservative high-recall mentions used to audit the strict finder.

    These matches are not used directly as fine-grained labels.  Extra loose
    matches force a whole-derivation rewrite, avoiding false precision.
    """

    needle = _value_pattern(node_id, value)
    spans: set[tuple[int, int]] = set()
    heavy_unit = r"\s+heavy(?:\s*\([^)]*\))?\s+atoms?"

    if node_id == "anchor_idx":
        spans |= _capture(
            rf"\b(?:atom|idx|index|map)\s*(?:number)?\s*(?:=|:)?\s*"
            rf"(?P<value>{needle})",
            natural_body,
        )
        spans |= _capture(rf"\b[A-Z][a-z]?(?P<value>{needle})\b", natural_body)
        spans |= _capture(
            rf"\[[^\]]*:\s*(?P<value>{needle})\s*\]",
            natural_body,
        )
    elif node_id == "anchor_element":
        spans |= _capture(
            rf"\belement\s*(?:symbol)?\s*(?:=|:)?\s*[\"']?"
            rf"(?P<value>{needle})(?![a-z])",
            natural_body,
        )
        spans |= _capture(
            rf"\banchor\b[^.\n]{{0,100}}?\(\s*(?P<value>{needle})\s*\)",
            natural_body,
        )
    elif node_id in {"source_heavy", "product_heavy"}:
        subject = "source" if node_id == "source_heavy" else "product"
        blocker = "product" if node_id == "source_heavy" else "source"
        spans |= _capture(
            rf"\b{subject}\b(?:(?!\b{blocker}\b)[^.\n]){{0,220}}?"
            rf"(?P<value>{needle}){heavy_unit}",
            natural_body,
        )
        if node_id == "source_heavy":
            spans |= _capture(
                rf"\b(?:max(?:imum)?\s+(?:atom[- ]?map|map)|map)\b"
                rf"[^.\n]{{0,50}}?(?P<value>{needle})",
                natural_body,
            )
        context = r"(?:HEAVY_ATOM_DELTA|net\s+change|change\s+in\s+heavy\s+atoms|check)"
        if node_id == "product_heavy":
            spans |= _capture(
                rf"{context}[^.\n]{{0,160}}?(?:is|=|:)\s*"
                rf"(?P<value>{needle})\s*-",
                natural_body,
            )
        else:
            spans |= _capture(
                rf"{context}[^.\n]{{0,160}}?(?:is|=|:)\s*"
                rf"{_NUMBER}\s*-\s*(?P<value>{needle})\s*=",
                natural_body,
            )
    elif node_id in {"source_rings", "product_rings"}:
        subject = "source" if node_id == "source_rings" else "product"
        blocker = "product" if node_id == "source_rings" else "source"
        spans |= _capture(
            rf"\b{subject}\b(?:(?!\b{blocker}\b)[^.\n]){{0,220}}?"
            rf"(?P<value>{needle})\s+rings?",
            natural_body,
        )
        context = r"(?:RING_DELTA|net\s+change(?:\s+in\s+rings)?|check)"
        if node_id == "product_rings":
            spans |= _capture(
                rf"{context}[^.\n]{{0,160}}?(?:is|=|:)\s*"
                rf"(?P<value>{needle})\s*-",
                natural_body,
            )
        else:
            spans |= _capture(
                rf"{context}[^.\n]{{0,160}}?(?:is|=|:)\s*"
                rf"{_NUMBER}\s*-\s*(?P<value>{needle})\s*=",
                natural_body,
            )
    elif node_id in {"heavy_delta", "ring_delta"}:
        domain = "HEAVY_ATOM_DELTA" if node_id == "heavy_delta" else "RING_DELTA"
        spans |= _capture(
            rf"\b(?:{domain}|claimed\s+delta|net\s+change|change\s+in\s+"
            rf"(?:heavy\s+atoms|rings)|expected\s+(?:difference|change)|matches|"
            rf"cross-check)\b[^.\n]{{0,220}}?(?:=|is)\s*"
            rf"(?P<value>{needle})",
            natural_body,
        )
    elif node_id in {"fragment_heavy", "remove_heavy", "add_heavy"}:
        if step_name in _FRAGMENT_COUNT_STEPS:
            spans |= _capture(
                rf"(?P<value>{needle}){heavy_unit}",
                natural_body,
            )
            spans |= _capture(
                rf"\b(?:total(?:\s+number)?|k|k_add|k_remove|ADD_HEAVY|"
                rf"REMOVE_HEAVY)\b[^.\n]{{0,80}}?(?:=|is|:|of)\s*"
                rf"(?P<value>{needle})",
                natural_body,
            )
        if node_id == "fragment_heavy":
            label = r"(?:fragment|ADD_FRAGMENT|k)"
        elif node_id == "remove_heavy":
            label = r"(?:removed\s+group|leaving\s+group|REMOVE_HEAVY|k_remove)"
        else:
            label = r"(?:added\s+fragment|ADD_HEAVY|k_add)"
        blocker = "k_remove" if node_id == "add_heavy" else "k_add"
        spans |= _capture(
            rf"\b{label}\b(?:(?!\b{blocker}\b)[^.\n]){{0,140}}?"
            rf"(?:=|is|\(|of)?\s*(?P<value>{needle})"
            rf"(?=\s*(?:{heavy_unit}|[-=).,]))",
            natural_body,
        )
        if step_name == "HEAVY_ATOM_VERIFICATION":
            if node_id in {"fragment_heavy", "add_heavy"}:
                spans |= _capture(
                    rf"\b(?:matches|cross-check|equals|difference)\b"
                    rf"[^.\n]{{0,160}}?(?P<value>{needle})\s*-\s*{_NUMBER}",
                    natural_body,
                )
            if node_id == "remove_heavy":
                spans |= _capture(
                    rf"\b(?:matches|cross-check|equals|difference)\b"
                    rf"[^.\n]{{0,160}}?{_NUMBER}\s*-\s*"
                    rf"(?P<value>{needle})(?=\s*(?:=|[).,]))",
                    natural_body,
                )
    else:
        spans |= {
            match.span()
            for match in re.finditer(re.escape(value), natural_body)
        }
    return tuple(sorted(spans))


def requires_derivation_rewrite(
    natural_body: str,
    affected_node_ids: frozenset[str],
) -> bool:
    """Return whether local value substitution cannot preserve coherent prose."""

    count_nodes = {
        "fragment_heavy",
        "remove_heavy",
        "add_heavy",
        "source_heavy",
        "product_heavy",
        "heavy_delta",
        "source_rings",
        "product_rings",
        "ring_delta",
    }
    if not (affected_node_ids & count_nodes):
        return False
    if _ARITHMETIC.search(natural_body) is not None:
        return True
    if affected_node_ids & {"fragment_heavy", "remove_heavy", "add_heavy"}:
        return _COUNT_DERIVATION_WORDS.search(natural_body) is not None
    return False


def arithmetic_violations(text: str) -> tuple[str, ...]:
    """Return simple numeric equations whose displayed arithmetic is false."""

    violations = []
    for match in _ARITHMETIC.finditer(text):
        expression = match.group("expression")
        result_text = match.group("result")
        terms = re.findall(r"[+-]?\s*\d+(?:\.\d+)?", expression)
        try:
            total = sum(
                (Decimal(term.replace(" ", "")) for term in terms),
                Decimal(0),
            )
            result = Decimal(result_text)
        except InvalidOperation:
            continue
        if total != result:
            violations.append(match.group(0))
    return tuple(violations)


__all__ = [
    "NUMERIC_NODE_IDS",
    "arithmetic_violations",
    "loose_occurrence_spans",
    "requires_derivation_rewrite",
]
