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

_PRIMARY_FRAGMENT_COUNT_NODE_BY_STEP = {
    "FRAGMENT_IDENTIFICATION": "fragment_heavy",
    "ADD_FRAGMENT_SIZE": "add_heavy",
    "REMOVE_GROUP_SIZE": "remove_heavy",
    "GROUP_SIZE_VERIFICATION": "remove_heavy",
}
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

_NUMBER_WORD_VALUES = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_ENUM_COUNT = (
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty)"
)
_ENUM_UNIT = r"(?:heavy atoms?|rings?)"
_ENUM_TOTAL_MARKER = re.compile(
    rf"\b(?:totaling|giving\s+(?:a\s+)?total\s+of|for\s+a\s+total\s+of)"
    rf"\s+(?P<total>{_ENUM_COUNT})\s+(?P<unit>{_ENUM_UNIT})\b",
    flags=re.IGNORECASE,
)
_ENUM_EQUALS_TOTAL = re.compile(
    rf"=\s*(?P<total>{_ENUM_COUNT})\s+(?P<unit>{_ENUM_UNIT})\b",
    flags=re.IGNORECASE,
)
_ENUM_TOTAL_FIRST = re.compile(
    rf"(?<![\w])(?P<total>{_ENUM_COUNT})\s+(?P<unit>{_ENUM_UNIT})\b",
    flags=re.IGNORECASE,
)
_ENUM_UNIT_FIRST = re.compile(
    rf"\b(?P<unit>heavy atoms?|SSSR\s+rings?|ring\s+count)\b"
    rf"[^.\n]{{0,70}}?:\s*(?P<total>{_ENUM_COUNT})(?=\s*\()",
    flags=re.IGNORECASE,
)
_ENUM_LEADING_COUNT = re.compile(
    rf"(?<![\w-])(?P<count>{_ENUM_COUNT})(?![\w-])(?=\s+(?:[A-Za-z(]))",
    flags=re.IGNORECASE,
)
_ENUM_OVERRIDE_COUNT = re.compile(
    rf"(?:\b(?:counts?\s+as|which\s+(?:counts?\s+as|is)|equals?)\s*|=\s*)"
    rf"(?P<count>{_ENUM_COUNT})(?:\s+rings?)?\b",
    flags=re.IGNORECASE,
)
_ENUM_BARE_PAREN_COUNT = re.compile(
    rf"\(\s*(?P<count>{_ENUM_COUNT})\s*\)",
    flags=re.IGNORECASE,
)
_ENUM_AMBIGUOUS_RING_ITEM = re.compile(
    r"\b(?:core|system|fused|bicyclic|tricyclic)\b",
    flags=re.IGNORECASE,
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
        spans |= _capture(
            rf"\b(?:is|atom|anchor)\s+[A-Z][a-z]?"
            rf"(?P<value>{needle})\b",
            natural_body,
        )
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
        # Bare phrases such as "3 heavy atoms" are only attributable to the
        # count node owned by this step.  Applying that pattern to every
        # changed count node makes sibling values collide (for example, an
        # edited remove_heavy claim being mistaken for the unchanged
        # add_heavy value in ADD_FRAGMENT_SIZE).  Explicitly labelled
        # cross-step mentions are still audited by the node-specific patterns
        # below, including REMOVE_GROUP = O (1 heavy atom) in another step.
        if _PRIMARY_FRAGMENT_COUNT_NODE_BY_STEP.get(step_name) == node_id:
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
    if _contains_explicit_enumeration(natural_body):
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


def _enumeration_count_value(value: str) -> int:
    lowered = value.lower()
    return _NUMBER_WORD_VALUES.get(lowered, int(lowered) if lowered.isdigit() else -1)


def _split_enumeration_items(text: str) -> tuple[str, ...]:
    """Split commas/conjunctions only at the current parenthesis depth."""

    parts: list[str] = []
    start = 0
    depth = 0
    index = 0
    while index < len(text):
        character = text[index]
        if character == "(":
            depth += 1
            index += 1
            continue
        if character == ")":
            depth = max(0, depth - 1)
            index += 1
            continue
        separator_end = None
        if depth == 0 and character == ",":
            previous = text[index - 1] if index else ""
            following = text[index + 1] if index + 1 < len(text) else ""
            if not (previous.isdigit() and following.isdigit()):
                separator_end = index + 1
        elif depth == 0 and character == "+":
            separator_end = index + 1
        elif depth == 0:
            conjunction = re.match(r"\s+(?:and|plus)\s+", text[index:], re.IGNORECASE)
            if conjunction is not None:
                separator_end = index + conjunction.end()
        if separator_end is None:
            index += 1
            continue
        part = text[start:index].strip(" ,+")
        if part:
            parts.append(part)
        start = separator_end
        index = separator_end
    tail = text[start:].strip(" ,+")
    if tail:
        parts.append(tail)
    return tuple(parts)


def _balanced_parenthetical(text: str, opening: int) -> tuple[str, int] | None:
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index], index + 1
    return None


def _enumeration_item_count(
    item: str,
    *,
    unit: str,
    allow_implicit_one: bool,
) -> tuple[int, int] | None:
    """Return ``(value, leaf_count)`` for one explicit component item."""

    override = _ENUM_OVERRIDE_COUNT.search(item)
    if override is not None:
        return _enumeration_count_value(override.group("count")), 1

    leading = tuple(_ENUM_LEADING_COUNT.finditer(item))
    if len(leading) == 1:
        if unit.startswith("ring") and _ENUM_AMBIGUOUS_RING_ITEM.search(item):
            # "one quinoline core" is one named system but can contain two
            # SSSR rings.  Without an explicit "counts as 2" it is unsafe to
            # infer a ring contribution, so the whole construction is skipped.
            return None
        return _enumeration_count_value(leading[0].group("count")), 1
    if len(leading) > 1:
        return None

    stripped = item.strip()
    if re.fullmatch(_ENUM_COUNT, stripped, flags=re.IGNORECASE):
        return _enumeration_count_value(stripped), 1

    bare = _ENUM_BARE_PAREN_COUNT.search(item)
    if bare is not None:
        return _enumeration_count_value(bare.group("count")), 1

    opening = item.find("(")
    if opening >= 0:
        parenthetical = _balanced_parenthetical(item, opening)
        if parenthetical is not None:
            nested, _ = parenthetical
            return _enumeration_sum(nested, unit=unit, allow_implicit_one=True)
    # Some corpus lists omit an explicit "one" for singular components, for
    # example "4 heavy atoms (methylene carbon, carbonyl carbon, two oxygens)".
    # Accept only a short, clearly singular item; plural or fused-system names
    # remain ambiguous and disable the audit.
    normalized = item.strip(" .")
    if (
        allow_implicit_one
        and normalized
        and len(normalized.split()) <= 6
        and re.search(r"\d", normalized) is None
        and re.search(
            r"\b(?:atoms|carbons|oxygens|nitrogens|sulfurs|rings|phenyls|"
            r"pyridines|methyls)\b",
            normalized,
            flags=re.IGNORECASE,
        )
        is None
        and _ENUM_AMBIGUOUS_RING_ITEM.search(normalized) is None
        and (
            not unit.startswith("ring")
            or (
                re.search(r"\bring\b", normalized, flags=re.IGNORECASE) is not None
                and re.search(
                    r"\b(?:adds?|introduces?|new|original)\b",
                    normalized,
                    flags=re.IGNORECASE,
                )
                is None
            )
        )
    ):
        return 1, 1
    return None


def _enumeration_sum(
    text: str,
    *,
    unit: str,
    allow_implicit_one: bool = False,
) -> tuple[int, int] | None:
    if unit.startswith("ring") and re.search(
        r"\b(?:adds?|introduces?)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return None
    items = _split_enumeration_items(text)
    if not items:
        return None
    values: list[tuple[int, int]] = []
    for item in items:
        parsed = _enumeration_item_count(
            item,
            unit=unit,
            allow_implicit_one=allow_implicit_one,
        )
        if parsed is None:
            return None
        values.append(parsed)
    leaf_count = sum(item[1] for item in values)
    if leaf_count < 2:
        return None
    return sum(item[0] for item in values), leaf_count


def _clause_start(text: str, end: int) -> int:
    return max(text.rfind(token, 0, end) for token in (".", "!", "?", "\n", ";")) + 1


def _contains_explicit_enumeration(text: str) -> bool:
    for match in (*_ENUM_TOTAL_MARKER.finditer(text), *_ENUM_EQUALS_TOTAL.finditer(text)):
        if _enumeration_sum(
            text[_clause_start(text, match.start()) : match.start()],
            unit=match.group("unit").lower(),
        ) is not None:
            return True
    for match in _ENUM_TOTAL_FIRST.finditer(text):
        opening = match.end()
        while opening < len(text) and text[opening].isspace():
            opening += 1
        if opening >= len(text) or text[opening] != "(":
            continue
        parenthetical = _balanced_parenthetical(text, opening)
        if parenthetical is not None and _enumeration_sum(
            parenthetical[0],
            unit=match.group("unit").lower(),
            allow_implicit_one=True,
        ) is not None:
            return True
    for match in _ENUM_UNIT_FIRST.finditer(text):
        opening = match.end()
        while opening < len(text) and text[opening].isspace():
            opening += 1
        parenthetical = (
            _balanced_parenthetical(text, opening)
            if opening < len(text) and text[opening] == "("
            else None
        )
        unit = "rings" if "ring" in match.group("unit").lower() else "heavy atoms"
        if parenthetical is not None and _enumeration_sum(
            parenthetical[0],
            unit=unit,
            allow_implicit_one=True,
        ) is not None:
            return True
    return False


def enumeration_violations(text: str) -> tuple[str, ...]:
    """Return explicit component enumerations whose displayed total is false.

    Detection is deliberately limited to totals whose unit is ``heavy atom(s)``
    or ``ring(s)``.  Ambiguous chemical descriptions (for example, an unnamed
    fused ``core`` without an explicit ring contribution) are ignored rather
    than guessed.
    """

    if type(text) is not str:
        raise TypeError("text must be str")
    violations: list[str] = []
    seen: set[tuple[int, int]] = set()

    def audit(prefix_start: int, body_end: int, match: re.Match[str]) -> None:
        unit = match.group("unit").lower()
        parsed = _enumeration_sum(text[prefix_start:body_end], unit=unit)
        if parsed is None or parsed[0] == _enumeration_count_value(match.group("total")):
            return
        span = (prefix_start, match.end())
        if span not in seen:
            seen.add(span)
            violations.append(text[span[0] : span[1]].strip())

    # Items first: "1 sulfur, 2 oxygens, ... totaling 9 heavy atoms".
    for match in _ENUM_TOTAL_MARKER.finditer(text):
        audit(_clause_start(text, match.start()), match.start(), match)

    # Equals form: "2 carbons, 1 sulfur, 2 oxygens = 5 heavy atoms".
    for match in _ENUM_EQUALS_TOTAL.finditer(text):
        audit(_clause_start(text, match.start()), match.start(), match)

    # Total first: "3 heavy atoms (one carbon, one oxygen, one nitrogen)".
    for match in _ENUM_TOTAL_FIRST.finditer(text):
        opening = match.end()
        while opening < len(text) and text[opening].isspace():
            opening += 1
        if opening >= len(text) or text[opening] != "(":
            continue
        parenthetical = _balanced_parenthetical(text, opening)
        if parenthetical is None:
            continue
        body, closing = parenthetical
        parsed = _enumeration_sum(
            body,
            unit=match.group("unit").lower(),
            allow_implicit_one=True,
        )
        if parsed is None or parsed[0] == _enumeration_count_value(match.group("total")):
            continue
        span = (match.start(), closing)
        if span not in seen:
            seen.add(span)
            violations.append(text[span[0] : span[1]].strip())

    # Unit first: "Heavy atoms: 4 (one carbon, one oxygen, two nitrogens)".
    for match in _ENUM_UNIT_FIRST.finditer(text):
        opening = match.end()
        while opening < len(text) and text[opening].isspace():
            opening += 1
        if opening >= len(text) or text[opening] != "(":
            continue
        parenthetical = _balanced_parenthetical(text, opening)
        if parenthetical is None:
            continue
        body, closing = parenthetical
        unit = "rings" if "ring" in match.group("unit").lower() else "heavy atoms"
        parsed = _enumeration_sum(body, unit=unit, allow_implicit_one=True)
        if parsed is None or parsed[0] == _enumeration_count_value(match.group("total")):
            continue
        span = (match.start(), closing)
        if span not in seen:
            seen.add(span)
            violations.append(text[span[0] : span[1]].strip())

    return tuple(violations)


__all__ = [
    "NUMERIC_NODE_IDS",
    "arithmetic_violations",
    "enumeration_violations",
    "loose_occurrence_spans",
    "requires_derivation_rewrite",
]
