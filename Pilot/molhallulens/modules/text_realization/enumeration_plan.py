"""Preserve source breakdowns and expose count consequences as text claims.

No chemistry is inferred here: component truths are source-trace claims, and
the altered component count is an explicitly false consequence of a total edit.
Ambiguous total ownership and unsupported count syntax fail closed.
"""
from __future__ import annotations

import re
from collections import Counter

from . import occurrence_audit as audit


def enumeration_inventory(clauses):
    """Reference items are constraints, not a sentence to copy verbatim."""
    result = []
    for clause in clauses:
        body, separator, total = clause.rpartition(" = ")
        if not separator:
            raise ValueError("enumeration_inventory: missing total boundary")
        result.append({"items": list(audit._split_enumeration_items(body)), "total": total.rstrip(".")})
    return result


def _inventory_normalize(text):
    # Marker identity/value remain significant; only its occurrence suffix varies.
    text = re.sub(r"(\[\[HALLU:[a-z0-9_]+)\.\d{2}", r"\1__occ", text)
    # A bare symbolic inventory (N + C + O) also has case-sensitive chemistry.
    text = re.sub(r"(?<!\w)(?:Cl|Br|Si|[BCNOFPSI])(?!\w)",
                  lambda match: " chem_" + match[0].encode("utf-8").hex() + " ", text)
    # Parenthesized chemical labels such as (C) and (c) are case-sensitive.
    text = re.sub(
        r"\(([^()\s]*[A-Za-z][^()\s]*)\)",
        lambda match: " chem_" + match[1].encode("utf-8").hex() + " ",
        text,
    )
    # Keep signs and chemical bond/stereo syntax significant. In particular,
    # punctuation tolerance must never turn -1 into 1 or C=O into C-O.
    text = re.sub(r"[^\w=#@+\-/\\\[\]]+", " ", text.lower()).strip()
    singulars = {word + "s": word for word in (
        "carbon", "oxygen", "nitrogen", "sulfur", "fluorine", "chlorine",
        "bromine", "iodine", "phosphorus", "atom", "ring",
    )}
    connectors = {"comprising": "comprises", "consisting": "consists"}
    return " ".join(connectors.get(word, singulars.get(word, word)) for word in text.split())


def _inventory_claim_clauses(protected):
    """Separate explicitly introduced claims, not arbitrary extra markers.

    A source breakdown and a product count may share a sentence. Do not split
    inside parentheses: that could hide an extra component in the breakdown.
    """
    boundary = re.compile(
        r"(?:[,;]\s*|\s+)(?=(?:(?:and|while|whereas|but)\s+)?"
        r"(?:the\s+)?(?:"
        r"(?:source|product)(?:\s+molecule)?(?:\s+(?:ring|heavy.atom)\s+count)?"
        r"\s+(?:also\s+)?(?:contains?|has|have|is|are)\b"
        r"|(?:difference|ring\s+delta|heavy.atom\s+delta)\s+(?:is|equals?)\b"
        r"|(?:RING_DELTA|HEAVY_ATOM_DELTA)\s*=))", re.I,
    )
    result = []
    for sentence in re.split(r"[.!?]+(?:\s+|$)", protected):
        depths = []
        depth = 0
        for character in sentence:
            depths.append(depth)
            if character == "(":
                depth += 1
            elif character == ")":
                depth = max(0, depth - 1)
        cursor = 0
        for match in boundary.finditer(sentence):
            if depths[match.start()] != 0:
                continue
            if sentence[cursor:match.start()].strip():
                result.append(sentence[cursor:match.start()])
            cursor = match.end()
        if sentence[cursor:].strip():
            result.append(sentence[cursor:])
    return result


def validate_enumeration_inventory(marked_body, clauses):
    """Match bound count/component items within a claim clause.

    Reordering, punctuation, whitespace and known noun plurals are accepted.
    The intervening prose is deliberately limited to connectors, so an extra
    component/count cannot be silently ignored. Ambiguous paraphrases reject.
    """
    if not clauses:
        return
    protected = re.sub(r"(\[\[HALLU:[a-z0-9_]+)\.\d{2}", r"\1__occ", marked_body)
    sentences = _inventory_claim_clauses(protected)
    connectors = set("the a an this that these those fragment molecule source product incoming outgoing group contains contain has have comprises comprise consists consist of is are with and plus also totaling totals total giving gives for to equal equals counting count precisely heavy atom ring respectively breakdown its it includes include".split())
    connectors.update({"=", "+"})
    connectors.update({"while", "whereas", "but"})
    between_items = set("and plus also totaling totals total giving gives for a of to equal equals respectively comprises comprise consists consist contains contain is are = +".split())
    used = set()
    for index, inventory in enumerate(enumeration_inventory(clauses), start=1):
        required = Counter(_inventory_normalize(item) for item in inventory["items"])
        total = _inventory_normalize(inventory["total"])
        matched = False
        best = None
        for sentence_index, sentence in enumerate(sentences):
            if sentence_index in used:
                continue
            remaining = " " + _inventory_normalize(sentence) + " "
            missing = []
            # Longer items first avoids swallowing a suffix of another item.
            for item, count in sorted(required.items(), key=lambda pair: -len(pair[0])):
                for _ in range(count):
                    needle = " " + item + " "
                    if needle not in remaining:
                        missing.append(item)
                    else:
                        remaining = remaining.replace(needle, " item_slot ", 1)
            total_needle = " " + total + " "
            missing_total = total_needle not in remaining
            if missing or missing_total:
                score = (len(missing) + int(missing_total), 0)
                detail = f"missing_items={missing!r}, missing_total={missing_total}"
                if best is None or score < best[0]:
                    best = (score, detail)
                continue
            # Repeated totals are the same claim, not extra components. Every
            # repetition must have the exact value AND marker binding; the Poe
            # contract separately checks unique/consecutive occurrence IDs.
            remaining = remaining.replace(total_needle, " total_slot ")
            extra = set()
            seen_value = False
            for token in remaining.split():
                if token in {"item_slot", "total_slot"}:
                    seen_value = True
                elif token not in (between_items if seen_value else connectors):
                    extra.add(token)
            if extra:
                score = (0, len(extra))
                if best is None or score < best[0]:
                    best = (score, f"unexpected_tokens={sorted(extra)!r}")
                continue
            used.add(sentence_index)
            matched = True
            break
        if not matched:
            raise ValueError(
                f"enumeration_preservation: inventory {index} missing/altered item, "
                f"wrong marker binding, extra component, or unsupported phrasing; "
                f"{best[1] if best else 'no unused enumeration sentence'}; "
                f"expected items={inventory['items']!r}, total={inventory['total']!r}"
            )


def enumeration_plan(body: str, claims, *, step_name: str):
    """Return preserved marked breakdowns and explicit derived-claim dictionaries."""
    entries = []
    for match in audit._ENUM_SYMBOLIC.finditer(body):
        entries.append((match.start("items"), match.end("items"), match))
    for pattern in (audit._ENUM_TOTAL_MARKER, audit._ENUM_EQUALS_TOTAL):
        for match in pattern.finditer(body):
            start = audit._clause_start(body, match.start())
            entries.append((start, match.start(), match))
    for pattern in (audit._ENUM_TOTAL_FIRST, audit._ENUM_UNIT_FIRST):
        for match in pattern.finditer(body):
            opening = match.end()
            while opening < len(body) and body[opening].isspace():
                opening += 1
            if opening < len(body) and body[opening] == "(":
                parenthetical = audit._balanced_parenthetical(body, opening)
                if parenthetical:
                    entries.append((opening + 1, parenthetical[1] - 1, match))
    clauses, derived, seen = [], [], set()
    for start, end, match in sorted(entries, key=lambda entry: entry[0]):
        if (start, end) in seen:
            continue
        unit = "rings" if "ring" in match.group("unit").lower() else "heavy atoms"
        breakdown = body[start:end].strip(" ,:=\n")
        if "items" in match.re.groupindex:
            # Make implicit atomic coefficients explicit on BOTH pair sides.
            # Never drop an atom/component to force the edited sum to agree.
            breakdown = " + ".join(
                item if re.match(r"\d+\s", item) else "1 " + item
                for item in audit._split_enumeration_items(breakdown)
            )
        leadins = list(re.finditer(r"\b(?:consists?\s+of|contains?|comprises?|includes?)\s+", breakdown, re.I))
        if leadins:
            breakdown = breakdown[leadins[-1].end():]
        parsed = audit._enumeration_sum(breakdown, unit=unit, allow_implicit_one=True)
        if parsed is None:
            continue
        if ":" in breakdown:
            # Remove only the lead-in (e.g. "Count heavy (non-hydrogen) atoms:").
            tail = breakdown.rsplit(":", 1)[1].strip()
            if audit._enumeration_sum(tail, unit=unit, allow_implicit_one=True) == parsed:
                breakdown = tail
        seen.add((start, end))
        total_text = match.group("total")
        total = audit._enumeration_count_value(total_text)
        if parsed[0] != total:
            raise ValueError(f"enumeration_source_inconsistent: {breakdown!r}, {parsed[0]} != {total}")
        owners = []
        for claim in claims:
            if getattr(claim, "parent_node_id", None):
                continue
            if claim.node_id not in audit.NUMERIC_NODE_IDS:
                continue
            spans = audit.loose_occurrence_spans(
                claim.node_id, claim.before_text, body, step_name=step_name,
            )
            if any(a <= match.start("total") and b >= match.end("total") for a, b in spans):
                owners.append(claim)
        if len(owners) > 1:
            raise ValueError("enumeration_ambiguous_owner: multiple total claims")
        if owners:
            owner = owners[0]
            delta = int(owner.after_text) - total
            items = audit._split_enumeration_items(breakdown)
            groups, cursor = [], 0
            for item in items:
                position = breakdown.index(item, cursor)
                cursor = position + len(item)
                item_count = audit._enumeration_item_count(item, unit=unit, allow_implicit_one=True)
                if item_count is None or item_count[1] != 1:
                    raise ValueError(f"enumeration_unsupported_components: {item!r}")
                count = item_count[0]
                override = audit._ENUM_OVERRIDE_COUNT.search(item)
                matches = [override] if override else list(re.finditer(
                    rf"(?<![\w-])(?P<count>{audit._ENUM_COUNT})(?![\w-])", item, re.I))
                tokens = [(position + t.start("count"), position + t.end("count"), t.group("count"))
                          for t in matches if audit._enumeration_count_value(t.group("count")) == count]
                if not tokens:
                    # An implicit singular item is preserved inside its marked phrase.
                    tokens = [(position, cursor, item)]
                groups.append((count, tokens))
            if sum(count for count, _ in groups) != total:
                raise ValueError("enumeration_unsupported_components: grouped counts disagree")
            for index, (count, tokens) in reversed(list(enumerate(groups, start=1))):
                if not delta:
                    break
                change = delta if delta > 0 else max(delta, -count)
                delta -= change
                if not change:
                    continue
                for token_index, (a, b, before) in reversed(list(enumerate(tokens, start=1))):
                    after = str(count + change)
                    if audit._enumeration_count_value(before) < 0:
                        after += " " + before
                    node = f"{owner.node_id}__enumeration_{len(clauses) + 1}_{index}_{token_index}"
                    derived.append(dict(node_id=node, before_text=before, after_text=after,
                                        parent_node_id=owner.node_id))
                    breakdown = (breakdown[:a] + f"[[HALLU:{node}.01]]{after}[[/HALLU]]" + breakdown[b:])
            if delta:
                raise ValueError("enumeration_negative_total: cannot preserve nonnegative counts")
            total_text = f"[[HALLU:{owner.node_id}.01]]{owner.after_text}[[/HALLU]]"
        clauses.append(f"{breakdown} = {total_text} {unit}.")
    if audit._contains_explicit_enumeration(body) and not clauses:
        raise ValueError("enumeration_unparsed: source breakdown must not be deleted")
    return tuple(clauses), tuple(derived)
