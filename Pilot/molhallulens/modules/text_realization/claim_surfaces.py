"""Conservative surface forms for one claim, not new chemistry or graph edits."""

import re


ELEMENT_NAMES = {
    "B": "boron", "C": "carbon", "N": "nitrogen", "O": "oxygen",
    "F": "fluorine", "Si": "silicon", "P": "phosphorus", "S": "sulfur",
    "Cl": "chlorine", "Br": "bromine", "I": "iodine",
}
# Keep quoted chemical values and unquoted SMILES tokens opaque to prose rules.
# In particular, N(C) must never expose its branch C as a prose '(C)' mention.
_PROTECTED = re.compile(
    r'"[^"\n]*"|\'[^\'\n]*\'|(?<!\w)[A-Za-z0-9@+\-]*'
    r'[\[(][A-Za-z0-9@+:=()\[\]#/\\\-]*[\])][A-Za-z0-9@+:=()\[\]#/\\\-]*'
)


def protected_spans(text):
    return tuple(match.span() for match in _PROTECTED.finditer(text))


def claim_surface_pairs(node_id, before, after):
    """Exact reversible alternatives; arbitrary chemical synonyms are not allowed."""
    pairs = [(before, after)]
    if node_id == "anchor_element" and before in ELEMENT_NAMES and after in ELEMENT_NAMES:
        old, new = ELEMENT_NAMES[before], ELEMENT_NAMES[after]
        pairs.extend(((old, new), (old.capitalize(), new.capitalize())))
    return tuple(pairs)


def anchor_element_spans(value, text):
    """Bind names/symbols to an explicit anchor, never a quoted fragment branch."""
    symbol = next((s for s, name in ELEMENT_NAMES.items() if name == value.lower()), value)
    name = ELEMENT_NAMES.get(symbol)
    variants = [re.escape(symbol)] + ([name] if name else [])
    needle = "(?:" + "|".join(variants) + ")"
    patterns = (
        rf'\belement\s*(?:symbol)?\s*(?:=|:)?\s*["\']?(?P<value>{needle})(?![a-z])',
        rf'\b(?:anchor|attachment\s+atom)(?:\s*(?:is|=|:|as)\s*'
        rf'(?:the\s+)?(?:[a-z-]+\s+){{0,3}}|\s+)'
        rf'(?P<value>{needle})(?![a-z])',
        rf'\b(?:anchor|attachment\s+atom)\b[^.;\n"\']{{0,80}}?'
        rf'\(\s*(?P<value>{needle})\s*\)',
        rf'\b(?P<value>{needle})\s+atom\b[^.;\n]{{0,50}}?\b(?:is|as)\s+(?:the\s+)?ANCHOR\b',
    )
    protected = protected_spans(text)
    result = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            start, end = match.span("value")
            # Standalone prose (C) is allowed, but not N(C), quoted SMILES,
            # or mapped atoms. Explicit element="C" is a labelled value.
            overlaps = [(a, b) for a, b in protected if start < b and a < end]
            if overlaps and not all(
                text[a:b] == "(" + text[start:end] + ")"
                or (text[a:b] in {'"' + text[start:end] + '"', "'" + text[start:end] + "'"}
                    and re.search(r"\belement\s*(?:symbol)?\s*(?:=|:)?\s*$", text[max(0, a-30):a], re.I))
                for a, b in overlaps
            ):
                continue
            # Chemical symbols are case-sensitive; don't mistake 'in' or
            # aromatic lowercase symbols for the uppercase element claim.
            surface = text[start:end]
            if surface != symbol and (not name or surface not in {name, name.capitalize()}):
                continue
            result.add((start, end))
    return tuple(sorted(result))


def patch_prose_signature(text):
    """Allow layout, scoped count verbs and unit inflection, not paraphrases.

    Chemical strings remain byte-exact. Numbers, negation, units, punctuation,
    and all other vocabulary remain significant. Both pair sides inherit any
    accepted wording from H; this is not an H-versus-source identity test.
    """
    def prose(part):
        part = re.sub(
            r"\b((?:The |the )?(?:source|product))(?: molecule)? (?:has|contains)(?=\s)",
            r"\1 contains", part,
        )
        part = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)\s+(heavy\s+)?atoms?\b", r"\1 \2atoms", part)
        part = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)\s+rings?\b", r"\1 rings", part)
        return re.sub(r"\s+", " ", part)
    pieces, cursor = [], 0
    for start, end in protected_spans(text):
        pieces.extend((prose(text[cursor:start]), text[start:end]))
        cursor = end
    pieces.append(prose(text[cursor:]))
    return "".join(pieces)
