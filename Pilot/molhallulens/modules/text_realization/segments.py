"""Compile untrusted prose/claim references into locally owned marked text.

The model never supplies a claim value, marker suffix, or enumeration item.
This module does not change planning, propagation, or the text validators.
"""
from __future__ import annotations

import re
from collections import Counter

from .claim_surfaces import claim_surface_pairs
from .enumeration_plan import enumeration_inventory


class SegmentContractError(ValueError):
    def __init__(self, code, message, *, expected=None, observed=None):
        super().__init__(message)
        self.code = code
        self.expected = expected
        self.observed = observed


def surface_values(claim):
    values = {"canonical": claim.after_text}
    pairs = claim_surface_pairs(claim.node_id, claim.before_text, claim.after_text)
    if len(pairs) > 1:
        values.update(symbol=claim.after_text, name=pairs[1][1], title_name=pairs[2][1])
    return values


def enumeration_blocks(step):
    from .poe_agent import HALLU_MARKER_PATTERN

    def describe(text):
        pieces, cursor = [], 0
        for match in HALLU_MARKER_PATTERN.finditer(text):
            if match.start() > cursor:
                pieces.append({"text": text[cursor:match.start()]})
            pieces.append({"claim_ref": match[1].rsplit(".", 1)[0], "planned_value": match[2]})
            cursor = match.end()
        if cursor < len(text):
            pieces.append({"text": text[cursor:]})
        return pieces
    return [
        {"enumeration_ref": f"enum_{index:02d}",
         "items": [describe(item) for item in inventory["items"]], "total": describe(inventory["total"]),
         "insertion_scope": "A complete standalone sentence, including its total and final period. Never put this reference inside parentheses or another enumeration."}
        for index, inventory in enumerate(enumeration_inventory(step.preserved_enumerations), 1)
    ]


def step_payload(step):
    from .poe_agent import FORMAL_MARKER, StepRewriteMode

    payload = step.to_prompt_dict()
    # Do not ask the model to copy marker-bearing reference examples. The
    # catalogue is explanatory; only enumeration_ref is accepted in output.
    payload.pop("preserved_enumerations")
    payload.pop("enumeration_inventory")
    prefix = f"Step {step.step_index} [{step.step_name}]: "
    payload["original_natural_body"] = payload.pop("original_step_text").split(FORMAL_MARKER, 1)[0][len(prefix):]
    payload["output_contract"] = {
        "scope": "Natural-language body only. Never include Step headers, FORMAL or Answer.",
        "allowed_reference": "patch_ref (preferred), or expanded occurrence_ref segments" if step.rewrite_mode is StepRewriteMode.OCCURRENCE_PATCH else "draft_ref (preferred), or expanded claim_ref/enumeration_ref segments",
        "formal_context": "modified_formal_ab is read-only context, never output text.",
    }
    payload["enumeration_blocks"] = enumeration_blocks(step)
    if step.rewrite_mode is StepRewriteMode.OCCURRENCE_PATCH:
        from .poe_agent import strip_hallucination_markers
        complete = compile_segments(response_segments_example(step), step)
        payload["complete_patch"] = {
            "recommended_segments": [{"patch_ref": "original_occurrences"}],
            "natural_body": strip_hallucination_markers(complete.split(FORMAL_MARKER, 1)[0][len(prefix):]),
            "instruction": "Explicitly select this full patch. An occurrence_ref-only list would delete all surrounding prose and is INVALID. patch_ref includes all required occurrences, even when the catalogue lists multiple occurrences.",
        }
    for claim, item in zip(step.affected_node_claims, payload["affected_node_claims"], strict=True):
        item["surfaces"] = surface_values(claim)
    if step.rewrite_mode is StepRewriteMode.DERIVATION_REWRITE:
        from .poe_agent import strip_hallucination_markers
        complete = compile_segments(response_segments_example(step), step)
        payload["complete_draft"] = {
            "draft_ref": "complete_derivation",
            "natural_body": strip_hallucination_markers(complete.split(FORMAL_MARKER, 1)[0][len(prefix):]),
            "source": "local_claim_sentence_and_preserved_inventory_draft",
            "instruction": "Review this complete candidate against the modified claims (not the original chemistry). Return draft_ref to select it explicitly, or submit your own valid segments. Do not reproduce its values manually.",
        }
    return payload


def response_segments_example(step):
    """Mode-correct examples using real IDs, never a misleading generic claim_ref."""
    from .poe_agent import StepRewriteMode

    if step.rewrite_mode is StepRewriteMode.OCCURRENCE_PATCH:
        from .poe_agent import FORMAL_MARKER
        prefix = f"Step {step.step_index} [{step.step_name}]: "
        body = step.original_step_text.split(FORMAL_MARKER, 1)[0][len(prefix):]
        segments, cursor = [], 0
        for occurrence in sorted(step.required_hallucination_occurrences, key=lambda o: o.original_start):
            if occurrence.original_start > cursor:
                segments.append({"text": body[cursor:occurrence.original_start]})
            segments.append({"occurrence_ref": occurrence.occurrence_id})
            cursor = occurrence.original_end
        if cursor < len(body):
            segments.append({"text": body[cursor:]})
        return segments
    # Complete prose, not node=value pseudo-code that invites a second model
    # rewrite. This candidate may be returned verbatim, but must still pass
    # compile_segments and every existing validator; it is not a fallback.
    phrases = {
        "anchor_idx": ("The attachment atom has index ", "."),
        "anchor_element": ("The attachment atom's element is ", "."),
        "leaving": ("The leaving group is ", "."),
        "add_fragment": ("The incoming fragment SMILES is ", "."),
        "remove_group": ("The removed group SMILES is ", "."),
        "remove_group_step1": ("The removed group SMILES is ", "."),
        "remove_group_step2": ("The removed group SMILES is ", "."),
        "fragment_heavy": ("The incoming fragment contains ", " heavy atoms."),
        "add_heavy": ("The incoming fragment contains ", " heavy atoms."),
        "remove_heavy": ("The removed group contains ", " heavy atoms."),
        "source_heavy": ("The source molecule contains ", " heavy atoms."),
        "product_heavy": ("The product molecule contains ", " heavy atoms."),
        "source_rings": ("The source molecule contains ", " rings."),
        "product_rings": ("The product molecule contains ", " rings."),
        "heavy_delta": ("The heavy atom delta is ", "."),
        "ring_delta": ("The ring delta is ", "."),
        "product": ("The product SMILES is ", "."),
        "final_answer": ("The final product SMILES is ", "."),
    }
    from .poe_agent import HALLU_MARKER_PATTERN
    enum_counts = Counter(m[1].rsplit(".", 1)[0] for clause in step.preserved_enumerations
                          for m in HALLU_MARKER_PATTERN.finditer(clause))
    segments = []
    for claim in step.affected_node_claims:
        if not claim.parent_node_id:
            lead, tail = phrases.get(claim.node_id, (claim.node_id + " = ", "."))
            count = 1 if claim.required_occurrence_count is None else max(0, claim.required_occurrence_count - enum_counts[claim.node_id])
            for _ in range(count):
                segments.extend([{"text": lead}, {"claim_ref": claim.node_id}, {"text": tail + " "}])
    for block in enumeration_blocks(step):
        segments.extend([{"enumeration_ref": block["enumeration_ref"]}, {"text": " "}])
    if segments and "text" in segments[-1]:
        segments[-1]["text"] = segments[-1]["text"].rstrip()
        if not segments[-1]["text"]:
            segments.pop()
    return segments


def compile_segments(segments, step):
    from .poe_agent import HALLU_MARKER_PATTERN, StepRewriteMode, validate_rewritten_step_text, FORMAL_MARKER

    if type(segments) is not list or not segments:
        raise SegmentContractError("segments_shape", "segments must be a non-empty array")
    if any(type(s) is dict and "draft_ref" in s for s in segments):
        if step.rewrite_mode is not StepRewriteMode.DERIVATION_REWRITE or segments != [{"draft_ref": "complete_derivation"}]:
            raise SegmentContractError("draft_reference", "draft_ref is an exclusive derivation_rewrite operation; use exactly [{draft_ref:complete_derivation}]")
        return compile_segments(response_segments_example(step), step)
    if any(type(s) is dict and "patch_ref" in s for s in segments):
        if step.rewrite_mode is not StepRewriteMode.OCCURRENCE_PATCH or segments != [{"patch_ref": "original_occurrences"}]:
            raise SegmentContractError("patch_reference", "patch_ref is an exclusive occurrence_patch operation; use exactly [{patch_ref:original_occurrences}]")
        # Explicit model selection only. Expand the exact same original slots,
        # then execute the normal compiler/validators. No invalid-response repair.
        return compile_segments(response_segments_example(step), step)
    claims = {c.node_id: c for c in step.affected_node_claims}
    occurrences = {o.occurrence_id: o for o in step.required_hallucination_occurrences}
    enums = {f"enum_{i:02d}": clause for i, clause in enumerate(step.preserved_enumerations, 1)}
    counts, used_occurrences, used_enums = Counter(), set(), set()

    def marker(node, value):
        counts[node] += 1
        if counts[node] > 99:
            raise SegmentContractError("occurrence_limit", "a node exceeds the two-digit occurrence limit")
        return f"[[HALLU:{node}.{counts[node]:02d}]]{value}[[/HALLU]]"

    parts = []
    for segment in segments:
        if type(segment) is not dict:
            raise SegmentContractError("segment_shape", "each segment must be an object")
        keys = set(segment)
        if keys == {"text"}:
            value = segment["text"]
            if type(value) is not str or any(t in value for t in ("[[", "]]", "\r", "\x00")) or re.search(r"\b(?:FORMAL|Answer)\s*:", value, re.I):
                raise SegmentContractError("literal_channel", "literal text cannot contain markers, FORMAL, Answer, CR or NUL")
            parts.append(value)
        elif keys in ({"claim_ref"}, {"claim_ref", "surface"}):
            node, surface = segment["claim_ref"], segment.get("surface", "canonical")
            if step.rewrite_mode is not StepRewriteMode.DERIVATION_REWRITE:
                raise SegmentContractError("wrong_reference_type",
                    "This step is occurrence_patch: claim_ref is forbidden. Use each exact occurrence_ref from the supplied patch example, preserving the surrounding text.",
                    expected=sorted(occurrences), observed=node)
            if type(node) is not str or node not in claims or step.rewrite_mode is not StepRewriteMode.DERIVATION_REWRITE:
                raise SegmentContractError("unknown_claim", "claim_ref must name an affected derivation claim", expected=sorted(claims), observed=node)
            values = surface_values(claims[node])
            if type(surface) is not str or surface not in values:
                raise SegmentContractError("unknown_surface", "surface is not allowed for this node", expected=sorted(values), observed=surface)
            parts.append(marker(node, values[surface]))
        elif keys == {"occurrence_ref"}:
            ref = segment["occurrence_ref"]
            if type(ref) is not str or ref not in occurrences or ref in used_occurrences:
                raise SegmentContractError("occurrence_reference", "occurrence_ref must name one unused patch occurrence", expected=sorted(set(occurrences) - used_occurrences), observed=ref)
            used_occurrences.add(ref)
            # ID belongs to the local occurrence inventory, not model numbering.
            parts.append(f"[[HALLU:{ref}]]{occurrences[ref].after_text}[[/HALLU]]")
        elif keys == {"enumeration_ref"}:
            ref = segment["enumeration_ref"]
            if type(ref) is not str or ref not in enums or ref in used_enums:
                raise SegmentContractError("enumeration_reference", "enumeration_ref must name one unused planned inventory", expected=sorted(set(enums) - used_enums), observed=ref)
            used_enums.add(ref)
            parts.append(HALLU_MARKER_PATTERN.sub(lambda m: marker(m[1].rsplit(".", 1)[0], m[2]), enums[ref]))
        else:
            raise SegmentContractError("segment_shape", "segment must contain exactly text, claim_ref[/surface], occurrence_ref or enumeration_ref", observed=sorted(keys))
    if used_enums != set(enums):
        raise SegmentContractError("missing_enumeration", "every planned enumeration must be referenced once", expected=sorted(set(enums) - used_enums))
    if used_occurrences != set(occurrences):
        raise SegmentContractError("missing_occurrence", "every patch occurrence must be referenced once", expected=sorted(set(occurrences) - used_occurrences))
    body = "".join(parts)
    prefix = f"Step {step.step_index} [{step.step_name}]: "
    return validate_rewritten_step_text(prefix + body + FORMAL_MARKER + step.modified_formal_ab, step)


def local_copy(step):
    from .poe_agent import FORMAL_MARKER, validate_rewritten_step_text
    head = step.original_step_text.split(FORMAL_MARKER, 1)[0]
    return validate_rewritten_step_text(head + FORMAL_MARKER + step.modified_formal_ab, step)
