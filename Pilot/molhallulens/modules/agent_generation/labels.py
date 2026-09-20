"""Compile frozen-reference patches and project explicit claims onto tokens.

All offsets are half-open Python Unicode character offsets, never byte offsets.
The compiler copies every untouched source slice verbatim. It does not verify
chemistry: ``kind`` is an explicit claim annotation supplied by the audited
generation pipeline, not a conclusion inferred from a textual difference.

``edits`` describe every replacement, including stylistic material and deletions.
``spans`` describe only explicitly annotated semantic errors in the output. A
patch's optional ``error_spans`` use replacement-relative offsets and may override
its kind/node/root provenance. An empty list marks a purely stylistic edit;
omitting it annotates the entire nonempty replacement as the stated error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


_KINDS = ("root_error", "propagated_error")


def _offsets(start: Any, end: Any, length: int, *, nonempty: bool = False) -> None:
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start <= end <= length
        or (nonempty and start == end)
    ):
        raise ValueError("offsets must be valid half-open character positions")


def _annotation(data: Mapping[str, Any]) -> dict[str, Any]:
    kind, node_id, root_ids = data.get("kind"), data.get("node_id"), data.get("root_ids")
    if kind not in _KINDS:
        raise ValueError("kind must be root_error or propagated_error")
    if not isinstance(node_id, str) or not node_id.strip():
        raise ValueError("node_id must be nonempty text")
    if (
        not isinstance(root_ids, (list, tuple))
        or not root_ids
        or any(not isinstance(root, str) or not root.strip() for root in root_ids)
        or len(set(root_ids)) != len(root_ids)
    ):
        raise ValueError("root_ids must contain distinct nonempty root identifiers")
    result = {"kind": kind, "node_id": node_id, "root_ids": list(root_ids)}
    if "evidence" in data:
        result["evidence"] = deepcopy(data["evidence"])
    return result


def compile_patches(reference: str, patches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply nonoverlapping patches against *reference*, without mutating inputs.

    Each patch needs ``start``, ``end`` and ``replacement``. Annotated errors
    additionally require ``kind``, ``node_id`` and ``root_ids``, on the patch or
    on each ``error_spans`` entry. If supplied, ``original`` must match exactly.
    Patches may be supplied out of order; ambiguous edits sharing a start are
    rejected, as are source overlaps and replacements that do not change text.
    Deletions retain edit provenance but have no output characters to label.
    """
    if not isinstance(reference, str):
        raise TypeError("reference must be text")
    if not isinstance(patches, (list, tuple)):
        raise TypeError("patches must be a list or tuple")

    prepared = []
    for index, patch in enumerate(patches):
        if not isinstance(patch, Mapping):
            raise TypeError("each patch must be an object")
        start, end = patch.get("start"), patch.get("end")
        _offsets(start, end, len(reference))
        replacement = patch.get("replacement")
        if not isinstance(replacement, str):
            raise TypeError("replacement must be text")
        original = reference[start:end]
        if "original" in patch and patch["original"] != original:
            raise ValueError("original does not match the frozen reference")
        if replacement == original:
            raise ValueError("patch must change the source text")
        annotations = patch.get("error_spans", [{"start": 0, "end": len(replacement)}] if replacement else [])
        if not isinstance(annotations, (list, tuple)):
            raise TypeError("error_spans must be a list or tuple")
        semantic_spans = []
        for annotation in annotations:
            if not isinstance(annotation, Mapping):
                raise TypeError("each error span must be an object")
            span_start, span_end = annotation.get("start"), annotation.get("end")
            _offsets(span_start, span_end, len(replacement), nonempty=True)
            if "text" in annotation and annotation["text"] != replacement[span_start:span_end]:
                raise ValueError("error span text does not match the replacement")
            metadata = _annotation({**patch, **annotation})
            semantic_spans.append({**metadata, "start": span_start, "end": span_end})
        patch_provenance = {
            key: deepcopy(patch[key]) for key in ("kind", "node_id", "root_ids", "evidence") if key in patch
        }
        if all(key in patch for key in ("kind", "node_id", "root_ids")):
            patch_provenance = _annotation(patch)
        prepared.append((start, end, index, replacement, semantic_spans, patch_provenance))

    prepared.sort(key=lambda item: (item[0], item[1]))
    previous_start = previous_end = -1
    for start, end, *_ in prepared:
        if start < previous_end or start == previous_start:
            raise ValueError("patches overlap or have an ambiguous shared start")
        previous_start, previous_end = start, end

    pieces, spans, edits = [], [], []
    source_cursor = output_cursor = 0
    for start, end, index, replacement, semantic_spans, patch_provenance in prepared:
        untouched = reference[source_cursor:start]
        pieces.append(untouched)
        output_cursor += len(untouched)
        edit_start = output_cursor
        pieces.append(replacement)
        output_cursor += len(replacement)
        span_indices = []
        for annotation in semantic_spans:
            absolute_start = edit_start + annotation["start"]
            absolute_end = edit_start + annotation["end"]
            span_indices.append(len(spans))
            spans.append({
                **annotation,
                "start": absolute_start,
                "end": absolute_end,
                "text": replacement[annotation["start"]:annotation["end"]],
                "source_start": start,
                "source_end": end,
                "patch_index": index,
            })
        edits.append({
            **patch_provenance,
            "patch_index": index,
            "source_start": start,
            "source_end": end,
            "original": reference[start:end],
            "start": edit_start,
            "end": output_cursor,
            "text": replacement,
            "error_span_indices": span_indices,
        })
        source_cursor = end
    pieces.append(reference[source_cursor:])
    return {"text": "".join(pieces), "spans": spans, "edits": edits}


def label_tokens(text: str, spans: Sequence[Mapping[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    """Label a supplied fast tokenizer's exact tokens without loading a model.

    Tokenization uses ``add_special_tokens=False`` and the tokenizer's own offset
    mapping. ``text`` is the exact source substring; ``token`` is the vocabulary
    spelling, which may differ. Labels are multilabel when tokens intersect
    several claims. ``unchanged`` means characters outside annotated errors;
    consult compiled ``edits`` to distinguish stylistic changes from source copy.
    Whitespace with no tokenizer offset has no synthetic token inserted for it.
    """
    if not isinstance(text, str):
        raise TypeError("text must be text")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("a fast tokenizer with character offsets is required")
    checked_spans = []
    for span in spans:
        if not isinstance(span, Mapping):
            raise TypeError("each span must be an object")
        start, end = span.get("start"), span.get("end")
        _offsets(start, end, len(text), nonempty=True)
        if "text" in span and span["text"] != text[start:end]:
            raise ValueError("span text does not match the tokenized text")
        checked_spans.append({**span, **_annotation(span)})

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(ids) != len(offsets):
        raise ValueError("token IDs and offsets must have equal length")
    tokens = tokenizer.convert_ids_to_tokens(ids)
    if len(tokens) != len(ids):
        raise ValueError("token strings and IDs must have equal length")
    tokenizer_name = getattr(tokenizer, "name_or_path", type(tokenizer).__name__)
    records = []
    for index, (token_id, token, offset) in enumerate(zip(ids, tokens, offsets)):
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise ValueError("token offsets must have two positions")
        start, end = offset
        _offsets(start, end, len(text))
        overlaps = []
        for span_index, span in enumerate(checked_spans):
            overlap_start, overlap_end = max(start, span["start"]), min(end, span["end"])
            if overlap_start < overlap_end:
                overlaps.append({
                    **deepcopy(span), "span_index": span_index,
                    "overlap_start": overlap_start, "overlap_end": overlap_end,
                })
        labels = [kind for kind in _KINDS if any(span["kind"] == kind for span in overlaps)]
        covered_until = start
        has_unannotated = start == end
        for overlap_start, overlap_end in sorted((span["overlap_start"], span["overlap_end"]) for span in overlaps):
            has_unannotated |= overlap_start > covered_until
            covered_until = max(covered_until, overlap_end)
        if has_unannotated or covered_until < end:
            labels.append("unchanged")
        records.append({
            "token_index": index,
            "token_id": int(token_id),
            "token": token,
            "text": text[start:end],
            "start": start,
            "end": end,
            "offset": [start, end],
            "tokenizer": tokenizer_name,
            "labels": labels,
            "root_ids": list(dict.fromkeys(root for span in overlaps for root in span["root_ids"])),
            "node_ids": list(dict.fromkeys(span["node_id"] for span in overlaps)),
            "overlaps": overlaps,
        })
    return records


def label_prompt_tokens(
    prompt: str,
    trace: str,
    spans: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    trace_start: int,
) -> dict[str, Any]:
    """Project trace annotations into the exact rendered model prompt.

    The caller supplies the known trace location from prompt construction; this
    function never guesses among repeated occurrences. Character annotations
    are shifted before tokenizing the entire prompt once. Tokens may include
    both trace and surrounding prompt characters and retain mixed labels.
    Standalone CoT token IDs or indices must never be substituted for these.
    Returned IDs use the same no-added-special-token policy as model inference
    over an already rendered chat template.
    """
    if not isinstance(prompt, str) or not isinstance(trace, str):
        raise TypeError("prompt and trace must be text")
    _offsets(trace_start, trace_start + len(trace) if type(trace_start) is int else None, len(prompt))
    trace_end = trace_start + len(trace)
    if prompt[trace_start:trace_end] != trace:
        raise ValueError("trace does not match the supplied prompt character range")
    shifted = []
    for span in spans:
        if not isinstance(span, Mapping):
            raise TypeError("each span must be an object")
        start, end = span.get("start"), span.get("end")
        _offsets(start, end, len(trace), nonempty=True)
        shifted.append({**deepcopy(span), "start": trace_start + start, "end": trace_start + end})
    tokens = label_tokens(prompt, shifted, tokenizer)
    return {
        "label_scope": "full_rendered_prompt",
        "trace_start": trace_start,
        "trace_end": trace_end,
        "prompt_token_ids": [token["token_id"] for token in tokens],
        "tokens": tokens,
    }
