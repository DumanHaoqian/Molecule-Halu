"""Bounded, redacted diagnostics; never serialize transports or environments."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from difflib import SequenceMatcher


def redact(text, secret=None):
    if secret:
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)\b(?:Authorization\s*[:=]\s*(?:Bearer\s+)?|Bearer\s+)\S+", "[REDACTED_AUTHORIZATION]", text)
    text = re.sub(r"(?i)(?:POE_API_KEY|api[_-]?key|token)\s*[=:]\s*[^\s,;]+", "[REDACTED_CREDENTIAL]", text)
    return re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED_KEY]", text)


def safe_value(value, *, secret, limit):
    if isinstance(value, str):
        cleaned = redact(value, secret)
        return cleaned[:limit] + ("…[truncated]" if len(cleaned) > limit else "")
    if isinstance(value, dict):
        return {str(k): safe_value(v, secret=secret, limit=limit) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_value(v, secret=secret, limit=limit) for v in value]
    return value


def rejection_code(error):
    if getattr(error, "code", None):
        return error.code
    text = str(error).lower()
    for needle, code in (
        ("outside required", "unapproved_prose_change"),
        ("enumeration", "false_enumeration"), ("component sum", "false_enumeration"),
        ("arithmetic", "false_arithmetic"), ("stale", "stale_claim"),
        ("unmarked", "unmarked_claim"), ("omitted", "missing_claim"),
        ("marker count", "pair_occurrence_count"), ("header", "step_header"),
    ):
        if needle in text:
            return code
    return "step_text_contract"


def make_diagnostic(*, run_id, origin_id, model, version, prompt_hash, step, attempt, error, response, secret, limit):
    raw = response if type(response) is str else json.dumps(response, ensure_ascii=False)
    data = {
        "run_id": run_id, "origin_id": origin_id, "model": model,
        "protocol_version": version, "prompt_sha256": prompt_hash,
        "step_index": step.step_index if step else None,
        "step_name": step.step_name if step else None,
        "rewrite_mode": step.rewrite_mode.value if step else None,
        "attempt": attempt, "error_code": rejection_code(error),
        "message": str(error), "expected": getattr(error, "expected", None),
        "observed": getattr(error, "observed", None),
        "original_step_text": step.original_step_text if step else None,
        "modified_formal_ab": step.modified_formal_ab if step else None,
        "affected_claims": [c.to_prompt_dict() for c in step.affected_node_claims] if step else [],
        "response_excerpt": raw,
        "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }
    data["node_id"] = next((c.node_id for c in step.affected_node_claims if repr(c.node_id) in str(error)), None) if step else None
    occurrence = re.search(r"\b[a-z][a-z0-9_]*\.\d{2}\b", str(error))
    data["occurrence_id"] = occurrence[0] if occurrence else None
    expected, observed = data["expected"], data["observed"]
    if data["error_code"] == "unapproved_prose_change" and type(expected) is str and type(observed) is str:
        data["text_differences"] = [
            {"operation": op, "before_span": [a, b], "after_span": [c, d],
             "before": expected[a:b], "after": observed[c:d]}
            for op, a, b, c, d in SequenceMatcher(None, expected, observed, autojunk=False).get_opcodes() if op != "equal"
        ]
    return safe_value(data, secret=secret, limit=limit)


def append_diagnostic(path: Path, diagnostic):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(diagnostic, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
