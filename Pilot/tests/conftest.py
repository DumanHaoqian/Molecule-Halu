from __future__ import annotations

from pathlib import Path
import re
import json

import pytest

from molhallulens.modules.error_planning import FragmentPool
from molhallulens.modules.ingestion import ChemCoTMolEditAdapter
from molhallulens.modules.reference import build_reference_dag


def preserve_enumerations(body, step):
    """Fake transport obeys the same preserved-breakdown contract as live Poe."""
    body += "".join("\n  " + item for item in fixture_enumerations(step))
    counts = {}
    def renumber(match):
        node = match[1]
        counts[node] = counts.get(node, 0) + 1
        return f"[[HALLU:{node}.{counts[node]:02d}]]"
    return re.sub(r"\[\[HALLU:([a-z0-9_]+)\.\d{2}\]\]", renumber, body)


def fixture_enumerations(step):
    if "preserved_enumerations" in step:
        return step["preserved_enumerations"]
    def render(parts):
        return "".join(p["text"] if "text" in p else f"[[HALLU:{p['claim_ref']}.01]]{p['planned_value']}[[/HALLU]]" for p in parts)
    return [", ".join(render(item) for item in block["items"]) + " = " + render(block["total"]) + "." for block in step.get("enumeration_blocks", ())]


def fixture_segments(body, step):
    """Explicit test-only migration from readable marked fixtures to wire segments.

    Production never accepts this legacy string channel. Wrong values are kept
    invalid rather than silently corrected by the fixture converter.
    """
    body = body.strip()
    prefix = f"Step {step['step_index']} [{step['step_name']}]: "
    if body.startswith(prefix):
        body = body[len(prefix):]
    body = body.split("\n  FORMAL:", 1)[0]
    for index, clause in enumerate(fixture_enumerations(step), 1):
        pattern = re.sub(r"\\\.\d{2}", lambda m: r"\.\d{2}", re.escape(clause))
        body = re.sub(pattern, lambda m: f"<<ENUM:{index:02d}>>", body, count=1)
    pattern = re.compile(r"\[\[HALLU:([a-z0-9_]+\.\d{2})\]\](.*?)\[\[/HALLU\]\]|<<ENUM:(\d{2})>>", re.S)
    claims = {c["node_id"]: c for c in step.get("affected_node_claims", ())}
    segments, cursor = [], 0
    for match in pattern.finditer(body):
        if match.start() > cursor:
            segments.append({"text": body[cursor:match.start()]})
        if match[3]:
            segments.append({"enumeration_ref": "enum_" + match[3]})
        elif step["rewrite_mode"] == "occurrence_patch":
            segments.append({"occurrence_ref": match[1]})
        else:
            node = match[1].rsplit(".", 1)[0]
            claim = claims.get(node, {})
            surfaces = claim.get("surfaces", {"canonical": claim.get("after_text")})
            surface = next((s for s, v in surfaces.items() if v == match[2]), None)
            segments.append({"claim_ref": node, "surface": surface} if surface else {"claim_ref": node, "value": match[2]})
        cursor = match.end()
    if cursor < len(body):
        segments.append({"text": body[cursor:]})
    return segments


def structured_fixture_transport(transport):
    """Adapt older readable fixtures explicitly; never used by application code."""
    def call(system, user, bot, temperature):
        prefix, encoded = user.split("\nINPUT:\n", 1)
        payload = json.loads(encoded)
        for step in payload["steps"]:
            step["preserved_enumerations"] = fixture_enumerations(step)
            # Legacy fixture input only. The real wire exposes body-only text.
            step["original_step_text"] = (
                f"Step {step['step_index']} [{step['step_name']}]: "
                + step["original_natural_body"] + "\n  FORMAL: " + step["modified_formal_ab"]
            )
        response = transport(system, prefix + "\nINPUT:\n" + json.dumps(payload), bot, temperature)
        data = json.loads(response)
        requested = {s["step_index"]: s for s in payload["steps"]}
        context = {s["step_index"] for s in payload["context_steps"]}
        rows = []
        for row in data["steps"]:
            index = row["step_index"]
            if index in context and index not in requested:
                continue  # Old fixture emits COPY too; COPY is now local.
            rows.append({"step_index": index, "segments": fixture_segments(row["rewritten_natural_language"], requested[index])})
        return json.dumps({"steps": rows})
    return call


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def references(project_root: Path):
    origins = ChemCoTMolEditAdapter().load(project_root / "Dataset")
    artifacts = tuple(build_reference_dag(origin) for origin in origins)
    return {artifact.normalized_subtask.value: artifact for artifact in artifacts}


@pytest.fixture(scope="session")
def all_references(project_root: Path):
    origins = ChemCoTMolEditAdapter().load(project_root / "Dataset")
    return tuple(build_reference_dag(origin) for origin in origins)


@pytest.fixture(scope="session")
def fragment_pool(all_references):
    return FragmentPool.from_reference_artifacts(all_references)
