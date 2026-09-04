from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from molhallulens.config import DEFAULT_HALLUCINATION_CONFIG
from molhallulens.generate_dataset import generate_dataset
from molhallulens.modules.error_injection import UnifiedHallucinationInjector
from molhallulens.modules.error_planning import UnifiedHallucinationPlanner
from molhallulens.modules.text_realization import (
    AffectedNodeClaim,
    FORMAL_MARKER,
    PoeRewriteRequest,
    PoeStepRewriteInput,
    PoeStepTextAgent,
    StepRewriteMode,
    build_poe_rewrite_request,
)
from molhallulens.modules.text_realization.poe_agent import POE_RENDERER_VERSION


def _marked_body(step: dict) -> str:
    prefix = f"Step {step['step_index']} [{step['step_name']}]: "
    if step["rewrite_mode"] == "copy":
        return step["original_step_text"].split(FORMAL_MARKER, 1)[0][len(prefix) :]
    if step["rewrite_mode"] == "derivation_rewrite":
        claims = "; ".join(
            f"{claim['node_id']}="
            f"[[HALLU:{claim['node_id']}.01]]{claim['after_text']}[[/HALLU]]"
            for claim in step["affected_node_claims"]
        )
        return f"Updated claims: {claims}."
    body = step["original_step_text"].split(FORMAL_MARKER, 1)[0][len(prefix) :]
    for occurrence in sorted(
        step["required_hallucination_occurrences"],
        key=lambda item: item["original_span"][0],
        reverse=True,
    ):
        start, end = occurrence["original_span"]
        marker = (
            f"[[HALLU:{occurrence['occurrence_id']}]]"
            f"{occurrence['after_text']}[[/HALLU]]"
        )
        body = body[:start] + marker + body[end:]
    return body


def _valid_response(user_prompt: str) -> str:
    payload = json.loads(user_prompt.split("\nINPUT:\n", 1)[1].split(
        "\nPREVIOUS_RESPONSE_REJECTED:",
        1,
    )[0])
    return json.dumps(
        {
            "steps": [
                {
                    "step_index": step["step_index"],
                    "rewritten_natural_language": _marked_body(step),
                }
                for step in payload["steps"]
            ]
        }
    )


def test_batch_records_one_failure_and_keeps_later_incremental_pairs(
    project_root,
    tmp_path,
):
    failed_origin = "mol_edit.add_v2.0012"

    def fake_transport(system_prompt, user_prompt, bot_name, temperature):
        del system_prompt, bot_name, temperature
        payload = json.loads(user_prompt.split("\nINPUT:\n", 1)[1])
        if payload["origin_id"] == failed_origin:
            raise RuntimeError("injected transport failure")
        return _valid_response(user_prompt)

    agent = PoeStepTextAgent(
        DEFAULT_HALLUCINATION_CONFIG,
        transport=fake_transport,
        environment={},
        cache_directory=tmp_path / "cache",
    )
    output = tmp_path / "batch.jsonl"
    manifest = tmp_path / "batch.failures.jsonl"
    summary = generate_dataset(
        dataset_root=project_root / "Dataset",
        output_path=output,
        failure_manifest_path=manifest,
        max_origins=3,
        poe_agent=agent,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    failures = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert summary.origin_count == 3
    assert summary.successful_origin_count == 2
    assert summary.failed_origin_count == 1
    assert summary.successful_variant_count == 2
    assert summary.failed_variant_count == 1
    assert summary.record_count == 4
    assert summary.poe_network_request_count == 3
    assert {row["origin_id"] for row in rows} == {
        "mol_edit.add_v2.0003",
        "mol_edit.add_v2.0013",
    }
    assert [row["variant_label"] for row in rows] == ["H", "N", "H", "N"]
    assert failures == list(summary.failures)
    assert failures[0]["origin_id"] == failed_origin
    assert failures[0]["stage"] == "E_TEXT_REALIZATION"
    assert failures[0]["error_type"] == "RuntimeError"


def test_interruption_preserves_already_flushed_pairs(project_root, tmp_path):
    call_count = 0

    def interrupting_transport(system_prompt, user_prompt, bot_name, temperature):
        nonlocal call_count
        del system_prompt, bot_name, temperature
        call_count += 1
        if call_count == 2:
            raise KeyboardInterrupt
        return _valid_response(user_prompt)

    agent = PoeStepTextAgent(
        DEFAULT_HALLUCINATION_CONFIG,
        transport=interrupting_transport,
        environment={},
        cache_directory=tmp_path / "cache",
    )
    output = tmp_path / "interrupted.jsonl"
    with pytest.raises(KeyboardInterrupt):
        generate_dataset(
            dataset_root=project_root / "Dataset",
            output_path=output,
            max_origins=3,
            poe_agent=agent,
        )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert [row["variant_label"] for row in rows] == ["H", "N"]
    assert {row["origin_id"] for row in rows} == {"mol_edit.add_v2.0003"}


def test_stale_cache_metadata_and_response_are_cache_misses(
    references,
    fragment_pool,
    tmp_path,
):
    reference = references["add"]
    plan = UnifiedHallucinationPlanner(fragment_pool).plan(reference, variant_index=0)
    injected = UnifiedHallucinationInjector().apply(reference.state_dag, plan)
    request = build_poe_rewrite_request(reference, injected)
    first = PoeStepTextAgent(
        DEFAULT_HALLUCINATION_CONFIG,
        transport=lambda system, user, bot, temperature: _valid_response(user),
        environment={},
        cache_directory=tmp_path,
    ).rewrite(request)
    cache_path = (
        tmp_path
        / DEFAULT_HALLUCINATION_CONFIG.poe_bot_name
        / f"{first.prompt_sha256}.json"
    )
    payload = json.loads(cache_path.read_text())
    payload["renderer_version"] = "stale-renderer"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    calls = []

    def fresh_transport(system_prompt, user_prompt, bot_name, temperature):
        del system_prompt, bot_name, temperature
        calls.append(user_prompt)
        return _valid_response(user_prompt)

    second = PoeStepTextAgent(
        DEFAULT_HALLUCINATION_CONFIG,
        transport=fresh_transport,
        environment={},
        cache_directory=tmp_path,
    ).rewrite(request)
    assert not second.cache_hit
    assert second.network_request_count == 1
    assert len(calls) == 1

    payload = json.loads(cache_path.read_text())
    assert payload["renderer_version"] == POE_RENDERER_VERSION
    payload["response_text"] = "{}"
    payload["response_sha256"] = hashlib.sha256(b"{}").hexdigest()
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    third = PoeStepTextAgent(
        DEFAULT_HALLUCINATION_CONFIG,
        transport=fresh_transport,
        environment={},
        cache_directory=tmp_path,
    ).rewrite(request)
    assert not third.cache_hit
    assert third.network_request_count == 1
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("invalid_body", "expected_code"),
    (
        (
            "It contains 1 sulfur, 2 oxygens, 3 carbons, and 3 fluorines, "
            "totaling [[HALLU:fragment_heavy.01]]10[[/HALLU]] heavy atoms.",
            "false_enumeration",
        ),
        (
            "The displayed check is 5 + 4 = "
            "[[HALLU:fragment_heavy.01]]10[[/HALLU]].",
            "false_arithmetic",
        ),
    ),
)
def test_retry_telemetry_classifies_enumeration_and_arithmetic_rejections(
    invalid_body,
    expected_code,
    tmp_path,
):
    step = PoeStepRewriteInput(
        step_index=1,
        step_name="FRAGMENT_IDENTIFICATION",
        original_step_text=(
            "Step 1 [FRAGMENT_IDENTIFICATION]: The fragment has 9 heavy atoms."
            "\n  FORMAL: FRAGMENT(heavy_atoms=9)"
        ),
        modified_formal_ab="FRAGMENT(heavy_atoms=10)",
        required_hallucination_occurrences=(),
        rewrite_mode=StepRewriteMode.DERIVATION_REWRITE,
        affected_node_claims=(
            AffectedNodeClaim("fragment_heavy", "9", "10"),
        ),
    )
    request = PoeRewriteRequest(
        origin_id=f"retry-{expected_code}",
        subtask="add",
        indexed_smiles="[CH4:1]",
        instruction="Use the synthetic count.",
        steps=(step,),
    )
    responses = [
        json.dumps(
            {
                "steps": [
                    {
                        "step_index": 1,
                        "rewritten_natural_language": invalid_body,
                    }
                ]
            }
        ),
        json.dumps(
            {
                "steps": [
                    {
                        "step_index": 1,
                        "rewritten_natural_language": (
                            "The claimed total is "
                            "[[HALLU:fragment_heavy.01]]10[[/HALLU]]."
                        ),
                    }
                ]
            }
        ),
    ]

    def retrying_transport(system_prompt, user_prompt, bot_name, temperature):
        del system_prompt, user_prompt, bot_name, temperature
        return responses.pop(0)

    config = replace(DEFAULT_HALLUCINATION_CONFIG, poe_max_attempts=2)
    agent = PoeStepTextAgent(
        config,
        transport=retrying_transport,
        environment={},
        cache_directory=tmp_path / expected_code,
    )
    result = agent.rewrite(request)
    telemetry = agent.telemetry()
    assert result.network_request_count == 2
    assert result.validation_rejection_codes == (expected_code,)
    assert telemetry["retry_count"] == 1
    assert telemetry["requests_with_retry"] == 1
    assert telemetry["validation_rejection_counts"] == {expected_code: 1}
