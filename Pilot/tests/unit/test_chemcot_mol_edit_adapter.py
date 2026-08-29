"""Tests for strict anonymous-ID joining of the ChemCoT molecule-editing inputs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Any

import pytest

from molhallulens.adapters import (
    ChemCoTMolEditAdapter,
    InputAdapterError,
    JoinedInputRecord,
)
from molhallulens.domain import DomainValidationError, Severity, ValidationStage


DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"
IDENTITY = ("mol_edit", "add_pilot_origin", "MolEdit/Add")


def _anonymous_id(number: int) -> str:
    return f"mol_edit.add_v2.{number:04d}"


def _raw(number: int, **updates: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "anonymous_sample_id": _anonymous_id(number),
        "task_family": IDENTITY[0],
        "subtask": IDENTITY[1],
        "reporting_task": IDENTITY[2],
        "orig_id": f"moledit_{number:05d}",
        "indexed_smiles": f"[C:{number}]",
        "instruction": "Attach a fragment.",
        "gt_smiles": "CN",
        "metadata": {"tags": [f"raw-{number}"]},
    }
    record.update(updates)
    return record


def _process(number: int, **updates: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "anonymous_sample_id": _anonymous_id(number),
        "task_family": IDENTITY[0],
        "subtask": IDENTITY[1],
        "reporting_task": IDENTITY[2],
        "orig_id": f"moledit_{number:05d}",
        "sample_id": number,
        "formal_cot_trace": [{"step_index": 1, "step_name": "TEST"}],
        "gt_smiles": "CN",
        "answer_smiles": "CN",
        "outcome": True,
        "parsed_reference_state": {"claim": number, "rdkit_claim": number},
        "verifier_checks": {"claim_ok": True, "all_pass": True},
    }
    record.update(updates)
    return record


def _template(n_samples: int = 1, **updates: Any) -> dict[str, Any]:
    template: dict[str, Any] = {
        "task_family": IDENTITY[0],
        "subtask": IDENTITY[1],
        "reporting_task": IDENTITY[2],
        "n_samples": n_samples,
        "step_fields": ["claim"],
        "rdkit_reference_fields": ["rdkit_claim"],
        "verifier_fields": ["claim_ok"],
    }
    template.update(updates)
    return template


def _join(
    raw_records: list[dict[str, Any]],
    process_records: list[dict[str, Any]],
    templates: list[dict[str, Any]],
) -> tuple[JoinedInputRecord, ...]:
    return ChemCoTMolEditAdapter.join_records(
        raw_records=raw_records,
        process_records=process_records,
        formal_templates=templates,
    )


def _codes(error: InputAdapterError) -> tuple[str, ...]:
    return tuple(issue.code for issue in error.report.issues)


def _issue_signature(error: InputAdapterError) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            issue.node_ids,
            issue.code,
            issue.evidence.get("field"),
            issue.evidence.get("source"),
            issue.evidence.get("identity"),
        )
        for issue in error.report.issues
    )


def test_join_is_id_based_sorted_namespaced_and_defensively_frozen() -> None:
    raw_records = [_raw(2), _raw(1)]
    process_records = [_process(1), _process(2)]
    template = _template(2)

    joined = _join(raw_records, process_records, [template])

    assert tuple(record.anonymous_sample_id for record in joined) == (
        _anonymous_id(1),
        _anonymous_id(2),
    )
    assert tuple(record.raw_record["orig_id"] for record in joined) == tuple(
        record.process_record["orig_id"] for record in joined
    )
    assert tuple(field.name for field in fields(JoinedInputRecord)) == (
        "anonymous_sample_id",
        "raw_record",
        "process_record",
        "formal_template",
    )
    raw_records[1]["metadata"]["tags"].append("mutated")
    process_records[0]["formal_cot_trace"].append({"step_index": 2})
    template["step_fields"].append("mutated")
    assert joined[0].raw_record["metadata"]["tags"] == ("raw-1",)
    assert len(joined[0].process_record["formal_cot_trace"]) == 1
    assert joined[0].formal_template["step_fields"] == ("claim",)
    with pytest.raises(TypeError):
        joined[0].raw_record["orig_id"] = "changed"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        joined[0].anonymous_sample_id = "changed"  # type: ignore[misc]


def test_joined_record_constructor_enforces_all_authoritative_shared_fields() -> None:
    with pytest.raises(DomainValidationError):
        JoinedInputRecord(
            anonymous_sample_id=_anonymous_id(1),
            raw_record=_raw(1),
            process_record=_process(1, gt_smiles="CC"),
            formal_template=_template(),
        )


@pytest.mark.parametrize("source", ("raw", "process"))
@pytest.mark.parametrize("divergent", (False, True))
def test_duplicate_ids_fail_closed_without_last_write_wins(
    source: str,
    divergent: bool,
) -> None:
    raws = [_raw(1)]
    processes = [_process(1)]
    duplicate = dict(raws[0] if source == "raw" else processes[0])
    if divergent:
        duplicate["orig_id"] = "different"
    (raws if source == "raw" else processes).append(duplicate)

    with pytest.raises(InputAdapterError) as captured:
        _join(raws, processes, [_template()])

    assert ("DUPLICATE_RAW_ID" if source == "raw" else "DUPLICATE_PROCESS_ID") in _codes(
        captured.value
    )
    duplicate_issue = next(
        issue for issue in captured.value.report.issues if issue.code.startswith("DUPLICATE_")
    )
    assert duplicate_issue.evidence["count"] == 2
    assert len(duplicate_issue.evidence["record_sha256s"]) == 2


@pytest.mark.parametrize(
    ("raws", "processes", "expected_code"),
    (
        ([_raw(1)], [_process(2)], "MISSING_PROCESS_RECORD"),
        ([_raw(2)], [_process(1)], "MISSING_RAW_RECORD"),
    ),
)
def test_missing_join_partners_are_structured(
    raws: list[dict[str, Any]],
    processes: list[dict[str, Any]],
    expected_code: str,
) -> None:
    with pytest.raises(InputAdapterError) as captured:
        _join(raws, processes, [_template()])

    assert expected_code in _codes(captured.value)
    assert all(
        issue.stage is ValidationStage.INPUT_RECORD
        and issue.severity is Severity.FATAL
        for issue in captured.value.report.issues
    )


@pytest.mark.parametrize("invalid_id", (None, "", "   ", 1))
def test_invalid_anonymous_ids_are_rejected(invalid_id: object) -> None:
    raw = _raw(1)
    raw["anonymous_sample_id"] = invalid_id

    with pytest.raises(InputAdapterError) as captured:
        _join([raw], [_process(1)], [_template()])

    assert "INVALID_ID" in _codes(captured.value)


def test_missing_required_field_is_distinct_from_conflict() -> None:
    raw = _raw(1)
    del raw["instruction"]

    with pytest.raises(InputAdapterError) as captured:
        _join([raw], [_process(1)], [_template()])

    assert "MISSING_REQUIRED_FIELD" in _codes(captured.value)
    assert "FIELD_CONFLICT" not in _codes(captured.value)


@pytest.mark.parametrize("field", ("task_family", "subtask", "reporting_task", "orig_id", "gt_smiles"))
def test_every_authoritative_shared_field_conflict_is_rejected_without_plaintext(
    field: str,
) -> None:
    process = _process(1)
    process[field] = "private-oracle-value" if field == "gt_smiles" else "different"

    with pytest.raises(InputAdapterError) as captured:
        _join([_raw(1)], [process], [_template()])

    issue = next(
        issue
        for issue in captured.value.report.issues
        if issue.code == "FIELD_CONFLICT" and issue.evidence["field"] == field
    )
    assert issue.node_ids == (_anonymous_id(1),)
    assert len(issue.evidence["left_sha256"]) == 64
    assert len(issue.evidence["right_sha256"]) == 64
    assert "private-oracle-value" not in str(issue.evidence)


def test_answer_smiles_drift_and_legacy_id_reuse_are_not_join_keys() -> None:
    raws = [_raw(1, orig_id="moledit_legacy"), _raw(2, orig_id="moledit_legacy")]
    processes = [
        _process(1, orig_id="moledit_legacy", sample_id=99, answer_smiles="C(N)"),
        _process(2, orig_id="moledit_legacy", sample_id=99, answer_smiles="N(C)"),
    ]

    joined = _join(raws, processes, [_template(2)])

    assert len(joined) == 2
    assert {record.process_record["sample_id"] for record in joined} == {99}
    assert all(
        record.process_record["answer_smiles"] != record.process_record["gt_smiles"]
        for record in joined
    )


def test_missing_duplicate_and_conflicting_templates_fail_closed() -> None:
    with pytest.raises(InputAdapterError) as missing:
        _join([_raw(1)], [_process(1)], [])
    assert "MISSING_TEMPLATE" in _codes(missing.value)

    template = _template()
    with pytest.raises(InputAdapterError) as duplicate:
        _join([_raw(1)], [_process(1)], [template, dict(template)])
    assert "DUPLICATE_TEMPLATE" in _codes(duplicate.value)

    with pytest.raises(InputAdapterError) as conflict:
        _join(
            [_raw(1)],
            [_process(1)],
            [_template(task_family="other_family")],
        )
    assert "MISSING_TEMPLATE" in _codes(conflict.value)


def test_template_count_and_inventory_must_match_sources_exactly() -> None:
    with pytest.raises(InputAdapterError) as count_error:
        _join([_raw(1)], [_process(1)], [_template(2)])
    assert "ORIGIN_COUNT_MISMATCH" in _codes(count_error.value)

    process = _process(1)
    process["parsed_reference_state"] = {
        "claim": 1,
        "unexpected": 1,
    }
    with pytest.raises(InputAdapterError) as inventory_error:
        _join([_raw(1)], [process], [_template()])
    issue = next(
        issue
        for issue in inventory_error.value.report.issues
        if issue.code == "TEMPLATE_INVENTORY_MISMATCH"
    )
    assert issue.evidence["missing_fields"] == ("rdkit_claim",)
    assert issue.evidence["unexpected_fields"] == ("unexpected",)


def test_error_order_does_not_depend_on_input_array_order() -> None:
    raws = [_raw(2), _raw(1), _raw(1)]
    processes = [_process(3), _process(2)]

    signatures = []
    for raw_order, process_order in (
        (raws, processes),
        (list(reversed(raws)), list(reversed(processes))),
    ):
        with pytest.raises(InputAdapterError) as captured:
            _join(raw_order, process_order, [_template(2)])
        signatures.append(_issue_signature(captured.value))

    assert signatures[0] == signatures[1]


def test_error_order_uses_complete_evidence_as_a_tie_breaker() -> None:
    reports = []
    malformed = [123, "bad"]
    for raw_order in (malformed, list(reversed(malformed))):
        with pytest.raises(InputAdapterError) as captured:
            ChemCoTMolEditAdapter.join_records(
                raw_records=raw_order,  # type: ignore[arg-type]
                process_records=[],
                formal_templates=[],
            )
        reports.append(captured.value.report.issues)

    assert reports[0] == reports[1]


def test_direct_join_turns_nonfinite_nested_values_into_structured_errors() -> None:
    raw = _raw(1, metadata={"score": float("nan")})

    with pytest.raises(InputAdapterError) as captured:
        _join([raw], [_process(1)], [_template()])

    assert "INVALID_RECORD_VALUE" in _codes(captured.value)


def _load_all_source_json() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_records = [
        record
        for path in sorted((DATASET_ROOT / "raw_benchmark_data/mol_edit").glob("*.json"))
        for record in json.loads(path.read_text())
    ]
    process_records = [
        record
        for path in sorted((DATASET_ROOT / "process_evaluation_data/mol_edit").glob("*.json"))
        for record in json.loads(path.read_text())
    ]
    templates = [
        json.loads(path.read_text())
        for path in sorted((DATASET_ROOT / "formal_templates/mol_edit").glob("*.json"))
    ]
    return raw_records, process_records, templates


def test_real_pilot_loads_150_unique_origins_and_preserves_known_legacy_anomalies() -> None:
    joined = ChemCoTMolEditAdapter().load(DATASET_ROOT)

    assert len(joined) == 150
    assert len({record.anonymous_sample_id for record in joined}) == 150
    assert tuple(record.anonymous_sample_id for record in joined) == tuple(
        sorted(record.anonymous_sample_id for record in joined)
    )
    assert Counter(record.pilot_subtask for record in joined) == {
        "add_pilot_origin": 50,
        "delete_pilot_origin": 50,
        "substitute_pilot_origin": 50,
    }
    assert len({record.raw_record["orig_id"] for record in joined}) == 126
    assert len({record.process_record["sample_id"] for record in joined}) == 125
    anomalous = next(
        record for record in joined if record.anonymous_sample_id == "mol_edit.add_v2.0295"
    )
    assert anomalous.process_record["sample_id"] == 299
    assert sum(
        record.process_record["answer_smiles"] != record.process_record["gt_smiles"]
        for record in joined
    ) == 7


def test_real_sources_join_identically_after_independent_reordering() -> None:
    raw_records, process_records, templates = _load_all_source_json()
    expected = ChemCoTMolEditAdapter().load(DATASET_ROOT)

    reordered = ChemCoTMolEditAdapter.join_records(
        raw_records=list(reversed(raw_records)),
        process_records=process_records[37:] + process_records[:37],
        formal_templates=list(reversed(templates)),
        expected_origin_count=150,
    )

    assert tuple(record.anonymous_sample_id for record in reordered) == tuple(
        record.anonymous_sample_id for record in expected
    )
    assert tuple(record.raw_record for record in reordered) == tuple(
        record.raw_record for record in expected
    )
    assert tuple(record.process_record for record in reordered) == tuple(
        record.process_record for record in expected
    )


@pytest.mark.parametrize("unsafe_path", ("../outside.json", "/tmp/outside.json"))
def test_manifest_paths_cannot_escape_dataset_root(tmp_path: Path, unsafe_path: str) -> None:
    (tmp_path / "active_benchmark_manifest.csv").write_text(
        "family,subtask,reporting_task,n_samples,raw_file,process_file\n"
        f"mol_edit,add_pilot_origin,MolEdit/Add,150,{unsafe_path},process.json\n"
    )

    with pytest.raises(InputAdapterError) as captured:
        ChemCoTMolEditAdapter().load(tmp_path)

    assert _codes(captured.value) == ("INVALID_SOURCE_PATH",)


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    (
        ('[{"anonymous_sample_id":"first","anonymous_sample_id":"second"}]', "DUPLICATE_JSON_KEY"),
        ("[{", "INVALID_JSON"),
        ("[NaN]", "INVALID_JSON_NUMBER"),
        ("[1e999]", "INVALID_JSON_NUMBER"),
    ),
)
def test_json_loader_rejects_duplicate_keys_and_malformed_json(
    tmp_path: Path,
    payload: str,
    expected_code: str,
) -> None:
    (tmp_path / "active_benchmark_manifest.csv").write_text(
        "family,subtask,reporting_task,n_samples,raw_file,process_file\n"
        "mol_edit,add_pilot_origin,MolEdit/Add,150,raw.json,process.json\n"
    )
    (tmp_path / "raw.json").write_text(payload)
    (tmp_path / "process.json").write_text("[]")

    with pytest.raises(InputAdapterError) as captured:
        ChemCoTMolEditAdapter().load(tmp_path)

    assert _codes(captured.value) == (expected_code,)


def test_template_symlinks_cannot_escape_dataset_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    template_dir = dataset_root / "formal_templates" / "mol_edit"
    template_dir.mkdir(parents=True)
    (dataset_root / "active_benchmark_manifest.csv").write_text(
        "family,subtask,reporting_task,n_samples,raw_file,process_file\n"
        "mol_edit,add_pilot_origin,MolEdit/Add,150,raw.json,process.json\n"
    )
    (dataset_root / "raw.json").write_text("[]")
    (dataset_root / "process.json").write_text("[]")
    outside = tmp_path / "outside-template.json"
    outside.write_text(json.dumps(_template(150)))
    (template_dir / "escaped.json").symlink_to(outside)

    with pytest.raises(InputAdapterError) as captured:
        ChemCoTMolEditAdapter().load(dataset_root)

    assert "INVALID_SOURCE_PATH" in _codes(captured.value)


def test_unrelated_valid_template_does_not_poison_molecule_editing_load(
    tmp_path: Path,
) -> None:
    class TinyAdapter(ChemCoTMolEditAdapter):
        EXPECTED_ORIGIN_COUNT = 1

    template_dir = tmp_path / "formal_templates" / "mol_edit"
    template_dir.mkdir(parents=True)
    (tmp_path / "active_benchmark_manifest.csv").write_text(
        "family,subtask,reporting_task,n_samples,raw_file,process_file\n"
        "mol_edit,add_pilot_origin,MolEdit/Add,1,raw.json,process.json\n"
    )
    (tmp_path / "raw.json").write_text(json.dumps([_raw(1)]))
    (tmp_path / "process.json").write_text(json.dumps([_process(1)]))
    (template_dir / "required.json").write_text(json.dumps(_template()))
    (template_dir / "future.json").write_text(
        json.dumps(
            _template(
                subtask="future_pilot_origin",
                reporting_task="MolEdit/Future",
            )
        )
    )

    joined = TinyAdapter().load(tmp_path)

    assert tuple(record.anonymous_sample_id for record in joined) == (_anonymous_id(1),)
