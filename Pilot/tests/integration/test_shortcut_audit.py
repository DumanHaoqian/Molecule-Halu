from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

import molhallulens.modules.release.shortcut_audit as shortcut


@pytest.fixture(scope="module")
def audit_rows() -> tuple[shortcut.AuditRow, ...]:
    return shortcut.load_t047_audit_rows()


@pytest.fixture(scope="module")
def audit_report() -> dict[str, object]:
    return shortcut.run_t047_shortcut_audit()


def test_audit_joins_complete_pair_balanced_release(
    audit_rows: tuple[shortcut.AuditRow, ...],
) -> None:
    assert len(audit_rows) == 120
    assert len({row.origin_id for row in audit_rows}) == 15
    assert Counter(row.label for row in audit_rows) == Counter({0: 60, 1: 60})
    assert Counter(row.record["split"] for row in audit_rows) == Counter(
        {"train": 72, "validation": 24, "test": 24}
    )

    pairs: dict[str, list[shortcut.AuditRow]] = defaultdict(list)
    for row in audit_rows:
        pairs[row.pair_id].append(row)
    assert len(pairs) == 60
    assert all(len(pair) == 2 for pair in pairs.values())
    assert all({row.label for row in pair} == {0, 1} for pair in pairs.values())
    assert all(len({row.origin_id for row in pair}) == 1 for pair in pairs.values())


def test_metadata_attack_uses_only_pair_matched_non_outcome_fields(
    audit_rows: tuple[shortcut.AuditRow, ...],
) -> None:
    by_pair: dict[str, list[shortcut.AuditRow]] = defaultdict(list)
    for row in audit_rows:
        by_pair[row.pair_id].append(row)
    for pair in by_pair.values():
        h_row = next(row for row in pair if row.label)
        n_row = next(row for row in pair if not row.label)
        assert shortcut._metadata_terms(h_row) == shortcut._metadata_terms(n_row)
        terms = " ".join(shortcut._metadata_terms(h_row)).lower()
        assert "label=" not in terms
        assert "hallucination" not in terms
        assert h_row.record_id not in terms
        assert h_row.origin_id not in terms


def test_all_learned_attacks_are_grouped_out_of_fold_and_finite(
    audit_report: dict[str, object],
) -> None:
    protocol = audit_report["audit_protocol"]
    assert protocol == {
        "fold_count": 5,
        "group_unit": "leakage_group_id",
        "held_out_split": "test",
        "id": shortcut.T047_AUDIT_PROTOCOL,
        "mandatory_gate_splits": ["train", "validation"],
        "molecule_comparison": "RDKit canonical isomeric graph equivalence",
        "out_of_fold_predictions_only": True,
        "test_used_for_model_or_threshold_selection": False,
    }
    baselines = audit_report["baselines"]
    for name in (
        "metadata_only_logistic",
        "span_only_char_tfidf_logistic",
        "reasoning_only_word_tfidf_logistic",
        "nearest_neighbor_retrieval_k5",
        "smiles_validity",
        "visible_reasoning_answer_graph_comparator",
        "hidden_oracle_answer_graph_comparator",
    ):
        metric = baselines[name]
        assert 0.0 <= metric["auroc"] <= 1.0
        assert metric["positive_count"] == metric["negative_count"] == 48

    development = audit_report["development_inventory"]
    assert development == {
        "h_count": 48,
        "leakage_group_count": 12,
        "n_count": 48,
        "origin_count": 12,
        "pair_count": 48,
        "record_count": 96,
        "splits": ["train", "validation"],
    }


def test_mandatory_gates_apply_the_frozen_strict_comparators(
    audit_report: dict[str, object],
) -> None:
    gates = audit_report["mandatory_gates"]
    expected = {
        "metadata_auroc": ("<=", shortcut.T047_METADATA_AUROC_LIMIT),
        "span_only_tfidf_auroc": ("<=", shortcut.T047_SPAN_TFIDF_AUROC_LIMIT),
        "reasoning_only_shallow_auroc": (
            "<=",
            shortcut.T047_REASONING_AUROC_LIMIT,
        ),
        "token_length_standardized_difference": (
            "<",
            shortcut.T047_TOKEN_LENGTH_SMD_LIMIT,
        ),
        "style_pair_matching": ("==", 0),
    }
    for name, (comparator, threshold) in expected.items():
        gate = gates[name]
        assert gate["comparator"] == comparator
        assert gate["threshold"] == threshold
        if comparator == "<=":
            assert gate["passed"] is (gate["actual"] <= threshold)
        elif comparator == "<":
            assert gate["passed"] is (gate["actual"] < threshold)
        else:
            assert gate["passed"] is (gate["actual"] == threshold)
    assert audit_report["all_pass"] is all(
        gate["passed"] for gate in gates.values()
    )
    assert audit_report["all_pass"] is True
    assert audit_report["threshold_failure_count"] == sum(
        not gate["passed"] for gate in gates.values()
    )


def test_length_and_style_matching_are_measured_on_real_h_n_pairs(
    audit_report: dict[str, object],
) -> None:
    matching = audit_report["matching"]
    length = matching["length"]
    for name in (
        "full_token_count",
        "evaluated_token_count",
        "reasoning_token_count",
        "answer_token_count",
        "serialized_character_count",
    ):
        assert set(length[name]) == {
            "absolute_standardized_difference",
            "h_mean",
            "n_mean",
            "standardized_difference",
        }
        assert length[name]["absolute_standardized_difference"] >= 0.0

    style = matching["style"]
    assert style["all_pairs_exact_on_frozen_style_fields"] is (
        style["mismatch_count"] == 0
    )
    assert set(style["inventory"]) == set(shortcut._PAIR_STYLE_FIELDS)
    assert style["renderer_diversity_count"] == len(
        style["inventory"]["renderer_id"]
    )


def test_symbolic_baselines_parse_visible_claims_and_keep_oracle_private(
    audit_report: dict[str, object],
) -> None:
    baselines = audit_report["baselines"]
    validity = baselines["smiles_validity"]
    assert validity["score_definition"].startswith("one_if_visible")
    visible = baselines["visible_reasoning_answer_graph_comparator"]
    assert "isomeric_graph_equivalent" in visible["score_definition"]
    hidden = baselines["hidden_oracle_answer_graph_comparator"]
    assert hidden["detector_visible"] is False
    assert "hidden_gt" in hidden["score_definition"]
    assert set(baselines["slices"]) == {
        "FULL_CF",
        "LOCAL",
        "PARTIAL",
        "TERMINAL",
    }


def test_test_split_is_held_out_from_every_mandatory_attack(
    audit_report: dict[str, object],
) -> None:
    assert audit_report["report_scope"] == "final_with_heldout_test_diagnostic"
    heldout = audit_report["heldout_test_diagnostics"]
    assert heldout["scope"] == "held_out_test_read_once_after_development_design_freeze"
    assert heldout["used_for_candidate_layer_or_threshold_selection"] is False
    assert heldout["training_splits"] == ["train", "validation"]
    assert heldout["evaluation_split"] == "test"
    assert heldout["record_count"] == 24
    assert heldout["origin_count"] == 3
    for name in (
        "metadata_only_logistic",
        "span_only_char_tfidf_logistic",
        "reasoning_only_word_tfidf_logistic",
        "nearest_neighbor_retrieval_k5",
        "smiles_validity",
        "visible_reasoning_answer_graph_comparator",
        "hidden_oracle_answer_graph_comparator",
    ):
        metric = heldout["baselines"][name]
        assert metric["positive_count"] == metric["negative_count"] == 12
        assert 0.0 <= metric["auroc"] <= 1.0


def test_passed_release_needs_no_remediation(
    audit_report: dict[str, object],
) -> None:
    remediation = audit_report["remediation"]
    assert remediation == {
        "required": False,
        "status": "not_needed",
        "strict_rerun_command": (
            "python -m molhallulens.modules.release.shortcut_audit --strict"
        ),
    }


def test_a_failed_gate_would_emit_executable_fail_closed_remediation(
    audit_rows: tuple[shortcut.AuditRow, ...],
) -> None:
    development_rows = tuple(
        row for row in audit_rows if row.record["split"] != "test"
    )
    failure = {
        "actual": 0.61,
        "comparator": "<=",
        "metric": "span_only_tfidf_auroc",
        "passed": False,
        "threshold": 0.55,
    }
    remediation = shortcut._remediation(
        development_rows,
        (failure,),
        {"h_associated": [], "n_associated": []},
    )
    assert remediation["required"] is True
    assert remediation["status"] == "requires_t045_surface_design_rebuild"
    assert remediation["failing_metrics"] == [failure]
    assert remediation["priority_pairs"]
    assert any(
        action["action"] == "regenerate_complete_matched_pairs"
        for action in remediation["actions"]
    )
    rerun = next(
        action
        for action in remediation["actions"]
        if action["action"] == "rebuild_then_strict_rerun"
    )
    assert "--strict" in rerun["command"]
    assert "exits zero only after every mandatory threshold passes" in rerun[
        "success_condition"
    ]
    assert any("do not relax" in item for item in remediation["safety_constraints"])


def test_report_is_deterministic_and_matches_committed_artifact(
    audit_report: dict[str, object],
) -> None:
    report_path = shortcut.DEFAULT_REPORT_PATH
    assert report_path.is_file()
    committed = json.loads(report_path.read_text(encoding="utf-8"))
    assert committed == audit_report
    assert shortcut.render_t047_report(committed) == report_path.read_text(
        encoding="utf-8"
    )


def test_writer_and_strict_exit_status_follow_the_same_report(
    tmp_path: Path,
    audit_report: dict[str, object],
) -> None:
    report_path = tmp_path / "t047.json"
    written = shortcut.write_t047_shortcut_report(report_path=report_path)
    assert written == audit_report
    assert json.loads(report_path.read_text(encoding="utf-8")) == audit_report

    cli_report = tmp_path / "t047-cli.json"
    remediation = tmp_path / "remediation.json"
    status = shortcut.main(
        (
            "--report",
            str(cli_report),
            "--remediation-manifest",
            str(remediation),
            "--strict",
        )
    )
    assert status == (0 if audit_report["all_pass"] else 2)
    assert json.loads(cli_report.read_text(encoding="utf-8")) == audit_report
    assert json.loads(remediation.read_text(encoding="utf-8")) == audit_report[
        "remediation"
    ]
