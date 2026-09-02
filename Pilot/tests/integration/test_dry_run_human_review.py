from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from molhallulens.modules.release.dry_run_review import (
    T046_EXPECTED_ORIGIN_COUNT,
    T046_EXPECTED_PAIR_COUNT,
    T046_EXPECTED_RECORD_COUNT,
    DryRunHumanReview,
    DryRunReviewError,
    review_t045_dry_run,
    write_t046_review_artifacts,
)

REVIEWED_AT = "2026-08-30T04:40:00+08:00"


@pytest.fixture(scope="module")
def review() -> DryRunHumanReview:
    return review_t045_dry_run(reviewed_at=REVIEWED_AT)


def _mutate_jsonl_record(
    root: Path,
    family: str,
    predicate: object,
    mutation: object,
) -> str:
    for path in sorted((root / family).glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if predicate(row):
                mutation(row)
                path.write_text(
                    "".join(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                        for item in rows
                    ),
                    encoding="utf-8",
                )
                return row["record_id"]
    raise AssertionError("record mutation target was not found")


def test_census_review_passes_all_six_domains(review: DryRunHumanReview) -> None:
    assert review.all_pass
    assert len(review.origin_ids) == T046_EXPECTED_ORIGIN_COUNT == 15
    assert len(review.pair_reviews) == T046_EXPECTED_PAIR_COUNT == 60
    assert len(review.record_ids) == T046_EXPECTED_RECORD_COUNT == 120
    assert not review.findings
    assert Counter(row.subtask for row in review.pair_reviews) == Counter(
        {"add": 20, "delete": 20, "substitute": 20}
    )
    assert Counter(row.policy for row in review.pair_reviews) == Counter(
        {"LOCAL": 15, "PARTIAL": 15, "FULL_CF": 15, "TERMINAL": 15}
    )
    assert all(row.all_pass and all(row.checks.values()) for row in review.pair_reviews)


def test_review_discloses_reviewer_and_closes_pair_rebuilds(
    review: DryRunHumanReview,
) -> None:
    checklist = review.checklist()
    issue_log = review.issue_log()
    adjudications = review.adjudications()
    reviewer = checklist["reviewer"]
    assert reviewer["reviewed_at"] == REVIEWED_AT
    assert reviewer["external_human_reviewer_present"] is False
    assert "Codex-assisted" in reviewer["disclosure"]
    assert checklist["scope"] == {
        "sampling_strategy": "census_all_dry_run_origins_and_pairs",
        "origin_count": 15,
        "pair_count": 60,
        "record_count": 120,
        "domains": (
            "root_truth",
            "candidate_chemistry",
            "propagation",
            "h_n_matching",
            "natural_formal_consistency",
            "token_spans",
        ),
    }
    assert issue_log["resolved_systemic_issue_count"] == 2
    assert issue_log["new_unresolved_finding_count"] == 0
    assert issue_log["all_resolved"] is True
    by_id = {
        item["issue_id"]: item for item in issue_log["resolved_systemic_issues"]
    }
    assert by_id["T046-ISSUE-001"]["rebuilt_record_count"] == 120
    assert len(by_id["T046-ISSUE-001"]["rebuilt_pair_ids"]) == 60
    assert by_id["T046-ISSUE-002"]["rebuilt_record_count"] == 30
    assert len(by_id["T046-ISSUE-002"]["rebuilt_pair_ids"]) == 15
    assert adjudications["all_adjudications_closed"] is True
    assert all(
        decision["status"] == "resolved"
        for decision in adjudications["decisions"]
    )


def test_generic_instruction_and_non_oracle_answer_label_fail_closed(
    tmp_path: Path,
) -> None:
    source = Path("HallucinationDataset/dry_run")
    root = tmp_path / "dry_run"
    shutil.copytree(source, root)

    _mutate_jsonl_record(
        root,
        "records",
        lambda row: row["variant"]["label"] == "H"
        and row["variant"]["propagation"] == "PARTIAL",
        lambda row: (
            row["detector_input"].__setitem__(
                "instruction", "Apply the requested molecular edit."
            ),
            row["trace_labels"].__setitem__(
                "answer_correct", not row["trace_labels"]["answer_correct"]
            ),
        ),
    )
    failed = review_t045_dry_run(
        dry_run_root=root,
        reviewed_at=REVIEWED_AT,
    )
    codes = {finding.code for finding in failed.findings}
    assert "T046_RAW_INSTRUCTION_MISMATCH" in codes
    assert "T046_ANSWER_CORRECT_NOT_GT_GROUNDED" in codes
    assert not failed.all_pass
    assert all(
        finding.to_dict()["disposition"]
        == "requires_complete_matched_pair_rebuild"
        for finding in failed.findings
    )
    with pytest.raises(DryRunReviewError) as caught:
        write_t046_review_artifacts(
            failed,
            dry_run_root=tmp_path / "published",
            report_path=tmp_path / "report.json",
        )
    assert caught.value.code == "T046_UNRESOLVED_FINDINGS"


def test_shifted_char_span_invalidates_its_complete_pair(tmp_path: Path) -> None:
    source = Path("HallucinationDataset/dry_run")
    root = tmp_path / "dry_run"
    shutil.copytree(source, root)

    record_id = _mutate_jsonl_record(
        root,
        "records",
        lambda row: row["variant"]["label"] == "H"
        and row["variant"]["propagation"] == "LOCAL",
        lambda row: row["spans"][0].__setitem__(
            "literal_span",
            [
                row["spans"][0]["literal_span"][0] + 1,
                row["spans"][0]["literal_span"][1] + 1,
            ],
        ),
    )
    failed = review_t045_dry_run(
        dry_run_root=root,
        reviewed_at=REVIEWED_AT,
    )
    relevant = [
        finding for finding in failed.findings if record_id in finding.affected_record_ids
    ]
    assert relevant
    assert {finding.code for finding in relevant} & {
        "T046_EVENT_SPAN_LITERAL_MISMATCH",
        "T046_H_SPAN_NOT_PROJECTED",
    }
    assert all(
        finding.to_dict()["required_rebuild_unit"]
        == "one_h_one_n_complete_pair"
        for finding in relevant
    )


def test_writer_publishes_traceable_closed_review(
    tmp_path: Path,
    review: DryRunHumanReview,
) -> None:
    root = tmp_path / "HallucinationDataset/dry_run"
    report_path = tmp_path / "Dataset/reports/t046_dry_run_human_review.json"
    assert (
        write_t046_review_artifacts(
            review,
            dry_run_root=root,
            report_path=report_path,
        )
        is review
    )
    expected = {
        "human_review_checklist.json",
        "human_review_issue_log.json",
        "human_review_adjudications.json",
    }
    assert {path.name for path in (root / "reports").iterdir()} == expected
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["all_pass"] is True
    assert report["acceptance"] == {
        "no_systemic_label_errors": True,
        "all_discovered_issues_adjudicated": True,
        "affected_complete_pairs_rebuilt": True,
        "reviewer_and_time_traceable": True,
        "all_15_origins_reviewed": True,
        "all_60_pairs_reviewed": True,
        "all_120_records_reviewed": True,
        "digest_or_sha_verification_performed": False,
    }
