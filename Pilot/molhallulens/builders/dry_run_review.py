"""T046 Codex-assisted chemistry review for the T045 dry-run release.

This review is intentionally independent of the T043 validator booleans.  It
re-reads the release-shaped artifacts, grounds every public instruction in the
raw benchmark row, recomputes molecular equivalence/descriptors with RDKit,
checks H/N pairs and propagation phenotypes, and audits char-to-token spans.

The artifact truthfully records that the reviewer is Codex-assisted and that no
external human reviewer participated.  No digest or SHA verification is
performed by this module.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from rdkit import Chem

from molhallulens.chemistry import (
    MoleculeParseError,
    compute_descriptors,
    isomeric_graph_equivalent,
)

T046_REVIEW_FORMAT_VERSION = "t046_codex_assisted_chemistry_review_v1"
T046_CHECKLIST_FORMAT_VERSION = "t046_human_review_checklist_v1"
T046_ISSUE_LOG_FORMAT_VERSION = "t046_human_review_issue_log_v1"
T046_ADJUDICATION_FORMAT_VERSION = "t046_human_review_adjudications_v1"

T046_REVIEWER_ID = "openai-codex-t046-reviewer"
T046_REVIEWER_DISPLAY_NAME = "OpenAI Codex (T046 chemistry review agent)"
T046_REVIEW_METHOD = "Codex-assisted structured chemistry review"
T046_EXPECTED_ORIGIN_COUNT = 15
T046_EXPECTED_RECORD_COUNT = 120
T046_EXPECTED_PAIR_COUNT = 60

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = DEFAULT_PROJECT_ROOT / "Dataset"
DEFAULT_DRY_RUN_ROOT = DEFAULT_PROJECT_ROOT / "HallucinationDataset/dry_run"
DEFAULT_REPORT_PATH = DEFAULT_DATASET_ROOT / "reports/t046_dry_run_human_review.json"

_SPLITS = ("train", "validation", "test")
_DOMAINS = (
    "root_truth",
    "candidate_chemistry",
    "propagation",
    "h_n_matching",
    "natural_formal_consistency",
    "token_spans",
)
_LABEL_LEAKAGE = re.compile(
    r"(?i)\b(?:hallucinated|hallucination|corrupted|incorrect variant|"
    r"reference answer)\b"
)


class DryRunReviewError(RuntimeError):
    """Fail-closed T046 review/publication error with a stable code."""

    def __init__(self, code: str, detail: str) -> None:
        if type(code) is not str or not code:
            raise ValueError("review error code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("review error detail must be non-empty text")
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class ReviewFinding:
    """One actionable review failure, always scoped to a complete H/N pair."""

    finding_id: str
    code: str
    domain: str
    pair_id: str
    affected_record_ids: tuple[str, str]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "code": self.code,
            "severity": "error",
            "domain": self.domain,
            "pair_id": self.pair_id,
            "affected_record_ids": self.affected_record_ids,
            "detail": self.detail,
            "disposition": "requires_complete_matched_pair_rebuild",
            "required_rebuild_unit": "one_h_one_n_complete_pair",
            "resolved": False,
        }


@dataclass(frozen=True, slots=True)
class PairReview:
    """Six-axis review result for one H/N pair."""

    pair_id: str
    origin_id: str
    subtask: str
    split: str
    policy: str
    record_ids: tuple[str, str]
    checks: Mapping[str, bool]

    def __post_init__(self) -> None:
        if set(self.checks) != set(_DOMAINS):
            raise ValueError("pair review must contain all six frozen domains")
        object.__setattr__(self, "checks", MappingProxyType(dict(self.checks)))

    @property
    def all_pass(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_unit_id": f"{self.pair_id}__t046_review",
            "review_unit": "complete_matched_pair",
            "pair_id": self.pair_id,
            "origin_id": self.origin_id,
            "subtask": self.subtask,
            "split": self.split,
            "policy": self.policy,
            "record_ids": self.record_ids,
            "checks": {
                domain: {
                    "status": "pass" if self.checks[domain] else "fail",
                    "reviewed": True,
                }
                for domain in _DOMAINS
            },
            "all_pass": self.all_pass,
            "adjudication": "accepted" if self.all_pass else "rebuild_pair",
        }


@dataclass(frozen=True, slots=True)
class DryRunHumanReview:
    """Complete T046 review result and publication payloads."""

    dry_run_id: str
    dataset_version: str
    reviewed_at: str
    pair_reviews: tuple[PairReview, ...]
    findings: tuple[ReviewFinding, ...]
    selected_origins: tuple[Mapping[str, Any], ...]

    @property
    def all_pass(self) -> bool:
        return not self.findings and all(row.all_pass for row in self.pair_reviews)

    @property
    def origin_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row.origin_id for row in self.pair_reviews}))

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(record_id for row in self.pair_reviews for record_id in row.record_ids)
        )

    def _reviewer(self) -> dict[str, Any]:
        return {
            "reviewer_id": T046_REVIEWER_ID,
            "display_name": T046_REVIEWER_DISPLAY_NAME,
            "review_method": T046_REVIEW_METHOD,
            "reviewed_at": self.reviewed_at,
            "external_human_reviewer_present": False,
            "disclosure": (
                "This is a Codex-assisted chemistry review; no external human "
                "chemist is claimed."
            ),
        }

    def checklist(self) -> dict[str, Any]:
        domain_counts = {
            domain: {
                "reviewed_pair_count": len(self.pair_reviews),
                "passed_pair_count": sum(row.checks[domain] for row in self.pair_reviews),
                "failed_pair_count": sum(
                    not row.checks[domain] for row in self.pair_reviews
                ),
            }
            for domain in _DOMAINS
        }
        return {
            "format_version": T046_CHECKLIST_FORMAT_VERSION,
            "dry_run_id": self.dry_run_id,
            "dataset_version": self.dataset_version,
            "reviewer": self._reviewer(),
            "scope": {
                "sampling_strategy": "census_all_dry_run_origins_and_pairs",
                "origin_count": len(self.origin_ids),
                "pair_count": len(self.pair_reviews),
                "record_count": len(self.record_ids),
                "domains": _DOMAINS,
            },
            "domain_counts": domain_counts,
            "selected_origins": [dict(row) for row in self.selected_origins],
            "pair_reviews": [
                {
                    **row.to_dict(),
                    "reviewer_id": T046_REVIEWER_ID,
                    "reviewed_at": self.reviewed_at,
                }
                for row in self.pair_reviews
            ],
            "all_pass": self.all_pass,
        }

    def _known_issues(self) -> list[dict[str, Any]]:
        partial_pairs = tuple(
            row.pair_id for row in self.pair_reviews if row.policy == "PARTIAL"
        )
        instruction_resolved = all(
            row.checks["root_truth"] and row.checks["natural_formal_consistency"]
            for row in self.pair_reviews
        )
        answer_resolved = all(
            row.checks["propagation"] for row in self.pair_reviews if row.policy == "PARTIAL"
        )
        return [
            {
                "issue_id": "T046-ISSUE-001",
                "code": "DETECTOR_INSTRUCTION_NOT_RAW_ORIGIN_INSTRUCTION",
                "classification": "systemic_grounding_error",
                "detected_by": T046_REVIEWER_ID,
                "affected_scope_at_detection": {
                    "origin_count": T046_EXPECTED_ORIGIN_COUNT,
                    "pair_count": T046_EXPECTED_PAIR_COUNT,
                    "record_count": T046_EXPECTED_RECORD_COUNT,
                },
                "root_cause": (
                    "T044/T045 serialization substituted a generic instruction for "
                    "the benchmark origin instruction."
                ),
                "adjudication": "fix_serializer_and_rebuild_entire_dry_run",
                "rebuild_unit": "all_15_complete_origin_bundles",
                "rebuilt_pair_ids": tuple(row.pair_id for row in self.pair_reviews),
                "rebuilt_record_count": len(self.record_ids),
                "resolution_verification": (
                    "Every public instruction now equals both the raw benchmark "
                    "instruction and the typed state instruction."
                ),
                "status": "resolved" if instruction_resolved else "unresolved",
            },
            {
                "issue_id": "T046-ISSUE-002",
                "code": "PARTIAL_ANSWER_CORRECT_USED_CANDIDATE_PRODUCT_RELATION",
                "classification": "systemic_label_semantics_error",
                "detected_by": T046_REVIEWER_ID,
                "affected_scope_at_detection": {
                    "origin_count": T046_EXPECTED_ORIGIN_COUNT,
                    "pair_count": len(partial_pairs),
                    "record_count": len(partial_pairs) * 2,
                    "affected_h_label_count": len(partial_pairs),
                },
                "root_cause": (
                    "PARTIAL answer_correct used Answer-to-candidate-product "
                    "equivalence instead of Answer-to-hidden-GT equivalence."
                ),
                "adjudication": "fix_oracle_semantics_and_rebuild_every_partial_pair",
                "rebuild_unit": "complete_h_n_pair",
                "rebuilt_pair_ids": partial_pairs,
                "rebuilt_record_count": len(partial_pairs) * 2,
                "resolution_verification": (
                    "answer_correct is independently recomputed from final Answer "
                    "versus hidden GT for every H and N record."
                ),
                "status": "resolved" if answer_resolved else "unresolved",
            },
        ]

    def issue_log(self) -> dict[str, Any]:
        known = self._known_issues()
        unresolved = [finding.to_dict() for finding in self.findings]
        return {
            "format_version": T046_ISSUE_LOG_FORMAT_VERSION,
            "dry_run_id": self.dry_run_id,
            "reviewer": self._reviewer(),
            "resolved_systemic_issues": known,
            "new_unresolved_findings": unresolved,
            "resolved_systemic_issue_count": sum(
                item["status"] == "resolved" for item in known
            ),
            "new_unresolved_finding_count": len(unresolved),
            "all_resolved": all(item["status"] == "resolved" for item in known)
            and not unresolved,
        }

    def adjudications(self) -> dict[str, Any]:
        known = self._known_issues()
        return {
            "format_version": T046_ADJUDICATION_FORMAT_VERSION,
            "dry_run_id": self.dry_run_id,
            "reviewer": self._reviewer(),
            "policy": (
                "Any chemistry, label, rendering, or span issue invalidates both "
                "members of the matched pair; pair-local patching is forbidden."
            ),
            "decisions": [
                {
                    "issue_id": item["issue_id"],
                    "decision": item["adjudication"],
                    "status": item["status"],
                    "rebuild_unit": item["rebuild_unit"],
                    "rebuilt_pair_ids": item["rebuilt_pair_ids"],
                    "verification": item["resolution_verification"],
                }
                for item in known
            ],
            "unresolved_decisions": [
                finding.to_dict() for finding in self.findings
            ],
            "all_adjudications_closed": self.all_pass
            and all(item["status"] == "resolved" for item in known),
        }

    def report(self) -> dict[str, Any]:
        issues = self.issue_log()
        checklist = self.checklist()
        return {
            "format_version": T046_REVIEW_FORMAT_VERSION,
            "dry_run_id": self.dry_run_id,
            "dataset_version": self.dataset_version,
            "reviewer": self._reviewer(),
            "all_pass": self.all_pass and issues["all_resolved"],
            "summary": {
                "origin_count": len(self.origin_ids),
                "pair_count": len(self.pair_reviews),
                "record_count": len(self.record_ids),
                "reviewed_subtasks": dict(
                    sorted(Counter(row.subtask for row in self.pair_reviews).items())
                ),
                "reviewed_policies": dict(
                    sorted(Counter(row.policy for row in self.pair_reviews).items())
                ),
                "passed_pair_count": sum(row.all_pass for row in self.pair_reviews),
                "failed_pair_count": sum(not row.all_pass for row in self.pair_reviews),
                "systemic_label_error_count_after_rebuild": 0 if self.all_pass else None,
                "resolved_systemic_issue_count": issues[
                    "resolved_systemic_issue_count"
                ],
                "unresolved_finding_count": len(self.findings),
            },
            "acceptance": {
                "no_systemic_label_errors": self.all_pass,
                "all_discovered_issues_adjudicated": issues["all_resolved"],
                "affected_complete_pairs_rebuilt": issues["all_resolved"],
                "reviewer_and_time_traceable": True,
                "all_15_origins_reviewed": len(self.origin_ids)
                == T046_EXPECTED_ORIGIN_COUNT,
                "all_60_pairs_reviewed": len(self.pair_reviews)
                == T046_EXPECTED_PAIR_COUNT,
                "all_120_records_reviewed": len(self.record_ids)
                == T046_EXPECTED_RECORD_COUNT,
                "digest_or_sha_verification_performed": False,
            },
            "domain_counts": checklist["domain_counts"],
            "limitations": [
                "The review is Codex-assisted; no external human chemist participated.",
                "T045 uses an offline tokenizer fixture rather than production ChemDFM-R weights.",
                "No digest or SHA verification was performed.",
            ],
        }

    def artifact_payloads(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "human_review_checklist.json": _render_json(self.checklist()),
                "human_review_issue_log.json": _render_json(self.issue_log()),
                "human_review_adjudications.json": _render_json(
                    self.adjudications()
                ),
            }
        )


def _render_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DryRunReviewError(
            "T046_ARTIFACT_READ_FAILED", f"cannot read {path.name}"
        ) from error
    if not isinstance(value, dict):
        raise DryRunReviewError(
            "T046_ARTIFACT_SHAPE", f"{path.name} must contain a JSON object"
        )
    return value


def _read_jsonl_family(root: Path, family: str) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for split in _SPLITS:
        path = root / family / f"{split}.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise DryRunReviewError(
                "T046_ARTIFACT_READ_FAILED", f"cannot read {family}/{split}.jsonl"
            ) from error
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise DryRunReviewError(
                    "T046_ARTIFACT_JSONL", f"invalid JSON in {family}/{split}.jsonl"
                ) from error
            if not isinstance(value, dict) or type(value.get("record_id")) is not str:
                raise DryRunReviewError(
                    "T046_ARTIFACT_SHAPE",
                    f"invalid record object in {family}/{split}.jsonl",
                )
            record_id = value["record_id"]
            if record_id in values or value.get("split") != split:
                raise DryRunReviewError(
                    "T046_ARTIFACT_IDENTITY",
                    f"duplicate or split-mismatched record {record_id}",
                )
            values[record_id] = value
    return values


def _raw_origins(dataset_root: Path) -> dict[str, dict[str, Any]]:
    root = dataset_root / "raw_benchmark_data/mol_edit"
    values: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DryRunReviewError(
                "T046_RAW_ORIGIN_READ_FAILED", f"cannot read {path.name}"
            ) from error
        if not isinstance(rows, list):
            raise DryRunReviewError(
                "T046_RAW_ORIGIN_SHAPE", f"{path.name} must contain a JSON array"
            )
        for row in rows:
            if not isinstance(row, dict) or type(row.get("anonymous_sample_id")) is not str:
                raise DryRunReviewError(
                    "T046_RAW_ORIGIN_SHAPE", f"invalid origin in {path.name}"
                )
            origin_id = row["anonymous_sample_id"]
            if origin_id in values:
                raise DryRunReviewError(
                    "T046_RAW_ORIGIN_DUPLICATE", f"duplicate raw origin {origin_id}"
                )
            values[origin_id] = row
    return values


def _node(state: Mapping[str, Any], node_id: str) -> Any:
    return state["nodes"][node_id]["normalized_value"]


def _equivalent(left: Any, right: Any) -> bool:
    if type(left) is not str or type(right) is not str:
        return False
    try:
        return isomeric_graph_equivalent(left, right)
    except (MoleculeParseError, RuntimeError, TypeError, ValueError):
        return False


def _descriptors(smiles: Any) -> Any | None:
    if type(smiles) is not str:
        return None
    try:
        return compute_descriptors(smiles)
    except (MoleculeParseError, RuntimeError, TypeError, ValueError):
        return None


def _anchor_matches(source: Any, anchor_idx: Any, anchor_element: Any) -> bool:
    if (
        type(source) is not str
        or type(anchor_idx) is not int
        or type(anchor_element) is not str
    ):
        return False
    try:
        molecule = Chem.MolFromSmiles(source, sanitize=True)
    except (RuntimeError, TypeError, ValueError):
        return False
    if molecule is None:
        return False
    matches = tuple(
        atom for atom in molecule.GetAtoms() if atom.GetAtomMapNum() == anchor_idx
    )
    return len(matches) == 1 and matches[0].GetSymbol() == anchor_element


def _overlap(left: Sequence[int], right: Sequence[int]) -> bool:
    return max(int(left[0]), int(right[0])) < min(int(left[1]), int(right[1]))


def _literal_matches(value: Any, literal: str) -> bool:
    if literal == str(value):
        return True
    if type(value) in {int, float}:
        return literal == f"+{value}"
    return False


class _PairAudit:
    def __init__(self, pair_id: str, h_id: str, n_id: str) -> None:
        self.pair_id = pair_id
        self.record_ids = (h_id, n_id)
        self.failed: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def expect(self, condition: bool, domain: str, code: str, detail: str) -> None:
        if domain not in _DOMAINS:
            raise ValueError("unknown T046 review domain")
        if not condition:
            self.failed[domain].append((code, detail))

    def findings(self, start_index: int) -> tuple[ReviewFinding, ...]:
        rows: list[ReviewFinding] = []
        index = start_index
        for domain in _DOMAINS:
            for code, detail in self.failed.get(domain, ()):
                rows.append(
                    ReviewFinding(
                        finding_id=f"T046-FINDING-{index:04d}",
                        code=code,
                        domain=domain,
                        pair_id=self.pair_id,
                        affected_record_ids=self.record_ids,
                        detail=detail,
                    )
                )
                index += 1
        return tuple(rows)


def _review_pair(
    pair: Sequence[Mapping[str, Any]],
    *,
    oracles: Mapping[str, Mapping[str, Any]],
    states: Mapping[str, Mapping[str, Any]],
    tokens: Mapping[str, Mapping[str, Any]],
    raw_origins: Mapping[str, Mapping[str, Any]],
    finding_start: int,
) -> tuple[PairReview, tuple[ReviewFinding, ...]]:
    by_label = {row["variant"]["label"]: row for row in pair}
    if set(by_label) != {"H", "N"}:
        raise DryRunReviewError(
            "T046_PAIR_SHAPE", "each pair must contain exactly one H and one N"
        )
    h = by_label["H"]
    n = by_label["N"]
    pair_id = h["pair_id"]
    audit = _PairAudit(pair_id, h["record_id"], n["record_id"])
    origin_id = h["origin_id"]
    raw = raw_origins.get(origin_id)
    if raw is None:
        raise DryRunReviewError("T046_RAW_ORIGIN_MISSING", f"missing {origin_id}")

    member_data = []
    for record in (h, n):
        record_id = record["record_id"]
        oracle = oracles[record_id]
        state = states[record_id]
        token = tokens[record_id]
        reference = state["reference"]
        locked = state["locked"]
        member_data.append((record, oracle, state, token, reference, locked))

        public = record["detector_input"]
        audit.expect(
            public["instruction"] == raw.get("instruction")
            and _node(reference, "instruction") == raw.get("instruction")
            and _node(locked, "instruction") == raw.get("instruction"),
            "root_truth",
            "T046_RAW_INSTRUCTION_MISMATCH",
            "public/reference/locked instruction must equal the raw origin instruction",
        )
        audit.expect(
            public["indexed_smiles"] == raw.get("indexed_smiles")
            and _node(reference, "source") == raw.get("indexed_smiles")
            and _node(locked, "source") == raw.get("indexed_smiles"),
            "root_truth",
            "T046_RAW_SOURCE_MISMATCH",
            "public/reference/locked source must equal the raw indexed SMILES",
        )
        gt = oracle["gt_smiles"]
        audit.expect(
            _equivalent(gt, raw.get("gt_smiles"))
            and _equivalent(_node(reference, "product"), gt)
            and _equivalent(_node(reference, "final_answer"), gt),
            "root_truth",
            "T046_REFERENCE_GT_MISMATCH",
            "reference product and final answer must be molecularly equivalent to raw GT",
        )
        audit.expect(
            _anchor_matches(
                _node(reference, "source"),
                _node(reference, "anchor_idx"),
                _node(reference, "anchor_element"),
            ),
            "root_truth",
            "T046_REFERENCE_ANCHOR_NOT_IN_SOURCE",
            "reference anchor map index and element must resolve in the source graph",
        )

        source_descriptors = _descriptors(_node(locked, "source"))
        product = _node(locked, "product")
        answer = _node(locked, "final_answer")
        product_descriptors = _descriptors(product)
        answer_descriptors = _descriptors(answer)
        audit.expect(
            source_descriptors is not None
            and product_descriptors is not None
            and answer_descriptors is not None,
            "candidate_chemistry",
            "T046_CANDIDATE_SANITIZE_FAILED",
            "source, candidate product, and final answer must strictly sanitize",
        )
        if source_descriptors is not None and product_descriptors is not None:
            subtask = record["subtask"]
            directional_edit = (
                product_descriptors.heavy_atom_count
                > source_descriptors.heavy_atom_count
                if subtask == "add"
                else product_descriptors.heavy_atom_count
                < source_descriptors.heavy_atom_count
                if subtask == "delete"
                else not _equivalent(_node(locked, "source"), product)
            )
            audit.expect(
                directional_edit,
                "candidate_chemistry",
                "T046_CANDIDATE_EDIT_DIRECTION_MISMATCH",
                "candidate product must realize the editing subtask rather than a no-op",
            )

        locked_nodes = locked["nodes"]
        fragment_checks = (
            ("add_fragment", "fragment_heavy"),
            ("add_fragment", "add_heavy"),
            ("remove_group", "remove_heavy"),
            ("remove_group_step1", "remove_heavy"),
        )
        for fragment_id, count_id in fragment_checks:
            if fragment_id not in locked_nodes or count_id not in locked_nodes:
                continue
            fragment = _node(locked, fragment_id)
            if fragment in {None, "none"}:
                continue
            fragment_descriptors = _descriptors(fragment)
            audit.expect(
                fragment_descriptors is not None
                and fragment_descriptors.heavy_atom_count == _node(locked, count_id),
                "candidate_chemistry",
                "T046_FRAGMENT_COUNT_MISMATCH",
                f"{fragment_id} must sanitize and agree with {count_id}",
            )

        expected_answer_correct = _equivalent(answer, gt)
        expected_constraint_satisfied = _equivalent(product, gt)
        labels = record["trace_labels"]
        audit.expect(
            labels["answer_correct"] is expected_answer_correct,
            "propagation",
            "T046_ANSWER_CORRECT_NOT_GT_GROUNDED",
            "answer_correct must equal final-answer versus hidden-GT graph equivalence",
        )
        audit.expect(
            labels["constraint_satisfied"] is expected_constraint_satisfied,
            "propagation",
            "T046_CONSTRAINT_LABEL_NOT_PRODUCT_GROUNDED",
            "constraint_satisfied must equal candidate-product versus GT equivalence",
        )
        audit.expect(
            labels["chemically_valid"]
            is (product_descriptors is not None and answer_descriptors is not None),
            "candidate_chemistry",
            "T046_CHEMICAL_VALIDITY_LABEL_MISMATCH",
            "chemically_valid must agree with strict product and answer parsing",
        )

        serialized = record["serialized"]["text"]
        segments = {item["field_name"]: item for item in record["serialized"]["segments"]}
        reasoning_segment = segments["reasoning_chain"]
        audit.expect(
            serialized[reasoning_segment["start"] : reasoning_segment["end"]]
            == public["reasoning_chain"],
            "natural_formal_consistency",
            "T046_REASONING_SERIALIZATION_MISMATCH",
            "detector reasoning must exactly occupy its serialized segment",
        )
        audit.expect(
            not _LABEL_LEAKAGE.search(public["reasoning_chain"])
            and not _LABEL_LEAKAGE.search(public["final_answer"]),
            "natural_formal_consistency",
            "T046_RENDERER_LABEL_LEAKAGE",
            "reasoning and final answer must remain label-blind",
        )
        audit.expect(
            all(
                public["reasoning_chain"].count(f"FORMAL: {step['formal_ab']}") == 1
                for step in state["formal_trace"]
            )
            and public["final_answer"] == answer,
            "natural_formal_consistency",
            "T046_NATURAL_FORMAL_STATE_MISMATCH",
            "each FORMAL line and final answer must be the exact locked projection",
        )

        arrays = (
            "input_ids",
            "attention_mask",
            "offset_mapping",
            "segment_ids",
            "evaluation_mask",
            "hallucination_core_mask",
            "error_any_mask",
            "local_falsehood_mask",
            "off_task_branch_mask",
            "reasoning_mask",
            "answer_mask",
            "boundary_ambiguous_mask",
            "error_char_fraction",
        )
        token_count = len(token["input_ids"])
        audit.expect(
            token["activation_alignment"] == "post_token_h_t"
            and all(len(token[key]) == token_count for key in arrays),
            "token_spans",
            "T046_TOKEN_ARRAY_ALIGNMENT",
            "token arrays must be equal length and post-token aligned",
        )
        nested_masks = (
            *token["semantic_type_masks"].values(),
            *token["edit_subtype_masks"].values(),
            *token["causal_role_masks"].values(),
        )
        audit.expect(
            all(len(mask) == token_count for mask in nested_masks)
            and all(
                not bit or bool(token["error_any_mask"][index])
                for mask in nested_masks
                for index, bit in enumerate(mask)
            ),
            "token_spans",
            "T046_TOKEN_AXIS_OUTSIDE_ERROR_MASK",
            "multi-axis masks must be equal length and subsets of error_any",
        )
        ignored_segments = (segments["indexed_smiles"], segments["instruction"])
        audit.expect(
            all(
                not token["evaluation_mask"][index]
                for index, offset in enumerate(token["offset_mapping"])
                if offset != [0, 0]
                and any(
                    _overlap(offset, (segment["start"], segment["end"]))
                    for segment in ignored_segments
                )
            ),
            "token_spans",
            "T046_PROMPT_PREFIX_NOT_IGNORED",
            "source and instruction tokens must be excluded from evaluation",
        )

    h_record, h_oracle, h_state, h_token, h_reference, h_locked = member_data[0]
    n_record, n_oracle, n_state, n_token, _n_reference, n_locked = member_data[1]
    policy = h_record["variant"]["propagation"]
    events = h_oracle["graph_delta"]
    differences = {
        (item["target_kind"], item["target_id"])
        for item in h_state["semantic_difference_targets"]
    }
    event_targets = {(item["target_kind"], item["target_id"]) for item in events}
    roots = [item for item in events if item["causal_role"] in {"ROOT", "TERMINAL"}]

    audit.expect(
        len(roots) == 1
        and roots[0]["target_id"] == h_record["mutation"]["root_state_id"]
        and roots[0]["before"]
        == _node(h_reference, roots[0]["target_id"])
        and roots[0]["after"] == _node(h_locked, roots[0]["target_id"])
        and roots[0]["before"] != roots[0]["after"],
        "root_truth",
        "T046_ROOT_EVENT_NOT_REFERENCE_BOUND",
        "H must contain one real root bound to reference-before and locked-after",
    )
    audit.expect(
        differences == event_targets and not n_oracle["graph_delta"]
        and not n_state["semantic_difference_targets"],
        "propagation",
        "T046_DELTA_STATE_MISMATCH",
        "H events must equal state differences and N must have no mutation",
    )

    gt = h_oracle["gt_smiles"]
    h_product = _node(h_locked, "product")
    h_answer = _node(h_locked, "final_answer")
    if policy == "LOCAL":
        phenotype_ok = (
            len(events) == 1
            and roots[0]["causal_role"] == "ROOT"
            and _equivalent(h_product, gt)
            and _equivalent(h_answer, gt)
            and not h_record["trace_labels"]["reasoning_valid"]
        )
    elif policy == "PARTIAL":
        phenotype_ok = (
            len(events) > 1
            and roots[0]["causal_role"] == "ROOT"
            and all(
                item["causal_role"].startswith("PROPAGATED") for item in events[1:]
            )
            and not _equivalent(h_product, gt)
            and not h_record["trace_labels"]["reasoning_valid"]
        )
    elif policy == "FULL_CF":
        phenotype_ok = (
            len(events) > 1
            and roots[0]["causal_role"] == "ROOT"
            and all(
                item["causal_role"].startswith("PROPAGATED") for item in events[1:]
            )
            and not _equivalent(h_product, gt)
            and _equivalent(h_answer, h_product)
            and not h_record["trace_labels"]["reasoning_valid"]
        )
    elif policy == "TERMINAL":
        phenotype_ok = (
            len(events) == 1
            and roots[0]["target_id"] == "final_answer"
            and roots[0]["causal_role"] == "TERMINAL"
            and _equivalent(h_product, gt)
            and not _equivalent(h_answer, gt)
            and h_record["trace_labels"]["reasoning_valid"]
        )
    else:
        phenotype_ok = False
    audit.expect(
        phenotype_ok,
        "propagation",
        "T046_PROPAGATION_PHENOTYPE",
        f"{policy} molecular/causal phenotype does not match its frozen contract",
    )

    product_descriptors = _descriptors(h_product)
    source_descriptors = _descriptors(_node(h_locked, "source"))
    if policy in {"FULL_CF", "TERMINAL"} and product_descriptors is not None:
        audit.expect(
            _node(h_locked, "product_heavy") == product_descriptors.heavy_atom_count
            and _node(h_locked, "product_rings") == product_descriptors.ring_count
            and _node(h_locked, "heavy_delta")
            == product_descriptors.heavy_atom_count - source_descriptors.heavy_atom_count
            and _node(h_locked, "ring_delta")
            == product_descriptors.ring_count - source_descriptors.ring_count,
            "candidate_chemistry",
            "T046_FULL_CANDIDATE_DESCRIPTOR_MISMATCH",
            "FULL_CF/TERMINAL product counts and deltas must be recomputed chemistry",
        )
    if policy == "LOCAL" and product_descriptors is not None:
        audit.expect(
            _node(h_locked, "product_rings") == product_descriptors.ring_count
            and _node(h_locked, "ring_delta")
            == product_descriptors.ring_count - source_descriptors.ring_count,
            "candidate_chemistry",
            "T046_LOCAL_UNMUTATED_DESCRIPTOR_MISMATCH",
            "unmutated LOCAL ring claims must remain chemically correct",
        )

    audit.expect(
        all(
            h_record[key] == n_record[key]
            for key in (
                "origin_id",
                "bundle_id",
                "pair_id",
                "split",
                "subtask",
                "leakage_group_id",
            )
        )
        and h_record["variant"]["matched_record_id"] == n_record["record_id"]
        and n_record["variant"]["matched_record_id"] == h_record["record_id"]
        and h_record["variant"]["propagation"]
        == n_record["variant"]["propagation"]
        and h_record["mutation"] == n_record["mutation"]
        and h_record["detector_input"]["indexed_smiles"]
        == n_record["detector_input"]["indexed_smiles"]
        and h_record["detector_input"]["instruction"]
        == n_record["detector_input"]["instruction"],
        "h_n_matching",
        "T046_PAIR_AXIS_MISMATCH",
        "H/N identities, matching axes, source, and instruction must be reciprocal",
    )
    audit.expect(
        all(
            _equivalent(_node(n_locked, node_id), gt)
            for node_id in ("product", "final_answer")
        )
        and not n_record["spans"]
        and all(
            value is expected
            for value, expected in (
                (n_record["trace_labels"]["hallucination_present"], False),
                (n_record["trace_labels"]["reasoning_valid"], True),
                (n_record["trace_labels"]["answer_correct"], True),
                (n_record["trace_labels"]["chemically_valid"], True),
                (n_record["trace_labels"]["constraint_satisfied"], True),
                (n_record["trace_labels"]["format_valid"], True),
                (n_record["trace_labels"]["answer_complete"], True),
            )
        ),
        "h_n_matching",
        "T046_N_CONTROL_NOT_FAITHFUL",
        "N must be GT-faithful with no positive span and all faithful trace flags",
    )

    spans = h_record["spans"]
    events_by_target = {(item["target_kind"], item["target_id"]): item for item in events}
    serialized = h_record["serialized"]["text"]
    audit.expect(
        len(spans) == len(events)
        and all(
            ("node", span["state_or_edge_id"]) in events_by_target
            and _literal_matches(
                events_by_target[("node", span["state_or_edge_id"])]["after"],
                serialized[span["literal_span"][0] : span["literal_span"][1]],
            )
            for span in spans
        ),
        "natural_formal_consistency",
        "T046_EVENT_SPAN_LITERAL_MISMATCH",
        "each H event must have one exact rendered literal span",
    )

    offsets = h_token["offset_mapping"]
    error_mask = h_token["error_any_mask"]
    audit.expect(
        bool(spans)
        and any(error_mask)
        and all(
            any(
                h_token["evaluation_mask"][index]
                and error_mask[index]
                and _overlap(offset, span["literal_span"])
                for index, offset in enumerate(offsets)
                if offset != [0, 0]
            )
            for span in spans
        ),
        "token_spans",
        "T046_H_SPAN_NOT_PROJECTED",
        "every H literal span must overlap an evaluated positive token",
    )
    n_masks = (
        n_token["hallucination_core_mask"],
        n_token["error_any_mask"],
        n_token["local_falsehood_mask"],
        n_token["off_task_branch_mask"],
        *n_token["semantic_type_masks"].values(),
        *n_token["edit_subtype_masks"].values(),
        *n_token["causal_role_masks"].values(),
    )
    target_span = n_token["matched_target_span"]
    audit.expect(
        all(not any(mask) for mask in n_masks)
        and target_span is not None
        and any(
            n_token["evaluation_mask"][index]
            and _overlap(offset, target_span)
            for index, offset in enumerate(n_token["offset_mapping"])
            if offset != [0, 0]
        ),
        "token_spans",
        "T046_N_MASK_OR_MATCHED_TARGET_INVALID",
        "N masks must be zero and its matched target must cover an evaluated token",
    )

    checks = {domain: not audit.failed.get(domain) for domain in _DOMAINS}
    review = PairReview(
        pair_id=pair_id,
        origin_id=origin_id,
        subtask=h["subtask"],
        split=h["split"],
        policy=policy,
        record_ids=(h["record_id"], n["record_id"]),
        checks=checks,
    )
    return review, audit.findings(finding_start)


def review_t045_dry_run(
    *,
    dry_run_root: Path = DEFAULT_DRY_RUN_ROOT,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    reviewed_at: str,
) -> DryRunHumanReview:
    """Review all 15 origins, 60 H/N pairs, and 120 T045 records."""

    if type(reviewed_at) is not str or not reviewed_at:
        raise ValueError("reviewed_at must be non-empty ISO-8601 text")
    try:
        datetime.fromisoformat(reviewed_at)
    except ValueError as error:
        raise ValueError("reviewed_at must be valid ISO-8601 text") from error

    manifest = _read_json(dry_run_root / "dataset_manifest.json")
    selection = _read_json(dry_run_root / "selection_manifest.json")
    records = _read_jsonl_family(dry_run_root, "records")
    oracles = _read_jsonl_family(dry_run_root, "oracle")
    states = _read_jsonl_family(dry_run_root, "state_graphs")
    tokens = _read_jsonl_family(dry_run_root, "tokenized/chemdfm_r")
    families = (records, oracles, states, tokens)
    if any(set(family) != set(records) for family in families[1:]):
        raise DryRunReviewError(
            "T046_FAMILY_JOIN_MISMATCH", "release families have different record IDs"
        )
    if len(records) != T046_EXPECTED_RECORD_COUNT:
        raise DryRunReviewError(
            "T046_RECORD_COUNT", "T046 requires exactly 120 dry-run records"
        )

    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records.values():
        pairs[record["pair_id"]].append(record)
    if len(pairs) != T046_EXPECTED_PAIR_COUNT or any(
        len(pair) != 2 for pair in pairs.values()
    ):
        raise DryRunReviewError(
            "T046_PAIR_COUNT", "T046 requires exactly sixty complete H/N pairs"
        )

    raw = _raw_origins(dataset_root)
    reviews: list[PairReview] = []
    findings: list[ReviewFinding] = []
    for pair_id in sorted(pairs):
        review, discovered = _review_pair(
            pairs[pair_id],
            oracles=oracles,
            states=states,
            tokens=tokens,
            raw_origins=raw,
            finding_start=len(findings) + 1,
        )
        reviews.append(review)
        findings.extend(discovered)

    origin_ids = {review.origin_id for review in reviews}
    state_by_origin = {
        state["origin_id"]: state
        for state in states.values()
        if state["record_id"].endswith("__LOCAL__N")
    }
    selected_rows: list[dict[str, Any]] = []
    for row in selection["selected"]:
        origin_id = row["origin_id"]
        reference = state_by_origin[origin_id]["reference"]
        nodes = reference["nodes"]

        def optional(node_id: str, source_nodes: Mapping[str, Any] = nodes) -> Any:
            node = source_nodes.get(node_id)
            return None if node is None else node["normalized_value"]

        selected_rows.append(
            {
                "origin_id": origin_id,
                "subtask": row["subtask"],
                "split": row["split"],
                "instruction": optional("instruction"),
                "reference_edit_claims": {
                    "anchor_idx": optional("anchor_idx"),
                    "anchor_element": optional("anchor_element"),
                    "remove_group": optional("remove_group")
                    or optional("remove_group_step1"),
                    "add_fragment": optional("add_fragment"),
                    "source_heavy": optional("source_heavy"),
                    "product_heavy": optional("product_heavy"),
                    "heavy_delta": optional("heavy_delta"),
                },
                "review_status": "pass"
                if all(
                    review.all_pass
                    for review in reviews
                    if review.origin_id == origin_id
                )
                else "fail",
            }
        )
    selected = tuple(selected_rows)
    if (
        len(origin_ids) != T046_EXPECTED_ORIGIN_COUNT
        or {row["origin_id"] for row in selected} != origin_ids
    ):
        raise DryRunReviewError(
            "T046_ORIGIN_COUNT", "selection manifest must bind all fifteen origins"
        )

    return DryRunHumanReview(
        dry_run_id=manifest["dry_run_id"],
        dataset_version=manifest["dataset_version"],
        reviewed_at=reviewed_at,
        pair_reviews=tuple(reviews),
        findings=tuple(findings),
        selected_origins=selected,
    )


def _publish(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    temporary.replace(path)


def write_t046_review_artifacts(
    review: DryRunHumanReview,
    *,
    dry_run_root: Path = DEFAULT_DRY_RUN_ROOT,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> DryRunHumanReview:
    """Publish the closed checklist, issue log, adjudications, and summary."""

    if type(review) is not DryRunHumanReview:
        raise TypeError("review must be DryRunHumanReview")
    if not review.all_pass or not review.issue_log()["all_resolved"]:
        raise DryRunReviewError(
            "T046_UNRESOLVED_FINDINGS",
            "review artifacts cannot be accepted before complete pair rebuilds pass",
        )
    for name, payload in review.artifact_payloads().items():
        _publish(dry_run_root / "reports" / name, payload)
    _publish(report_path, _render_json(review.report()))
    return review


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_DRY_RUN_ROOT",
    "DEFAULT_REPORT_PATH",
    "T046_ADJUDICATION_FORMAT_VERSION",
    "T046_CHECKLIST_FORMAT_VERSION",
    "T046_ISSUE_LOG_FORMAT_VERSION",
    "T046_REVIEW_FORMAT_VERSION",
    "DryRunHumanReview",
    "DryRunReviewError",
    "PairReview",
    "ReviewFinding",
    "review_t045_dry_run",
    "write_t046_review_artifacts",
]
