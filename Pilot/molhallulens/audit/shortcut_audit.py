"""Deterministic shortcut audit for the T045 dry-run release.

The audit deliberately trains only small, transparent attack models.  Every
learned score is out-of-fold with leakage groups kept intact.  Chemistry
baselines parse detector-visible SMILES with RDKit; string equality is never
used as a molecular comparator.

This module does not rewrite a failed dataset.  Instead, it emits the exact
matched pairs and surface dimensions that need regeneration, and exposes a
strict CLI that can be rerun after the T045 construction design is repaired.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

T047_REPORT_FORMAT_VERSION = "t047_shortcut_audit_v2"
T047_AUDIT_PROTOCOL = "development_grouped_5_fold_oof_heldout_test_v2"
T047_FOLD_COUNT = 5
T047_METADATA_AUROC_LIMIT = 0.55
T047_SPAN_TFIDF_AUROC_LIMIT = 0.55
T047_REASONING_AUROC_LIMIT = 0.60
T047_TOKEN_LENGTH_SMD_LIMIT = 0.10

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRY_RUN_ROOT = DEFAULT_PROJECT_ROOT / "HallucinationDataset/dry_run"
DEFAULT_REPORT_PATH = DEFAULT_PROJECT_ROOT / "Dataset/reports/t047_shortcut_audit.json"

_SPLITS = ("train", "validation", "test")
_PAIR_STYLE_FIELDS = (
    "propagation",
    "candidate_source",
    "operator_family",
    "operator_id",
    "renderer_id",
    "root_state_id",
)
_PRODUCT_LINE = re.compile(
    r"(?m)^Step\s+\d+\s+\[PRODUCT_CONSTRUCTION\]:\s*(\S+)\s*$"
)
_WORD_TOKEN = re.compile(r"[A-Za-z]+|\d+|[^\w\s]")


class ShortcutAuditError(RuntimeError):
    """Fail-closed error for incomplete or inconsistent audit inputs."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("shortcut audit error code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("shortcut audit error detail must be non-empty text")
        self.code = code
        self.detail = detail
        self.evidence = dict(evidence or {})
        super().__init__(f"{code}: {detail}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "evidence": _stable(self.evidence),
        }


@dataclass(frozen=True)
class AuditRow:
    """One joined detector record, token artifact, and hidden oracle row."""

    record: Mapping[str, Any]
    tokenized: Mapping[str, Any]
    oracle: Mapping[str, Any]

    @property
    def record_id(self) -> str:
        return str(self.record["record_id"])

    @property
    def origin_id(self) -> str:
        return str(self.record["origin_id"])

    @property
    def leakage_group_id(self) -> str:
        return str(self.record["leakage_group_id"])

    @property
    def pair_id(self) -> str:
        return str(self.record["pair_id"])

    @property
    def label(self) -> int:
        return int(self.record["variant"]["label"] == "H")


@dataclass(frozen=True)
class _LinearAttack:
    scores: tuple[float, ...]
    fold_by_group: Mapping[str, int]
    vocabulary_sizes: tuple[int, ...]


def _stable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_stable(item) for item in sorted(value, key=str)]
    if isinstance(value, float):
        if not math.isfinite(value):
            if math.isnan(value):
                return "nan"
            return "positive_infinity" if value > 0 else "negative_infinity"
        # BLAS/libm implementations can differ in the final machine-precision
        # bit.  Reports are release artifacts, so quantize diagnostics at a
        # precision far below any acceptance threshold before serialization.
        rounded = round(value, 12)
        return 0.0 if rounded == 0.0 else rounded
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise ShortcutAuditError(
            "SHORTCUT_ARTIFACT_MISSING",
            "required dry-run artifact is missing",
            evidence={"path": str(path)},
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ShortcutAuditError(
                "SHORTCUT_JSONL_INVALID",
                "dry-run JSONL contains an invalid row",
                evidence={"path": str(path), "line_number": line_number},
            ) from exc
        if not isinstance(value, dict):
            raise ShortcutAuditError(
                "SHORTCUT_JSONL_ROW_TYPE",
                "dry-run JSONL rows must be objects",
                evidence={"path": str(path), "line_number": line_number},
            )
        rows.append(value)
    return tuple(rows)


def _indexed(rows: Iterable[Mapping[str, Any]], artifact: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        record_id = row.get("record_id")
        if type(record_id) is not str or not record_id:
            raise ShortcutAuditError(
                "SHORTCUT_RECORD_ID_INVALID",
                f"{artifact} row has no valid record_id",
            )
        if record_id in result:
            raise ShortcutAuditError(
                "SHORTCUT_RECORD_ID_DUPLICATE",
                f"{artifact} contains a duplicate record_id",
                evidence={"record_id": record_id},
            )
        result[record_id] = row
    return result


def load_t047_audit_rows(
    dry_run_root: Path | None = None,
    *,
    splits: Sequence[str] | None = None,
) -> tuple[AuditRow, ...]:
    """Join and validate the T045 detector/token/oracle artifacts by record ID."""

    root = DEFAULT_DRY_RUN_ROOT if dry_run_root is None else Path(dry_run_root)
    selected_splits = _SPLITS if splits is None else tuple(splits)
    if (
        not selected_splits
        or len(set(selected_splits)) != len(selected_splits)
        or any(split not in _SPLITS for split in selected_splits)
    ):
        raise ValueError("splits must be a non-empty unique subset of train/validation/test")
    records: list[dict[str, Any]] = []
    tokenized: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    for split in selected_splits:
        records.extend(_read_jsonl(root / "records" / f"{split}.jsonl"))
        tokenized.extend(
            _read_jsonl(root / "tokenized/chemdfm_r" / f"{split}.jsonl")
        )
        oracle.extend(_read_jsonl(root / "oracle" / f"{split}.jsonl"))

    record_by_id = _indexed(records, "records")
    token_by_id = _indexed(tokenized, "tokenized")
    oracle_by_id = _indexed(oracle, "oracle")
    id_sets = (set(record_by_id), set(token_by_id), set(oracle_by_id))
    if not (id_sets[0] == id_sets[1] == id_sets[2]):
        raise ShortcutAuditError(
            "SHORTCUT_ARTIFACT_JOIN_MISMATCH",
            "record, token, and oracle inventories must match exactly",
            evidence={
                "record_count": len(id_sets[0]),
                "token_count": len(id_sets[1]),
                "oracle_count": len(id_sets[2]),
            },
        )

    result: list[AuditRow] = []
    for record_id in sorted(record_by_id):
        record = record_by_id[record_id]
        token = token_by_id[record_id]
        hidden = oracle_by_id[record_id]
        split = record.get("split")
        if split not in _SPLITS or token.get("split") != split or hidden.get("split") != split:
            raise ShortcutAuditError(
                "SHORTCUT_SPLIT_MISMATCH",
                "joined artifacts disagree on split",
                evidence={"record_id": record_id},
            )
        if hidden.get("visible_to_detector") is not False:
            raise ShortcutAuditError(
                "SHORTCUT_ORACLE_BOUNDARY",
                "oracle input must be marked invisible to the detector",
                evidence={"record_id": record_id},
            )
        variant = record.get("variant", {})
        if variant.get("label") not in {"H", "N"}:
            raise ShortcutAuditError(
                "SHORTCUT_LABEL_INVALID",
                "variant label must be H or N",
                evidence={"record_id": record_id},
            )
        arrays = (
            token.get("input_ids"),
            token.get("attention_mask"),
            token.get("evaluation_mask"),
            token.get("reasoning_mask"),
            token.get("answer_mask"),
            token.get("offset_mapping"),
        )
        if not arrays[0] or any(not isinstance(array, list) for array in arrays):
            raise ShortcutAuditError(
                "SHORTCUT_TOKEN_ARRAY_INVALID",
                "required token arrays must be non-empty lists",
                evidence={"record_id": record_id},
            )
        if len({len(array) for array in arrays}) != 1:
            raise ShortcutAuditError(
                "SHORTCUT_TOKEN_ARRAY_LENGTH",
                "joined token arrays must be equal length",
                evidence={"record_id": record_id},
            )
        result.append(AuditRow(record, token, hidden))

    pairs: dict[str, list[AuditRow]] = defaultdict(list)
    for row in result:
        pairs[row.pair_id].append(row)
    for pair_id, pair in pairs.items():
        if len(pair) != 2 or {row.label for row in pair} != {0, 1}:
            raise ShortcutAuditError(
                "SHORTCUT_PAIR_INCOMPLETE",
                "each audited pair must contain exactly one H and one N",
                evidence={"pair_id": pair_id, "record_ids": [row.record_id for row in pair]},
            )
        if len({row.origin_id for row in pair}) != 1:
            raise ShortcutAuditError(
                "SHORTCUT_PAIR_ORIGIN_MISMATCH",
                "matched H/N pair must share one origin",
                evidence={"pair_id": pair_id},
            )
    return tuple(result)


def _metadata_terms(row: AuditRow) -> tuple[str, ...]:
    mutation = row.record["mutation"]
    variant = row.record["variant"]
    fields = {
        "subtask": row.record["subtask"],
        "split": row.record["split"],
        "propagation": variant["propagation"],
        "candidate_source": mutation["candidate_source"],
        "operator_family": mutation["operator_family"],
        "operator_id": mutation["operator_id"],
        "renderer_id": mutation["renderer_id"],
        "root_state_id": mutation["root_state_id"],
    }
    return tuple(f"{key}={fields[key]}" for key in sorted(fields))


def _word_terms(text: str) -> tuple[str, ...]:
    tokens = _WORD_TOKEN.findall(text.lower())
    return tuple(tokens + [f"{left} {right}" for left, right in pairwise(tokens)])


def _char_terms(text: str) -> tuple[str, ...]:
    normalized = " ".join(text.lower().split())
    return tuple(
        normalized[index : index + width]
        for width in (3, 4, 5)
        for index in range(max(0, len(normalized) - width + 1))
    )


def _target_span_text(row: AuditRow) -> str:
    text = row.record["serialized"]["text"]
    if row.label:
        spans = row.record.get("spans", [])
        roots = [
            span
            for span in spans
            if span.get("causal_role") in {"ROOT", "TERMINAL"}
        ]
        if not roots:
            raise ShortcutAuditError(
                "SHORTCUT_ROOT_SPAN_MISSING",
                "H record has no root or terminal literal span",
                evidence={"record_id": row.record_id},
            )
        interval = roots[0].get("literal_span")
    else:
        interval = row.tokenized.get("matched_target_span")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(type(value) is not int for value in interval)
        or not 0 <= interval[0] < interval[1] <= len(text)
    ):
        raise ShortcutAuditError(
            "SHORTCUT_TARGET_SPAN_INVALID",
            "target-span baseline requires one valid half-open span",
            evidence={"record_id": row.record_id, "interval": interval},
        )
    return text[interval[0] : interval[1]]


def _group_folds(rows: Sequence[AuditRow]) -> dict[str, int]:
    members: dict[str, set[str]] = defaultdict(set)
    sizes: Counter[str] = Counter()
    for row in rows:
        members[row.leakage_group_id].add(row.origin_id)
        sizes[row.leakage_group_id] += 1
    if len(members) < T047_FOLD_COUNT:
        raise ShortcutAuditError(
            "SHORTCUT_GROUP_COUNT",
            "not enough leakage groups for the frozen grouped audit protocol",
            evidence={"group_count": len(members), "fold_count": T047_FOLD_COUNT},
        )
    order = sorted(
        members,
        key=lambda group: (-sizes[group], tuple(sorted(members[group])), group),
    )
    loads = [0] * T047_FOLD_COUNT
    assignment: dict[str, int] = {}
    for group in order:
        fold = min(range(T047_FOLD_COUNT), key=lambda index: (loads[index], index))
        assignment[group] = fold
        loads[fold] += sizes[group]
    if any(load == 0 for load in loads):
        raise ShortcutAuditError(
            "SHORTCUT_EMPTY_FOLD",
            "grouped audit protocol produced an empty fold",
            evidence={"fold_loads": loads},
        )
    return assignment


def _fit_tfidf(
    train_documents: Sequence[Any],
    analyzer: Callable[[Any], Sequence[str]],
    *,
    max_features: int,
    minimum_document_frequency: int = 2,
) -> tuple[Callable[[Any], dict[int, float]], int]:
    counts = [Counter(analyzer(document)) for document in train_documents]
    document_frequency = Counter(term for count in counts for term in count)
    ordered = sorted(document_frequency, key=lambda term: (-document_frequency[term], term))
    vocabulary_terms = [
        term
        for term in ordered
        if document_frequency[term] >= minimum_document_frequency
    ][:max_features]
    vocabulary = {term: index for index, term in enumerate(vocabulary_terms)}
    training_size = len(train_documents)

    def transform(document: Any) -> dict[int, float]:
        term_count = Counter(analyzer(document))
        vector = {
            vocabulary[term]: (1.0 + math.log(count))
            * (math.log((1.0 + training_size) / (1.0 + document_frequency[term])) + 1.0)
            for term, count in term_count.items()
            if term in vocabulary
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm:
            return {index: value / norm for index, value in vector.items()}
        return vector

    return transform, len(vocabulary)


def _fit_linear_logistic(
    vectors: Sequence[Mapping[int, float]],
    labels: Sequence[int],
    *,
    feature_count: int,
    iterations: int = 300,
    initial_learning_rate: float = 0.5,
    l2_penalty: float = 0.1,
) -> tuple[tuple[float, ...], float]:
    if len(vectors) != len(labels) or not vectors:
        raise ValueError("logistic fit requires equally sized non-empty inputs")
    weights = [0.0] * feature_count
    intercept = 0.0
    sample_count = len(vectors)
    for iteration in range(iterations):
        gradient = [0.0] * feature_count
        intercept_gradient = 0.0
        for vector, label in zip(vectors, labels):
            linear = intercept + sum(weights[index] * value for index, value in vector.items())
            bounded = max(-30.0, min(30.0, linear))
            probability = 1.0 / (1.0 + math.exp(-bounded))
            error = probability - label
            intercept_gradient += error
            for index, value in vector.items():
                gradient[index] += error * value
        rate = initial_learning_rate / (1.0 + 0.01 * iteration)
        for index in range(feature_count):
            weights[index] -= rate * (
                gradient[index] / sample_count + l2_penalty * weights[index]
            )
        intercept -= rate * intercept_gradient / sample_count
    return tuple(weights), intercept


def _linear_score(
    vector: Mapping[int, float], weights: Sequence[float], intercept: float
) -> float:
    return intercept + sum(weights[index] * value for index, value in vector.items())


def _grouped_linear_attack(
    rows: Sequence[AuditRow],
    documents: Sequence[Any],
    analyzer: Callable[[Any], Sequence[str]],
    *,
    max_features: int,
) -> _LinearAttack:
    if len(rows) != len(documents):
        raise ValueError("rows and attack documents must be equally sized")
    fold_by_group = _group_folds(rows)
    scores: list[float | None] = [None] * len(rows)
    vocabulary_sizes: list[int] = []
    for fold in range(T047_FOLD_COUNT):
        train_indices = [
            index
            for index, row in enumerate(rows)
            if fold_by_group[row.leakage_group_id] != fold
        ]
        test_indices = [
            index
            for index, row in enumerate(rows)
            if fold_by_group[row.leakage_group_id] == fold
        ]
        transform, feature_count = _fit_tfidf(
            [documents[index] for index in train_indices],
            analyzer,
            max_features=max_features,
        )
        vocabulary_sizes.append(feature_count)
        train_vectors = [transform(documents[index]) for index in train_indices]
        weights, intercept = _fit_linear_logistic(
            train_vectors,
            [rows[index].label for index in train_indices],
            feature_count=feature_count,
        )
        for index in test_indices:
            scores[index] = _linear_score(
                transform(documents[index]), weights, intercept
            )
    if any(score is None for score in scores):
        raise AssertionError("every row must receive exactly one out-of-fold score")
    return _LinearAttack(
        tuple(float(score) for score in scores),
        fold_by_group,
        tuple(vocabulary_sizes),
    )


def _fixed_train_linear_attack(
    train_rows: Sequence[AuditRow],
    train_documents: Sequence[Any],
    evaluation_documents: Sequence[Any],
    analyzer: Callable[[Any], Sequence[str]],
    *,
    max_features: int,
) -> tuple[tuple[float, ...], int]:
    if len(train_rows) != len(train_documents):
        raise ValueError("training rows and documents must be equally sized")
    transform, feature_count = _fit_tfidf(
        train_documents,
        analyzer,
        max_features=max_features,
    )
    weights, intercept = _fit_linear_logistic(
        [transform(document) for document in train_documents],
        [row.label for row in train_rows],
        feature_count=feature_count,
    )
    return (
        tuple(
            _linear_score(transform(document), weights, intercept)
            for document in evaluation_documents
        ),
        feature_count,
    )


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(labels) != len(scores):
        raise ValueError("AUROC labels and scores must be equally sized")
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        raise ShortcutAuditError(
            "SHORTCUT_AUROC_UNDEFINED",
            "AUROC requires both H and N records",
        )
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _bootstrap_free_metric(
    rows: Sequence[AuditRow], scores: Sequence[float]
) -> dict[str, Any]:
    labels = [row.label for row in rows]
    return {
        "auroc": _auroc(labels, scores),
        "negative_count": labels.count(0),
        "positive_count": labels.count(1),
        "score_max": max(scores),
        "score_min": min(scores),
    }


def _presence_log_odds_terms(
    rows: Sequence[AuditRow],
    documents: Sequence[Any],
    analyzer: Callable[[Any], Sequence[str]],
    *,
    limit: int = 12,
) -> dict[str, list[dict[str, Any]]]:
    present = {0: Counter[str](), 1: Counter[str]()}
    totals = Counter(row.label for row in rows)
    for row, document in zip(rows, documents):
        present[row.label].update(set(analyzer(document)))
    terms = sorted(set(present[0]) | set(present[1]))
    scored: list[tuple[float, str, int, int]] = []
    for term in terms:
        h_count = present[1][term]
        n_count = present[0][term]
        h_odds = (h_count + 0.5) / (totals[1] - h_count + 0.5)
        n_odds = (n_count + 0.5) / (totals[0] - n_count + 0.5)
        scored.append((math.log(h_odds / n_odds), term, h_count, n_count))

    def render(items: Sequence[tuple[float, str, int, int]]) -> list[dict[str, Any]]:
        return [
            {
                "term": term,
                "presence_log_odds_h_over_n": value,
                "h_document_count": h_count,
                "n_document_count": n_count,
            }
            for value, term, h_count, n_count in items
        ]

    return {
        "h_associated": render(
            sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]
        ),
        "n_associated": render(
            sorted(scored, key=lambda item: (item[0], item[1]))[:limit]
        ),
    }


def _retrieval_scores(
    rows: Sequence[AuditRow], documents: Sequence[str]
) -> tuple[float, ...]:
    fold_by_group = _group_folds(rows)
    result: list[float | None] = [None] * len(rows)
    for fold in range(T047_FOLD_COUNT):
        train_indices = [
            index
            for index, row in enumerate(rows)
            if fold_by_group[row.leakage_group_id] != fold
        ]
        test_indices = [
            index
            for index, row in enumerate(rows)
            if fold_by_group[row.leakage_group_id] == fold
        ]
        transform, _ = _fit_tfidf(
            [documents[index] for index in train_indices],
            _word_terms,
            max_features=5000,
        )
        training_vectors = {
            index: transform(documents[index]) for index in train_indices
        }
        for index in test_indices:
            query = transform(documents[index])
            neighbors: list[tuple[float, str, int]] = []
            for training_index, candidate in training_vectors.items():
                similarity = sum(
                    value * candidate.get(feature, 0.0)
                    for feature, value in query.items()
                )
                neighbors.append(
                    (similarity, rows[training_index].record_id, rows[training_index].label)
                )
            selected = sorted(neighbors, key=lambda item: (-item[0], item[1]))[:5]
            denominator = sum(max(0.0, item[0]) for item in selected)
            if denominator:
                result[index] = sum(
                    max(0.0, similarity) * label
                    for similarity, _, label in selected
                ) / denominator
            else:
                result[index] = 0.5
    if any(score is None for score in result):
        raise AssertionError("retrieval baseline did not score every row")
    return tuple(float(score) for score in result)


def _fixed_train_retrieval_scores(
    train_rows: Sequence[AuditRow],
    train_documents: Sequence[str],
    evaluation_rows: Sequence[AuditRow],
    evaluation_documents: Sequence[str],
) -> tuple[float, ...]:
    if len(train_rows) != len(train_documents):
        raise ValueError("retrieval training rows and documents must be equally sized")
    if len(evaluation_rows) != len(evaluation_documents):
        raise ValueError("retrieval evaluation rows and documents must be equally sized")
    transform, _ = _fit_tfidf(
        train_documents,
        _word_terms,
        max_features=5000,
    )
    training_vectors = [transform(document) for document in train_documents]
    scores: list[float] = []
    for evaluation_row, document in zip(evaluation_rows, evaluation_documents):
        query = transform(document)
        neighbors: list[tuple[float, str, int]] = []
        for train_row, candidate in zip(train_rows, training_vectors):
            if train_row.leakage_group_id == evaluation_row.leakage_group_id:
                raise ShortcutAuditError(
                    "SHORTCUT_HELDOUT_GROUP_LEAKAGE",
                    "held-out evaluation group appears in development training data",
                    evidence={"leakage_group_id": evaluation_row.leakage_group_id},
                )
            similarity = sum(
                value * candidate.get(feature, 0.0)
                for feature, value in query.items()
            )
            neighbors.append((similarity, train_row.record_id, train_row.label))
        selected = sorted(neighbors, key=lambda item: (-item[0], item[1]))[:5]
        denominator = sum(max(0.0, similarity) for similarity, _, _ in selected)
        scores.append(
            sum(
                max(0.0, similarity) * label
                for similarity, _, label in selected
            )
            / denominator
            if denominator
            else 0.5
        )
    return tuple(scores)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    return sum((value - average) ** 2 for value in values) / (len(values) - 1)


def _standardized_difference(h_values: Sequence[float], n_values: Sequence[float]) -> float:
    h_mean = _mean(h_values)
    n_mean = _mean(n_values)
    pooled = math.sqrt(
        (_sample_variance(h_values) + _sample_variance(n_values)) / 2.0
    )
    if pooled == 0.0:
        return 0.0 if h_mean == n_mean else math.inf
    return (h_mean - n_mean) / pooled


def _length_metric(rows: Sequence[AuditRow], extractor: Callable[[AuditRow], float]) -> dict[str, Any]:
    h_values = [extractor(row) for row in rows if row.label]
    n_values = [extractor(row) for row in rows if not row.label]
    difference = _standardized_difference(h_values, n_values)
    return {
        "h_mean": _mean(h_values),
        "n_mean": _mean(n_values),
        "standardized_difference": difference,
        "absolute_standardized_difference": abs(difference),
    }


def _pair_rows(rows: Sequence[AuditRow]) -> dict[str, tuple[AuditRow, AuditRow]]:
    grouped: dict[str, list[AuditRow]] = defaultdict(list)
    for row in rows:
        grouped[row.pair_id].append(row)
    return {
        pair_id: (
            next(row for row in pair if row.label),
            next(row for row in pair if not row.label),
        )
        for pair_id, pair in grouped.items()
    }


def _style_audit(rows: Sequence[AuditRow]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    inventory: dict[str, set[str]] = {field: set() for field in _PAIR_STYLE_FIELDS}
    for pair_id, (h_row, n_row) in sorted(_pair_rows(rows).items()):
        h_mutation = h_row.record["mutation"]
        n_mutation = n_row.record["mutation"]
        h_values = {
            "propagation": h_row.record["variant"]["propagation"],
            **{field: h_mutation[field] for field in _PAIR_STYLE_FIELDS[1:]},
        }
        n_values = {
            "propagation": n_row.record["variant"]["propagation"],
            **{field: n_mutation[field] for field in _PAIR_STYLE_FIELDS[1:]},
        }
        for field in _PAIR_STYLE_FIELDS:
            inventory[field].update((str(h_values[field]), str(n_values[field])))
            if h_values[field] != n_values[field]:
                mismatches.append(
                    {
                        "pair_id": pair_id,
                        "field": field,
                        "h_value": h_values[field],
                        "n_value": n_values[field],
                    }
                )
    return {
        "all_pairs_exact_on_frozen_style_fields": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "inventory": {
            field: sorted(values) for field, values in sorted(inventory.items())
        },
        "renderer_diversity_count": len(inventory["renderer_id"]),
    }


def _rdkit_backend() -> tuple[Any, Any]:
    try:
        from rdkit import Chem, rdBase
    except ImportError as exc:
        raise ShortcutAuditError(
            "SHORTCUT_RDKIT_UNAVAILABLE",
            "RDKit is required for validity and graph-comparator baselines",
        ) from exc
    return Chem, rdBase


def _extract_reasoning_product(row: AuditRow) -> str:
    reasoning = row.record["detector_input"]["reasoning_chain"]
    match = _PRODUCT_LINE.search(reasoning)
    if match is None:
        raise ShortcutAuditError(
            "SHORTCUT_PRODUCT_EXTRACTION",
            "visible reasoning has no parseable PRODUCT_CONSTRUCTION line",
            evidence={"record_id": row.record_id},
        )
    return match.group(1)


def _chemistry_baselines(rows: Sequence[AuditRow]) -> dict[str, Any]:
    Chem, rdBase = _rdkit_backend()
    validity_scores: list[float] = []
    visible_comparator_scores: list[float] = []
    oracle_comparator_scores: list[float] = []
    parse_rows: list[dict[str, Any]] = []
    with rdBase.BlockLogs():
        for row in rows:
            final_smiles = row.record["detector_input"]["final_answer"]
            reasoning_smiles = _extract_reasoning_product(row)
            oracle_smiles = row.oracle.get("gt_smiles")
            final_mol = Chem.MolFromSmiles(final_smiles)
            reasoning_mol = Chem.MolFromSmiles(reasoning_smiles)
            oracle_mol = (
                Chem.MolFromSmiles(oracle_smiles)
                if type(oracle_smiles) is str and oracle_smiles
                else None
            )
            final_valid = final_mol is not None
            reasoning_valid = reasoning_mol is not None
            oracle_valid = oracle_mol is not None
            any_visible_invalid = not (final_valid and reasoning_valid)
            validity_scores.append(float(any_visible_invalid))

            if final_valid and reasoning_valid:
                visible_equal = Chem.MolToSmiles(
                    final_mol, canonical=True, isomericSmiles=True
                ) == Chem.MolToSmiles(
                    reasoning_mol, canonical=True, isomericSmiles=True
                )
            else:
                visible_equal = False
            visible_comparator_scores.append(float(not visible_equal))

            if final_valid and oracle_valid:
                oracle_equal = Chem.MolToSmiles(
                    final_mol, canonical=True, isomericSmiles=True
                ) == Chem.MolToSmiles(
                    oracle_mol, canonical=True, isomericSmiles=True
                )
            else:
                oracle_equal = False
            oracle_comparator_scores.append(float(not oracle_equal))
            parse_rows.append(
                {
                    "label": row.label,
                    "final_valid": final_valid,
                    "reasoning_valid": reasoning_valid,
                }
            )

    labels = [row.label for row in rows]
    return {
        "smiles_validity": {
            **_bootstrap_free_metric(rows, validity_scores),
            "score_definition": "one_if_visible_final_or_reasoning_product_fails_rdkit_parse",
            "h_invalid_visible_count": sum(
                (not item["final_valid"] or not item["reasoning_valid"])
                for item in parse_rows
                if item["label"] == 1
            ),
            "n_invalid_visible_count": sum(
                (not item["final_valid"] or not item["reasoning_valid"])
                for item in parse_rows
                if item["label"] == 0
            ),
        },
        "visible_reasoning_answer_graph_comparator": {
            **_bootstrap_free_metric(rows, visible_comparator_scores),
            "score_definition": "one_if_reasoning_product_and_final_answer_are_not_isomeric_graph_equivalent",
        },
        "hidden_oracle_answer_graph_comparator": {
            **_bootstrap_free_metric(rows, oracle_comparator_scores),
            "score_definition": "one_if_final_answer_and_hidden_gt_are_not_isomeric_graph_equivalent",
            "detector_visible": False,
        },
        "slices": _symbolic_slices(rows, visible_comparator_scores, labels),
    }


def _symbolic_slices(
    rows: Sequence[AuditRow], scores: Sequence[float], labels: Sequence[int]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    policies = sorted({row.record["variant"]["propagation"] for row in rows})
    for policy in policies:
        indices = [
            index
            for index, row in enumerate(rows)
            if row.record["variant"]["propagation"] == policy
        ]
        result[policy] = {
            "auroc": _auroc(
                [labels[index] for index in indices],
                [scores[index] for index in indices],
            ),
            "record_count": len(indices),
        }
    return result


def _operator_renderer_analysis(rows: Sequence[AuditRow]) -> dict[str, Any]:
    by_operator: dict[str, list[AuditRow]] = defaultdict(list)
    by_renderer: dict[str, list[AuditRow]] = defaultdict(list)
    for row in rows:
        by_operator[row.record["mutation"]["operator_family"]].append(row)
        by_renderer[row.record["mutation"]["renderer_id"]].append(row)

    def summarize(group: Sequence[AuditRow]) -> dict[str, Any]:
        pairs = _pair_rows(group)
        token_deltas = [
            len(h_row.tokenized["input_ids"]) - len(n_row.tokenized["input_ids"])
            for h_row, n_row in pairs.values()
        ]
        char_deltas = [
            len(h_row.record["serialized"]["text"])
            - len(n_row.record["serialized"]["text"])
            for h_row, n_row in pairs.values()
        ]
        target_h_lengths = [
            float(len(_target_span_text(h_row))) for h_row, _ in pairs.values()
        ]
        target_n_lengths = [
            float(len(_target_span_text(n_row))) for _, n_row in pairs.values()
        ]
        return {
            "record_count": len(group),
            "pair_count": len(pairs),
            "max_absolute_pair_token_delta": max(map(abs, token_deltas), default=0),
            "mean_pair_character_delta_h_minus_n": _mean(char_deltas)
            if char_deltas
            else 0.0,
            "target_literal_character_standardized_difference": (
                _standardized_difference(target_h_lengths, target_n_lengths)
                if target_h_lengths and target_n_lengths
                else 0.0
            ),
            "max_absolute_target_literal_character_delta": max(
                (
                    abs(len(_target_span_text(h_row)) - len(_target_span_text(n_row)))
                    for h_row, n_row in pairs.values()
                ),
                default=0,
            ),
        }

    return {
        "operators": {
            key: summarize(value) for key, value in sorted(by_operator.items())
        },
        "renderers": {
            key: summarize(value) for key, value in sorted(by_renderer.items())
        },
    }


def _remediation(
    rows: Sequence[AuditRow],
    failures: Sequence[Mapping[str, Any]],
    span_terms: Mapping[str, Any],
) -> dict[str, Any]:
    if not failures:
        return {
            "required": False,
            "status": "not_needed",
            "strict_rerun_command": (
                "python -m molhallulens.audit.shortcut_audit --strict"
            ),
        }
    pairs = _pair_rows(rows)
    ranked_pairs: list[dict[str, Any]] = []
    for pair_id, (h_row, n_row) in sorted(pairs.items()):
        h_target = _target_span_text(h_row)
        n_target = _target_span_text(n_row)
        ranked_pairs.append(
            {
                "pair_id": pair_id,
                "origin_id": h_row.origin_id,
                "operator_family": h_row.record["mutation"]["operator_family"],
                "renderer_id": h_row.record["mutation"]["renderer_id"],
                "target_character_delta_h_minus_n": len(h_target) - len(n_target),
                "full_token_delta_h_minus_n": len(h_row.tokenized["input_ids"])
                - len(n_row.tokenized["input_ids"]),
            }
        )
    ranked_pairs.sort(
        key=lambda item: (
            -abs(item["target_character_delta_h_minus_n"]),
            item["pair_id"],
        )
    )
    return {
        "required": True,
        "status": "requires_t045_surface_design_rebuild",
        "failing_metrics": [dict(item) for item in failures],
        "actions": [
            {
                "action": "regenerate_complete_matched_pairs",
                "owner": "T045 MatchedBundleBuilder / label-blind renderer",
                "constraint": (
                    "regenerate H and N together; preserve locked chemistry, policy, "
                    "operator, validator gates, and one-H/one-N pair integrity"
                ),
                "selection": "pairs in affected operator families, prioritized by target span imbalance",
            },
            {
                "action": "cross_balance_target_literals",
                "owner": "T045 candidate selection",
                "constraint": (
                    "reselect from already validated candidate pools so H replacement "
                    "literals also occur as faithful values in other origins; never relax chemistry gates"
                ),
            },
            {
                "action": "rebuild_then_strict_rerun",
                "command": (
                    "python -m molhallulens.audit.shortcut_audit "
                    "--dry-run-root HallucinationDataset/dry_run "
                    "--report Dataset/reports/t047_shortcut_audit.json --strict"
                ),
                "success_condition": (
                    "process exits zero only after every mandatory threshold passes"
                ),
            },
        ],
        "priority_pairs": ranked_pairs[:24],
        "span_presence_diagnostics": span_terms,
        "safety_constraints": [
            "do not remove or relabel failed examples after observing audit labels",
            "do not relax RDKit, graph-edit, propagation, renderer, or token gates",
            "do not change H/N or propagation quotas",
            "do not use the test split to tune classifier, layer, or detector threshold",
        ],
    }


def _run_development_rows(
    rows: Sequence[AuditRow],
    *,
    inventory_rows: Sequence[AuditRow],
) -> dict[str, Any]:
    if not rows or not inventory_rows:
        raise ShortcutAuditError(
            "SHORTCUT_EMPTY_INVENTORY",
            "shortcut audit requires non-empty development and release inventories",
        )
    audit_inventory_id = rows[0].record.get("dry_run_id")
    if audit_inventory_id is None:
        audit_inventory_id = rows[0].record.get("dataset_version")
    if type(audit_inventory_id) is not str or not audit_inventory_id:
        raise ShortcutAuditError(
            "SHORTCUT_INVENTORY_ID_MISSING",
            "shortcut inventory needs a dry-run identity or release dataset version",
        )
    labels = [row.label for row in rows]
    metadata_documents = [_metadata_terms(row) for row in rows]
    span_documents = [_target_span_text(row) for row in rows]
    reasoning_documents = [
        row.record["detector_input"]["reasoning_chain"] for row in rows
    ]

    metadata_attack = _grouped_linear_attack(
        rows,
        metadata_documents,
        lambda terms: terms,
        max_features=256,
    )
    span_attack = _grouped_linear_attack(
        rows,
        span_documents,
        _char_terms,
        max_features=5000,
    )
    reasoning_attack = _grouped_linear_attack(
        rows,
        reasoning_documents,
        _word_terms,
        max_features=5000,
    )
    retrieval_scores = _retrieval_scores(rows, reasoning_documents)

    metadata_metric = _bootstrap_free_metric(rows, metadata_attack.scores)
    span_metric = _bootstrap_free_metric(rows, span_attack.scores)
    reasoning_metric = _bootstrap_free_metric(rows, reasoning_attack.scores)
    retrieval_metric = _bootstrap_free_metric(rows, retrieval_scores)
    for metric, attack in (
        (metadata_metric, metadata_attack),
        (span_metric, span_attack),
        (reasoning_metric, reasoning_attack),
    ):
        metric["fold_vocabulary_sizes"] = list(attack.vocabulary_sizes)

    length_matching = {
        "full_token_count": _length_metric(
            rows, lambda row: float(len(row.tokenized["input_ids"]))
        ),
        "evaluated_token_count": _length_metric(
            rows, lambda row: float(sum(row.tokenized["evaluation_mask"]))
        ),
        "reasoning_token_count": _length_metric(
            rows, lambda row: float(sum(row.tokenized["reasoning_mask"]))
        ),
        "answer_token_count": _length_metric(
            rows, lambda row: float(sum(row.tokenized["answer_mask"]))
        ),
        "serialized_character_count": _length_metric(
            rows, lambda row: float(len(row.record["serialized"]["text"]))
        ),
    }
    pairs = _pair_rows(rows)
    length_matching["max_absolute_pair_token_delta"] = max(
        abs(len(h.tokenized["input_ids"]) - len(n.tokenized["input_ids"]))
        for h, n in pairs.values()
    )
    style = _style_audit(rows)
    chemistry = _chemistry_baselines(rows)
    span_terms = _presence_log_odds_terms(rows, span_documents, _char_terms)
    operator_renderer_analysis = _operator_renderer_analysis(rows)
    operator_renderer_analysis["span_presence_diagnostics"] = span_terms
    operator_renderer_analysis["renderer_diversity_warning"] = (
        style["renderer_diversity_count"] < 2
    )

    gates = {
        "metadata_auroc": {
            "actual": metadata_metric["auroc"],
            "comparator": "<=",
            "threshold": T047_METADATA_AUROC_LIMIT,
            "passed": metadata_metric["auroc"] <= T047_METADATA_AUROC_LIMIT,
        },
        "span_only_tfidf_auroc": {
            "actual": span_metric["auroc"],
            "comparator": "<=",
            "threshold": T047_SPAN_TFIDF_AUROC_LIMIT,
            "passed": span_metric["auroc"] <= T047_SPAN_TFIDF_AUROC_LIMIT,
        },
        "reasoning_only_shallow_auroc": {
            "actual": reasoning_metric["auroc"],
            "comparator": "<=",
            "threshold": T047_REASONING_AUROC_LIMIT,
            "passed": reasoning_metric["auroc"] <= T047_REASONING_AUROC_LIMIT,
        },
        "token_length_standardized_difference": {
            "actual": length_matching["full_token_count"][
                "absolute_standardized_difference"
            ],
            "comparator": "<",
            "threshold": T047_TOKEN_LENGTH_SMD_LIMIT,
            "passed": length_matching["full_token_count"][
                "absolute_standardized_difference"
            ]
            < T047_TOKEN_LENGTH_SMD_LIMIT,
        },
        "style_pair_matching": {
            "actual": style["mismatch_count"],
            "comparator": "==",
            "threshold": 0,
            "passed": style["mismatch_count"] == 0,
        },
    }
    failures = [
        {"metric": name, **gate}
        for name, gate in gates.items()
        if not gate["passed"]
    ]
    origin_ids = {row.origin_id for row in inventory_rows}
    leakage_groups = {row.leakage_group_id for row in inventory_rows}
    development_origin_ids = {row.origin_id for row in rows}
    development_groups = {row.leakage_group_id for row in rows}
    report = {
        "format_version": T047_REPORT_FORMAT_VERSION,
        "dry_run_id": audit_inventory_id,
        "audit_protocol": {
            "id": T047_AUDIT_PROTOCOL,
            "fold_count": T047_FOLD_COUNT,
            "group_unit": "leakage_group_id",
            "out_of_fold_predictions_only": True,
            "mandatory_gate_splits": ["train", "validation"],
            "held_out_split": "test",
            "molecule_comparison": "RDKit canonical isomeric graph equivalence",
            "test_used_for_model_or_threshold_selection": False,
        },
        "inventory": {
            "record_count": len(inventory_rows),
            "origin_count": len(origin_ids),
            "leakage_group_count": len(leakage_groups),
            "pair_count": len(_pair_rows(inventory_rows)),
            "h_count": sum(row.label for row in inventory_rows),
            "n_count": sum(not row.label for row in inventory_rows),
            "split_record_counts": dict(
                sorted(
                    Counter(row.record["split"] for row in inventory_rows).items()
                )
            ),
        },
        "development_inventory": {
            "record_count": len(rows),
            "origin_count": len(development_origin_ids),
            "leakage_group_count": len(development_groups),
            "pair_count": len(pairs),
            "h_count": labels.count(1),
            "n_count": labels.count(0),
            "splits": ["train", "validation"],
        },
        "mandatory_gates": gates,
        "baselines": {
            "metadata_only_logistic": metadata_metric,
            "span_only_char_tfidf_logistic": span_metric,
            "reasoning_only_word_tfidf_logistic": reasoning_metric,
            "nearest_neighbor_retrieval_k5": retrieval_metric,
            **chemistry,
        },
        "matching": {
            "length": length_matching,
            "style": style,
        },
        "operator_renderer_failure_analysis": operator_renderer_analysis,
        "threshold_failure_count": len(failures),
        "threshold_failures": failures,
        "remediation": _remediation(rows, failures, span_terms),
        "limitations": [
            "The dry run has 15 origins; attack AUROCs are engineering screens, not paper estimates.",
            "Mandatory gates use only train and validation; held-out test diagnostics cannot trigger candidate reselection.",
            "The frozen tokenizer artifact is an offline offset fixture, not production ChemDFM-R weights.",
            "A high symbolic comparator score is legitimate executable chemistry signal and is reported, not suppressed.",
            "Renderer diversity is diagnosed separately from exact within-pair style matching.",
        ],
        "all_pass": not failures,
    }
    return _stable(report)


def _heldout_test_diagnostics(
    development_rows: Sequence[AuditRow],
    test_rows: Sequence[AuditRow],
) -> dict[str, Any]:
    if not test_rows:
        raise ShortcutAuditError(
            "SHORTCUT_TEST_EMPTY",
            "final shortcut audit requires a non-empty held-out test split",
        )
    overlap = {row.leakage_group_id for row in development_rows} & {
        row.leakage_group_id for row in test_rows
    }
    if overlap:
        raise ShortcutAuditError(
            "SHORTCUT_HELDOUT_GROUP_LEAKAGE",
            "held-out test shares leakage groups with development data",
            evidence={"overlapping_group_count": len(overlap)},
        )

    development_metadata = [_metadata_terms(row) for row in development_rows]
    test_metadata = [_metadata_terms(row) for row in test_rows]
    development_spans = [_target_span_text(row) for row in development_rows]
    test_spans = [_target_span_text(row) for row in test_rows]
    development_reasoning = [
        row.record["detector_input"]["reasoning_chain"] for row in development_rows
    ]
    test_reasoning = [
        row.record["detector_input"]["reasoning_chain"] for row in test_rows
    ]

    metadata_scores, metadata_vocabulary_size = _fixed_train_linear_attack(
        development_rows,
        development_metadata,
        test_metadata,
        lambda terms: terms,
        max_features=256,
    )
    span_scores, span_vocabulary_size = _fixed_train_linear_attack(
        development_rows,
        development_spans,
        test_spans,
        _char_terms,
        max_features=5000,
    )
    reasoning_scores, reasoning_vocabulary_size = _fixed_train_linear_attack(
        development_rows,
        development_reasoning,
        test_reasoning,
        _word_terms,
        max_features=5000,
    )
    retrieval_scores = _fixed_train_retrieval_scores(
        development_rows,
        development_reasoning,
        test_rows,
        test_reasoning,
    )
    metadata_metric = _bootstrap_free_metric(test_rows, metadata_scores)
    span_metric = _bootstrap_free_metric(test_rows, span_scores)
    reasoning_metric = _bootstrap_free_metric(test_rows, reasoning_scores)
    metadata_metric["training_vocabulary_size"] = metadata_vocabulary_size
    span_metric["training_vocabulary_size"] = span_vocabulary_size
    reasoning_metric["training_vocabulary_size"] = reasoning_vocabulary_size

    style = _style_audit(test_rows)
    length = {
        "full_token_count": _length_metric(
            test_rows, lambda row: float(len(row.tokenized["input_ids"]))
        ),
        "evaluated_token_count": _length_metric(
            test_rows, lambda row: float(sum(row.tokenized["evaluation_mask"]))
        ),
        "serialized_character_count": _length_metric(
            test_rows, lambda row: float(len(row.record["serialized"]["text"]))
        ),
    }
    return _stable(
        {
            "scope": "held_out_test_read_once_after_development_design_freeze",
            "used_for_candidate_layer_or_threshold_selection": False,
            "record_count": len(test_rows),
            "origin_count": len({row.origin_id for row in test_rows}),
            "training_splits": ["train", "validation"],
            "evaluation_split": "test",
            "baselines": {
                "metadata_only_logistic": metadata_metric,
                "span_only_char_tfidf_logistic": span_metric,
                "reasoning_only_word_tfidf_logistic": reasoning_metric,
                "nearest_neighbor_retrieval_k5": _bootstrap_free_metric(
                    test_rows, retrieval_scores
                ),
                **_chemistry_baselines(test_rows),
            },
            "matching": {"length": length, "style": style},
            "reference_thresholds_not_used_as_reselection_gates": {
                "metadata_auroc": T047_METADATA_AUROC_LIMIT,
                "span_only_tfidf_auroc": T047_SPAN_TFIDF_AUROC_LIMIT,
                "reasoning_only_shallow_auroc": T047_REASONING_AUROC_LIMIT,
                "token_length_standardized_difference": T047_TOKEN_LENGTH_SMD_LIMIT,
            },
        }
    )


def run_t047_development_audit(
    *, dry_run_root: Path | None = None
) -> dict[str, Any]:
    """Run mandatory gates without reading the held-out test artifact."""

    root = DEFAULT_DRY_RUN_ROOT if dry_run_root is None else Path(dry_run_root)
    development_rows = load_t047_audit_rows(
        root,
        splits=("train", "validation"),
    )
    report = _run_development_rows(
        development_rows,
        inventory_rows=development_rows,
    )
    report["report_scope"] = "development_only_test_not_read"
    return _stable(report)


def run_t047_shortcut_audit(
    *, dry_run_root: Path | None = None
) -> dict[str, Any]:
    """Run final dev-gated audit plus one untouched held-out test diagnostic."""

    root = DEFAULT_DRY_RUN_ROOT if dry_run_root is None else Path(dry_run_root)
    rows = load_t047_audit_rows(root)
    development_rows = tuple(row for row in rows if row.record["split"] != "test")
    test_rows = tuple(row for row in rows if row.record["split"] == "test")
    report = _run_development_rows(development_rows, inventory_rows=rows)
    report["report_scope"] = "final_with_heldout_test_diagnostic"
    report["heldout_test_diagnostics"] = _heldout_test_diagnostics(
        development_rows,
        test_rows,
    )
    return _stable(report)


def render_t047_report(report: Mapping[str, Any]) -> str:
    return (
        json.dumps(_stable(report), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )


def write_t047_shortcut_report(
    *,
    dry_run_root: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Run the audit and write its report, replacing only the T047 report."""

    report = run_t047_shortcut_audit(dry_run_root=dry_run_root)
    path = DEFAULT_REPORT_PATH if report_path is None else Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_t047_report(report), encoding="utf-8", newline="\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the T047 dry-run shortcut audit")
    parser.add_argument("--dry-run-root", type=Path, default=DEFAULT_DRY_RUN_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--remediation-manifest", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return exit status 2 when a mandatory shortcut gate fails",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = write_t047_shortcut_report(
        dry_run_root=arguments.dry_run_root,
        report_path=arguments.report,
    )
    if arguments.remediation_manifest is not None:
        arguments.remediation_manifest.parent.mkdir(parents=True, exist_ok=True)
        arguments.remediation_manifest.write_text(
            json.dumps(
                report["remediation"],
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if arguments.strict and not report["all_pass"]:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DRY_RUN_ROOT",
    "DEFAULT_REPORT_PATH",
    "T047_AUDIT_PROTOCOL",
    "T047_FOLD_COUNT",
    "T047_METADATA_AUROC_LIMIT",
    "T047_REASONING_AUROC_LIMIT",
    "T047_REPORT_FORMAT_VERSION",
    "T047_SPAN_TFIDF_AUROC_LIMIT",
    "T047_TOKEN_LENGTH_SMD_LIMIT",
    "AuditRow",
    "ShortcutAuditError",
    "load_t047_audit_rows",
    "main",
    "render_t047_report",
    "run_t047_development_audit",
    "run_t047_shortcut_audit",
    "write_t047_shortcut_report",
]
