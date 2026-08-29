"""Corpus-level acceptance test for the 150 molecule-editing reference DAGs."""

from __future__ import annotations

from pathlib import Path

from molhallulens.adapters import ChemCoTMolEditAdapter
from molhallulens.builders import build_reference_dag_corpus
from molhallulens.domain import Visibility


DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"


def test_real_150_origin_reference_dag_build_report() -> None:
    records = ChemCoTMolEditAdapter().load(DATASET_ROOT)
    result = build_reference_dag_corpus(reversed(records))
    report = result.report.to_dict()

    assert tuple(artifact.anonymous_sample_id for artifact in result.artifacts) == tuple(
        sorted(record.anonymous_sample_id for record in records)
    )
    assert report["summary"] == {
        "attempted": 150,
        "succeeded": 150,
        "failed": 0,
        "counts_by_subtask": {"add": 50, "delete": 50, "substitute": 50},
        "trace_steps": 800,
        "natural_round_trip_steps": 800,
        "formal_round_trip_slots": 2350,
        "node_values": 3400,
        "schema_edges": 3250,
        "logical_mentions": 2800,
        "raw_answer_gt_string_mismatches": 7,
        "build_only_detector_mentions": 0,
        "issue_count": 0,
    }
    assert len(report["origins"]) == 150
    assert all(item["status"] == "built" for item in report["origins"])
    assert all(
        artifact.state_dag.schema.nodes_by_id[mention.node_id].visibility
        is not Visibility.BUILD_ONLY
        for artifact in result.artifacts
        for mention in artifact.mentions
    )
