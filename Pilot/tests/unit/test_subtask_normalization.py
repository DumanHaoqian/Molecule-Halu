"""Tests for explicit, table-driven subtask normalization."""

from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from molhallulens.modules.ingestion import (
    AmbiguousSubtaskError,
    ChemCoTMolEditAdapter,
    DEFAULT_SUBTASK_NORMALIZER,
    MOLECULE_EDITING_SUBTASK_MAPPINGS,
    SubtaskMapping,
    SubtaskNormalizer,
    UnknownSubtaskError,
)
from molhallulens.config.loader import load_config_bundle
from molhallulens.core import EditingSubtask


DATASET_ROOT = Path(__file__).resolve().parents[2] / "Dataset"
EXPECTED = (
    ("add_pilot_origin", "add_v2", EditingSubtask.ADD),
    ("delete_pilot_origin", "delete_v2", EditingSubtask.DELETE),
    ("substitute_pilot_origin", "substitute_v2", EditingSubtask.SUBSTITUTE),
)


def test_frozen_registry_contains_the_three_explicit_mappings() -> None:
    assert tuple(
        (
            mapping.pilot_subtask,
            mapping.source_subtask,
            mapping.normalized_subtask,
        )
        for mapping in MOLECULE_EDITING_SUBTASK_MAPPINGS
    ) == EXPECTED
    assert load_config_bundle().dataset.input.subtasks == tuple(
        expected[2].value for expected in EXPECTED
    )


@pytest.mark.parametrize(
    ("expected", "alias_index"),
    [
        (expected, alias_index)
        for expected in EXPECTED
        for alias_index in range(3)
    ],
)
def test_all_nine_exact_names_resolve_to_the_complete_mapping(
    expected: tuple[str, str, EditingSubtask],
    alias_index: int,
) -> None:
    alias = expected[alias_index]
    name = alias.value if type(alias) is EditingSubtask else alias

    resolved = DEFAULT_SUBTASK_NORMALIZER.normalize(name)

    assert (
        resolved.pilot_subtask,
        resolved.source_subtask,
        resolved.normalized_subtask,
    ) == expected
    assert DEFAULT_SUBTASK_NORMALIZER.for_normalized(expected[2]) is resolved


def test_real_joined_pilot_normalizes_all_150_records_without_branching() -> None:
    joined = ChemCoTMolEditAdapter().load(DATASET_ROOT)
    resolved = tuple(
        DEFAULT_SUBTASK_NORMALIZER.normalize(record.pilot_subtask)
        for record in joined
    )

    assert len(resolved) == 150
    assert Counter(mapping.normalized_subtask for mapping in resolved) == {
        EditingSubtask.ADD: 50,
        EditingSubtask.DELETE: 50,
        EditingSubtask.SUBSTITUTE: 50,
    }
    assert Counter(mapping.source_subtask for mapping in resolved) == {
        "add_v2": 50,
        "delete_v2": 50,
        "substitute_v2": 50,
    }


@pytest.mark.parametrize(
    "unknown",
    (
        "",
        " ",
        "ADD",
        " add",
        "add ",
        "add_v3",
        "addition",
        "MolEdit/Add",
        "add_pilot_origin.json",
        "raw_benchmark_data/mol_edit/add_pilot_origin.json",
        "mol_edit.add_v2.0001",
    ),
)
def test_unknown_or_noncanonical_names_fail_closed(unknown: str) -> None:
    with pytest.raises(UnknownSubtaskError):
        DEFAULT_SUBTASK_NORMALIZER.normalize(unknown)


@pytest.mark.parametrize("invalid", (None, 1, EditingSubtask.ADD))
def test_non_string_lookup_values_are_rejected(invalid: object) -> None:
    with pytest.raises(TypeError):
        DEFAULT_SUBTASK_NORMALIZER.normalize(invalid)  # type: ignore[arg-type]


def test_conflicting_explicit_names_fail_closed() -> None:
    with pytest.raises(AmbiguousSubtaskError):
        DEFAULT_SUBTASK_NORMALIZER.reconcile("add_pilot_origin", "delete_v2")


def test_matching_explicit_names_reconcile_to_one_identity() -> None:
    for pilot_subtask, source_subtask, normalized_subtask in EXPECTED:
        resolved = DEFAULT_SUBTASK_NORMALIZER.reconcile(
            normalized_subtask.value,
            pilot_subtask,
            source_subtask,
        )
        assert resolved.normalized_subtask is normalized_subtask


def test_ambiguous_alias_definitions_fail_during_registry_construction() -> None:
    addition = MOLECULE_EDITING_SUBTASK_MAPPINGS[0]
    collision = replace(
        MOLECULE_EDITING_SUBTASK_MAPPINGS[1],
        pilot_subtask=addition.source_subtask,
    )

    with pytest.raises(AmbiguousSubtaskError, match="registered more than once"):
        SubtaskNormalizer((addition, collision))


def test_duplicate_mapping_definitions_also_fail_closed() -> None:
    mapping = MOLECULE_EDITING_SUBTASK_MAPPINGS[0]

    with pytest.raises(AmbiguousSubtaskError):
        SubtaskNormalizer((mapping, mapping))


def test_registry_is_input_order_independent_and_repeatable() -> None:
    forward = SubtaskNormalizer(MOLECULE_EDITING_SUBTASK_MAPPINGS)
    reverse = SubtaskNormalizer(reversed(MOLECULE_EDITING_SUBTASK_MAPPINGS))

    assert forward.accepted_names == reverse.accepted_names
    for name in forward.accepted_names:
        assert forward.normalize(name) == reverse.normalize(name)
        assert forward.normalize(name) is forward.normalize(name)


def test_registry_and_mapping_are_immutable_and_defensively_indexed() -> None:
    normalizer = SubtaskNormalizer()
    mapping = normalizer.normalize("add_v2")

    with pytest.raises(FrozenInstanceError):
        mapping.source_subtask = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        normalizer.mappings = ()  # type: ignore[misc]
    assert normalizer.accepted_names == tuple(
        sorted(
            alias
            for pilot, source, normalized in EXPECTED
            for alias in (pilot, source, normalized.value)
        )
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"pilot_subtask": " add_pilot_origin"},
        {"pilot_subtask": ""},
        {"pilot_subtask": "add_v2"},
        {"source_subtask": "add_v2 "},
        {"normalized_subtask": "add"},
    ),
)
def test_mapping_definitions_reject_noncanonical_fields(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "pilot_subtask": "add_pilot_origin",
        "source_subtask": "add_v2",
        "normalized_subtask": EditingSubtask.ADD,
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        SubtaskMapping(**values)  # type: ignore[arg-type]
