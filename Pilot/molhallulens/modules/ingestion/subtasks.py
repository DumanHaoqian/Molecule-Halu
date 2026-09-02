"""Explicit, fail-closed subtask-name normalization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from molhallulens.core import EditingSubtask


class SubtaskNormalizationError(ValueError):
    """Base error for names that cannot be resolved unambiguously."""


class UnknownSubtaskError(SubtaskNormalizationError):
    """Raised when a name is not explicitly present in the frozen registry."""


class AmbiguousSubtaskError(SubtaskNormalizationError):
    """Raised when registry definitions assign one name more than once."""


@dataclass(frozen=True, slots=True)
class SubtaskMapping:
    """The three canonical names for one molecule-editing subtask."""

    pilot_subtask: str
    source_subtask: str
    normalized_subtask: EditingSubtask

    def __post_init__(self) -> None:
        text_fields = {
            "pilot_subtask": self.pilot_subtask,
            "source_subtask": self.source_subtask,
        }
        if any(type(value) is not str for value in text_fields.values()):
            raise TypeError("SubtaskMapping names must be strings")
        invalid = tuple(
            sorted(
                name
                for name, value in text_fields.items()
                if not value or value != value.strip()
            )
        )
        if invalid:
            raise ValueError(
                f"SubtaskMapping names must be non-empty canonical strings: {invalid}"
            )
        if type(self.normalized_subtask) is not EditingSubtask:
            raise TypeError(
                "SubtaskMapping normalized_subtask must be an EditingSubtask"
            )
        if len(set(self.accepted_names)) != 3:
            raise ValueError("SubtaskMapping canonical names must be distinct")

    @property
    def accepted_names(self) -> tuple[str, str, str]:
        """Exact aliases accepted for this mapping; no inference is performed."""

        return (
            self.pilot_subtask,
            self.source_subtask,
            self.normalized_subtask.value,
        )


MOLECULE_EDITING_SUBTASK_MAPPINGS = (
    SubtaskMapping(
        pilot_subtask="add_pilot_origin",
        source_subtask="add_v2",
        normalized_subtask=EditingSubtask.ADD,
    ),
    SubtaskMapping(
        pilot_subtask="delete_pilot_origin",
        source_subtask="delete_v2",
        normalized_subtask=EditingSubtask.DELETE,
    ),
    SubtaskMapping(
        pilot_subtask="substitute_pilot_origin",
        source_subtask="substitute_v2",
        normalized_subtask=EditingSubtask.SUBSTITUTE,
    ),
)


@dataclass(frozen=True, slots=True, init=False)
class SubtaskNormalizer:
    """Resolve only explicitly registered names through immutable table lookups."""

    mappings: tuple[SubtaskMapping, ...]
    _by_name: Mapping[str, SubtaskMapping] = field(repr=False, compare=False)
    _by_normalized: Mapping[EditingSubtask, SubtaskMapping] = field(
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        mappings: Iterable[SubtaskMapping] = MOLECULE_EDITING_SUBTASK_MAPPINGS,
    ) -> None:
        if isinstance(mappings, (str, bytes)) or not isinstance(mappings, Iterable):
            raise TypeError("SubtaskNormalizer mappings must be an iterable")
        frozen_mappings = tuple(mappings)
        if not frozen_mappings:
            raise ValueError("SubtaskNormalizer requires at least one mapping")
        if any(type(mapping) is not SubtaskMapping for mapping in frozen_mappings):
            raise TypeError("SubtaskNormalizer mappings must be SubtaskMapping values")

        by_name: dict[str, SubtaskMapping] = {}
        by_normalized: dict[EditingSubtask, SubtaskMapping] = {}
        for mapping in frozen_mappings:
            for name in dict.fromkeys(mapping.accepted_names):
                if name in by_name:
                    raise AmbiguousSubtaskError(
                        f"subtask name is registered more than once: {name!r}"
                    )
                by_name[name] = mapping
            if mapping.normalized_subtask in by_normalized:
                raise AmbiguousSubtaskError(
                    "normalized subtask is registered more than once: "
                    f"{mapping.normalized_subtask.value!r}"
                )
            by_normalized[mapping.normalized_subtask] = mapping

        object.__setattr__(self, "mappings", frozen_mappings)
        object.__setattr__(self, "_by_name", MappingProxyType(by_name))
        object.__setattr__(self, "_by_normalized", MappingProxyType(by_normalized))

    @property
    def accepted_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))

    def normalize(self, name: str) -> SubtaskMapping:
        """Return the complete canonical mapping for one exact registered name."""

        if type(name) is not str:
            raise TypeError("subtask name must be a string")
        try:
            return self._by_name[name]
        except KeyError as error:
            raise UnknownSubtaskError(f"unknown subtask name: {name!r}") from error

    def for_normalized(self, subtask: EditingSubtask) -> SubtaskMapping:
        """Resolve a typed normalized enum without StrEnum/string key collisions."""

        if type(subtask) is not EditingSubtask:
            raise TypeError("normalized subtask must be an EditingSubtask")
        try:
            return self._by_normalized[subtask]
        except KeyError as error:
            raise UnknownSubtaskError(
                f"unknown normalized subtask: {subtask.value!r}"
            ) from error

    def reconcile(self, *names: str) -> SubtaskMapping:
        """Require multiple explicit source names to resolve to the same table row."""

        if not names:
            raise ValueError("at least one subtask name is required")
        resolved = tuple(self.normalize(name) for name in names)
        first = resolved[0]
        if any(mapping is not first for mapping in resolved[1:]):
            raise AmbiguousSubtaskError("subtask names resolve to conflicting mappings")
        return first


DEFAULT_SUBTASK_NORMALIZER = SubtaskNormalizer()
