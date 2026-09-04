"""Small enum vocabulary used by the current molecule-editing pipeline."""

from __future__ import annotations

from enum import EnumType, StrEnum
from types import MappingProxyType


_SEALED_ENUM_MEMBER_IDS: set[int] = set()


class _FrozenEnumType(EnumType):
    def __new__(
        metacls: type[_FrozenEnumType],
        cls: str,
        bases: tuple[type, ...],
        classdict: dict[str, object],
        **kwds: object,
    ) -> _FrozenEnumType:
        enum_class = super().__new__(metacls, cls, bases, classdict, **kwds)
        _SEALED_ENUM_MEMBER_IDS.update(
            id(member) for member in enum_class.__members__.values()
        )
        return enum_class


class _FrozenEnumMixin:
    def __getattribute__(self, name: str) -> object:
        if name == "__dict__" and id(self) in _SEALED_ENUM_MEMBER_IDS:
            raw_values = object.__getattribute__(self, "__dict__")
            return MappingProxyType(dict(raw_values))
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: object) -> None:
        if id(self) in _SEALED_ENUM_MEMBER_IDS:
            raise TypeError("MolHalluLens enum members are immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if id(self) in _SEALED_ENUM_MEMBER_IDS:
            raise TypeError("MolHalluLens enum members are immutable")
        object.__delattr__(self, name)


class _DomainStrEnum(_FrozenEnumMixin, StrEnum, metaclass=_FrozenEnumType):
    pass


def is_domain_enum_member(value: object) -> bool:
    """Return whether value is one of this module's immutable enum members."""

    return id(value) in _SEALED_ENUM_MEMBER_IDS


class EditingSubtask(_DomainStrEnum):
    ADD = "add"
    DELETE = "delete"
    SUBSTITUTE = "substitute"


class Visibility(_DomainStrEnum):
    BUILD_ONLY = "build_only"
    PROMPT_PREFIX = "prompt_prefix"
    CANDIDATE_OUTPUT = "candidate_output"


class NodeRole(_DomainStrEnum):
    EVIDENCE = "evidence"
    INTERNAL_TRUTH = "internal_truth"
    PRIMARY_CLAIM = "primary_claim"
    DERIVED_CLAIM = "derived_claim"
    FINAL_ANSWER = "final_answer"


class ValueType(_DomainStrEnum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    SMILES = "smiles"
    INDEXED_SMILES = "indexed_smiles"
    ATOM_INDEX = "atom_index"
    ELEMENT = "element"
    FRAGMENT = "fragment"
    MOLECULE = "molecule"
    COUNT = "count"
    BOND_EDIT = "bond_edit"
    ATOM_SET = "atom_set"
    UNKNOWN = "unknown"


class ComparatorKind(_DomainStrEnum):
    EXACT = "exact"
    CASE_INSENSITIVE = "case_insensitive"
    INTEGER_EQUAL = "integer_equal"
    FLOAT_TOLERANCE = "float_tolerance"
    ISOMERIC_GRAPH_EQUIVALENCE = "isomeric_graph_equivalence"
    FRAGMENT_GRAPH_EQUIVALENCE = "fragment_graph_equivalence"
    SET_EQUAL = "set_equal"
    CUSTOM = "custom"


class DependencyType(_DomainStrEnum):
    DERIVED_FROM = "derived_from"
    COUNT_OF = "count_of"
    DELTA_OF = "delta_of"
    MUST_EQUAL = "must_equal"
    MOLECULARLY_EQUIVALENT_TO = "molecularly_equivalent_to"
    FRAGMENT_GRAPH_EQUIVALENCE = "fragment_graph_equivalence"
    EDIT_PRODUCES = "edit_produces"
    CONSTRAINED_BY_INSTRUCTION = "constrained_by_instruction"
    ATTACHED_TO = "attached_to"
    REMOVED_FROM = "removed_from"


class ValueProvenance(_DomainStrEnum):
    REFERENCE = "reference"
    RULE = "rule"
    RDKIT = "rdkit"
    LLM = "llm"


class MutationCategory(_DomainStrEnum):
    """The six explicit kinds of editable reasoning/final-answer claims."""

    NUMERIC_INTEGER = "numeric_integer"
    NUMERIC_FLOAT = "numeric_float"
    ATOM_INDEX = "atom_index"
    ELEMENT_SYMBOL = "element_symbol"
    MOLECULAR_FRAGMENT = "molecular_fragment"
    SMILES_STRUCTURE = "smiles_structure"


class MutationTargetKind(_DomainStrEnum):
    NODE = "node"
    EDGE = "edge"


class CausalRole(_DomainStrEnum):
    """Whether a changed claim is the sampled error or its deterministic consequence."""

    ROOT_HALLUCINATION = "root_hallucination"
    PROPAGATED_ERROR = "propagated_error"


class Severity(_DomainStrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class ValidationStage(_DomainStrEnum):
    INPUT_RECORD = "input_record"
    REFERENCE_DAG = "reference_dag"
    RDKIT_STRUCTURE = "rdkit_structure"
    GRAPH_EDIT = "graph_edit"
    HALLUCINATION_SEMANTICS = "hallucination_semantics"
    RENDERER = "renderer"
