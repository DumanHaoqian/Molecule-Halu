"""Stable enum vocabulary for MolHalluLens domain objects."""

from __future__ import annotations

from enum import EnumType, IntEnum, StrEnum
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
        _SEALED_ENUM_MEMBER_IDS.update(id(member) for member in enum_class.__members__.values())
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


class _DomainIntEnum(_FrozenEnumMixin, IntEnum, metaclass=_FrozenEnumType):
    pass


def is_domain_enum_member(value: object) -> bool:
    """Return whether ``value`` is a sealed member created by this module's enum types."""

    return id(value) in _SEALED_ENUM_MEMBER_IDS


class TaskFamily(_DomainStrEnum):
    MOLECULE_EDITING = "mol_edit"
    MOLECULAR_OPTIMIZATION = "mol_opt"
    MOLECULE_UNDERSTANDING = "mol_und"
    REACTION_PREDICTION = "rxn_pred"


class EditingSubtask(_DomainStrEnum):
    ADD = "add"
    DELETE = "delete"
    SUBSTITUTE = "substitute"


class EditKind(_DomainStrEnum):
    ADDITION = "addition"
    DELETION = "deletion"
    SUBSTITUTION = "substitution"


class OperationSubtype(_DomainStrEnum):
    STANDARD = "standard"
    DEPROTECTION = "deprotection"
    DELETE_WITH_REPLACEMENT = "delete_with_replacement"


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
    EDIT_PRODUCES = "edit_produces"
    CONSTRAINED_BY_INSTRUCTION = "constrained_by_instruction"
    ATTACHED_TO = "attached_to"
    REMOVED_FROM = "removed_from"


class ValueProvenance(_DomainStrEnum):
    REFERENCE = "reference"
    RULE = "rule"
    RDKIT = "rdkit"
    LLM = "llm"
    PROPAGATED = "propagated"


class PropagationPolicy(_DomainStrEnum):
    STOP = "local"
    LOCAL = "local"
    PARTIAL = "partial"
    FULL_CF = "full_cf"
    TERMINAL = "terminal"

    @property
    def dataset_name(self) -> str:
        return {
            PropagationPolicy.STOP: "LOCAL",
            PropagationPolicy.PARTIAL: "PARTIAL",
            PropagationPolicy.FULL_CF: "FULL_CF",
            PropagationPolicy.TERMINAL: "TERMINAL",
        }[self]

    @classmethod
    def from_dataset_name(cls, value: str) -> PropagationPolicy:
        if type(value) is not str:
            raise TypeError("propagation policy dataset name must be a string")
        try:
            return {
                "LOCAL": cls.STOP,
                "PARTIAL": cls.PARTIAL,
                "FULL_CF": cls.FULL_CF,
                "TERMINAL": cls.TERMINAL,
            }[value]
        except KeyError as error:
            raise ValueError(f"Unknown propagation policy {value!r}") from error


class CandidateSourceType(_DomainStrEnum):
    RULE = "RULE"
    RDKIT = "RDKIT"
    LLM = "LLM"
    HYBRID = "HYBRID"


class CausalRole(_DomainStrEnum):
    ROOT = "ROOT"
    PROPAGATED_FALSE = "PROPAGATED_FALSE"
    PROPAGATED_CONDITIONAL = "PROPAGATED_CONDITIONAL"
    TERMINAL = "TERMINAL"


class HallucinationType(_DomainIntEnum):
    CONTRADICTION = 0
    UNSUPPORTED = 1
    REASONING_ERROR = 2
    INVALID_CHEMISTRY = 3
    CONSTRAINT_VIOLATION = 4
    FORMAT_ERROR = 5
    OMISSION = 6
    UNVERIFIABLE = 7


class EditErrorSubtype(_DomainStrEnum):
    ANCHOR_GROUNDING = "E01"
    REMOVE_OR_LEAVING_GROUP_IDENTIFICATION = "E02"
    ADD_FRAGMENT_IDENTIFICATION = "E03"
    ATTACHMENT_OR_BOND_EDIT = "E04"
    PRODUCT_CONSTRUCTION = "E05"
    HEAVY_ATOM_COUNT = "E06"
    HEAVY_ATOM_ARITHMETIC = "E07"
    RING_COUNT = "E08"
    RING_ARITHMETIC = "E09"
    CHEMICAL_VALIDITY = "E10"
    INSTRUCTION_CONSTRAINT = "E11"
    FINAL_ANSWER_IDENTITY = "E12"
    INTERNAL_INCONSISTENCY = "E13"
    FORMAT_SCHEMA = "E14"
    UNSUPPORTED_NATURAL_CLAIM = "E15"


class EvidenceRelation(_DomainStrEnum):
    CONTRADICTS_SOURCE = "CONTRADICTS_SOURCE"
    CONTRADICTS_INSTRUCTION = "CONTRADICTS_INSTRUCTION"
    CONTRADICTS_REFERENCE_STATE = "CONTRADICTS_REFERENCE_STATE"
    UNSUPPORTED_BY_EVIDENCE = "UNSUPPORTED_BY_EVIDENCE"
    INTERNAL_INCONSISTENCY = "INTERNAL_INCONSISTENCY"


class MutationTargetKind(_DomainStrEnum):
    NODE = "node"
    EDGE = "edge"


class VariantLabel(_DomainStrEnum):
    HALLUCINATED = "H"
    FAITHFUL = "N"


class SegmentKind(_DomainStrEnum):
    SOURCE = "source"
    INSTRUCTION = "instruction"
    REASONING = "reasoning"
    FINAL_ANSWER = "final_answer"
    SPECIAL = "special"
    PADDING = "padding"


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
    PROPAGATION = "propagation"
    RENDERER = "renderer"
    TOKEN_ALIGNMENT = "token_alignment"
    BUNDLE_INTEGRITY = "bundle_integrity"


class BondTypeName(_DomainStrEnum):
    SINGLE = "SINGLE"
    DOUBLE = "DOUBLE"
    TRIPLE = "TRIPLE"
    AROMATIC = "AROMATIC"
