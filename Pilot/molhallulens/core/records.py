"""Minimal detector-input contract shared by reference and release stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DetectorInput:
    """Exactly what a hallucination detector is allowed to see."""

    indexed_smiles: str
    instruction: str
    reasoning_chain: str
    final_answer: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.indexed_smiles, "indexed_smiles"),
            (self.instruction, "instruction"),
            (self.reasoning_chain, "reasoning_chain"),
            (self.final_answer, "final_answer"),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"DetectorInput {name} must be non-empty text")

    @property
    def field_order(self) -> tuple[str, ...]:
        return ("indexed_smiles", "instruction", "reasoning_chain", "final_answer")
