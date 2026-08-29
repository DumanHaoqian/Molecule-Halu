"""Split-local fragment, group, and product donor pools for T030."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import permutations
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from rdkit import Chem

from molhallulens.builders.origin_audit import OriginSplitAudit
from molhallulens.domain import (
    AtomReference,
    AtomReferenceNamespace,
    EditTruth,
    FragmentSpec,
)

if TYPE_CHECKING:
    from molhallulens.builders.split_manifest import VerifiedSplitManifest
    from molhallulens.validation.reference import OriginValidationInput


DONOR_POOL_FORMAT_VERSION = "split_local_donor_pool_v1"
DONOR_POOL_SCHEMA_VERSION = "donor_pool_schema_v1"
DEFAULT_DONOR_POOL_DIRECTORY = Path("HallucinationDataset/donor_pools")
_SPLIT_NAMES = ("train", "validation", "test")


class DonorPoolError(RuntimeError):
    """Structured fail-closed donor construction/import error."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if type(code) is not str or not code:
            raise ValueError("donor error code must be non-empty text")
        if type(detail) is not str or not detail:
            raise ValueError("donor error detail must be non-empty text")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise TypeError("donor error evidence must be a mapping or None")
        self.code = code
        self.detail = detail
        self.evidence = MappingProxyType(dict(evidence or {}))
        super().__init__(f"{code}: {detail}")


class DonorKind(StrEnum):
    FRAGMENT = "fragment"
    GROUP = "group"
    PRODUCT = "product"


def _verified_manifest_type() -> type[Any]:
    # Local import lets T030 remain isolated while T029 is authored in parallel.
    from molhallulens.builders.split_manifest import VerifiedSplitManifest

    return VerifiedSplitManifest


def _require_verified_manifest(manifest: object) -> Any:
    verified_type = _verified_manifest_type()
    if type(manifest) is not verified_type:
        raise TypeError("manifest must be an exact VerifiedSplitManifest")
    return manifest


def _split_value(value: object) -> str:
    raw = getattr(value, "value", value)
    if type(raw) is not str or raw not in _SPLIT_NAMES:
        raise ValueError("manifest split must be train, validation, or test")
    return raw


def _manifest_split(manifest: Any, anonymous_sample_id: str) -> str:
    try:
        return _split_value(manifest.split_for_origin(anonymous_sample_id))
    except Exception as error:
        raise DonorPoolError(
            "UNKNOWN_MANIFEST_ORIGIN",
            "origin is not present in the verified split manifest",
            evidence={"anonymous_sample_id": anonymous_sample_id},
        ) from error


def _manifest_ids(manifest: Any) -> tuple[str, ...]:
    try:
        ids = tuple(sorted(row.anonymous_sample_id for row in manifest.rows))
    except (AttributeError, TypeError) as error:
        raise TypeError(
            "VerifiedSplitManifest rows do not expose origin IDs"
        ) from error
    if len(ids) != len(set(ids)):
        raise DonorPoolError(
            "MANIFEST_IDENTITY_MISMATCH",
            "verified manifest contains duplicate origin IDs",
        )
    return ids


def _require_same_manifest_split(
    manifest: Any,
    left_origin_id: str,
    right_origin_id: str,
) -> str:
    left_split = _manifest_split(manifest, left_origin_id)
    right_split = _manifest_split(manifest, right_origin_id)
    try:
        manifest.require_same_split(left_origin_id, right_origin_id)
    except Exception as error:
        raise DonorPoolError(
            "CROSS_SPLIT_DONOR_EDGE",
            "verified manifest rejected a cross-split donor edge",
            evidence={
                "left_origin_id": left_origin_id,
                "right_origin_id": right_origin_id,
                "left_split": left_split,
                "right_split": right_split,
            },
        ) from error
    if left_split != right_split:
        raise DonorPoolError(
            "CROSS_SPLIT_DONOR_EDGE",
            "donor edge endpoints belong to different frozen splits",
            evidence={
                "left_origin_id": left_origin_id,
                "right_origin_id": right_origin_id,
                "left_split": left_split,
                "right_split": right_split,
            },
        )
    return left_split


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _heavy_bucket(value: int) -> str:
    if value == 1:
        return "1"
    if value <= 3:
        return "2_3"
    if value <= 7:
        return "4_7"
    if value <= 15:
        return "8_15"
    if value <= 30:
        return "16_30"
    if value <= 45:
        return "31_45"
    return "46_plus"


def _ring_bucket(value: int) -> str:
    if value == 0:
        return "0"
    if value <= 2:
        return "1_2"
    if value <= 4:
        return "3_4"
    return "5_plus"


def _heteroatom_bucket(value: int) -> str:
    if value <= 2:
        return "0_2"
    if value <= 5:
        return "3_5"
    return "6_plus"


@dataclass(frozen=True, slots=True)
class DonorEntry:
    donor_id: str
    donor_origin_id: str
    kind: DonorKind
    split: str
    canonical_smiles: str
    heavy_atom_count: int
    ring_count: int
    formal_charge: int
    heteroatom_counts: tuple[tuple[int, int], ...]
    attachment_atomic_numbers: tuple[int, ...]
    boundary_bond_types: tuple[str, ...]
    difficulty_bucket: str

    def __post_init__(self) -> None:
        if type(self.kind) is not DonorKind:
            raise TypeError("donor kind must be DonorKind")
        expected_id = f"{self.kind.value}:{self.donor_origin_id}"
        if self.donor_id != expected_id:
            raise ValueError("donor ID must bind kind and origin")
        if (
            type(self.donor_origin_id) is not str
            or not self.donor_origin_id
            or type(self.canonical_smiles) is not str
            or not self.canonical_smiles
        ):
            raise ValueError("donor origin and canonical SMILES must be non-empty")
        if self.split not in _SPLIT_NAMES:
            raise ValueError("donor split is invalid")
        if type(self.heavy_atom_count) is not int or self.heavy_atom_count <= 0:
            raise ValueError("donor heavy atom count must be positive")
        if type(self.ring_count) is not int or self.ring_count < 0:
            raise ValueError("donor ring count must be non-negative")
        if type(self.formal_charge) is not int:
            raise TypeError("donor formal charge must be int")
        heteroatoms = tuple(sorted(self.heteroatom_counts))
        attachments = tuple(sorted(self.attachment_atomic_numbers))
        boundary_types = tuple(sorted(self.boundary_bond_types))
        if any(
            type(atomic_number) is not int
            or atomic_number <= 0
            or type(count) is not int
            or count <= 0
            for atomic_number, count in heteroatoms
        ):
            raise ValueError("heteroatom counts must be positive integer pairs")
        if len({item[0] for item in heteroatoms}) != len(heteroatoms):
            raise ValueError("heteroatom atomic numbers must be unique")
        if any(type(item) is not int or item <= 0 for item in attachments):
            raise ValueError("attachment atomic numbers must be positive")
        if any(type(item) is not str or not item for item in boundary_types):
            raise ValueError("boundary bond types must be non-empty text")
        if self.kind is DonorKind.PRODUCT:
            if attachments or boundary_types:
                raise ValueError("product donors cannot carry fragment attachment data")
        elif not attachments or len(attachments) != len(boundary_types):
            raise ValueError(
                "fragment/group donors require one bond type per attachment"
            )
        if type(self.difficulty_bucket) is not str or not self.difficulty_bucket:
            raise ValueError("difficulty bucket must be non-empty text")
        object.__setattr__(self, "heteroatom_counts", heteroatoms)
        object.__setattr__(self, "attachment_atomic_numbers", attachments)
        object.__setattr__(self, "boundary_bond_types", boundary_types)

    @property
    def attachment_bucket(self) -> str:
        if self.kind is DonorKind.PRODUCT:
            return "none"
        atoms = ",".join(str(item) for item in self.attachment_atomic_numbers)
        bonds = ",".join(self.boundary_bond_types)
        return f"atoms={atoms};bonds={bonds}"

    @property
    def descriptor_bucket(self) -> str:
        charge = (
            "negative"
            if self.formal_charge < 0
            else "positive"
            if self.formal_charge > 0
            else "neutral"
        )
        heteroatom_total = sum(count for _, count in self.heteroatom_counts)
        return ";".join(
            (
                f"heavy={_heavy_bucket(self.heavy_atom_count)}",
                f"rings={_ring_bucket(self.ring_count)}",
                f"charge={charge}",
                f"hetero={_heteroatom_bucket(heteroatom_total)}",
            )
        )

    @property
    def bucket_id(self) -> str:
        return "|".join(
            (
                self.kind.value,
                self.attachment_bucket,
                self.descriptor_bucket,
                f"difficulty={self.difficulty_bucket}",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "donor_id": self.donor_id,
            "donor_origin_id": self.donor_origin_id,
            "kind": self.kind.value,
            "split": self.split,
            "canonical_smiles": self.canonical_smiles,
            "heavy_atom_count": self.heavy_atom_count,
            "ring_count": self.ring_count,
            "formal_charge": self.formal_charge,
            "heteroatom_counts": [list(item) for item in self.heteroatom_counts],
            "attachment_atomic_numbers": list(self.attachment_atomic_numbers),
            "boundary_bond_types": list(self.boundary_bond_types),
            "attachment_bucket": self.attachment_bucket,
            "descriptor_bucket": self.descriptor_bucket,
            "difficulty_bucket": self.difficulty_bucket,
            "bucket_id": self.bucket_id,
        }


@dataclass(frozen=True, slots=True)
class DonorEdge:
    edge_id: str
    split: str
    recipient_origin_id: str
    donor_id: str
    donor_origin_id: str
    kind: DonorKind
    bucket_id: str

    def __post_init__(self) -> None:
        expected_id = f"{self.recipient_origin_id}->{self.donor_id}"
        if self.edge_id != expected_id:
            raise ValueError("donor edge ID must bind recipient and donor")
        if self.split not in _SPLIT_NAMES:
            raise ValueError("donor edge split is invalid")
        for value, name in (
            (self.recipient_origin_id, "recipient_origin_id"),
            (self.donor_id, "donor_id"),
            (self.donor_origin_id, "donor_origin_id"),
            (self.bucket_id, "bucket_id"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty text")
        if self.recipient_origin_id == self.donor_origin_id:
            raise ValueError("self donor edges are forbidden")
        if type(self.kind) is not DonorKind:
            raise TypeError("donor edge kind must be DonorKind")

    def to_dict(self) -> dict[str, str]:
        return {
            "edge_id": self.edge_id,
            "split": self.split,
            "recipient_origin_id": self.recipient_origin_id,
            "donor_id": self.donor_id,
            "donor_origin_id": self.donor_origin_id,
            "kind": self.kind.value,
            "bucket_id": self.bucket_id,
        }


@dataclass(frozen=True, slots=True)
class SplitBoundDonorQuery:
    manifest_sha256: str
    split: str
    recipient_origin_id: str
    kind: DonorKind
    attachment_bucket: str | None = None
    descriptor_bucket: str | None = None
    difficulty_bucket: str | None = None
    limit: int | None = None

    def __post_init__(self) -> None:
        if type(self.manifest_sha256) is not str or len(self.manifest_sha256) != 64:
            raise ValueError("query manifest identity must be SHA256")
        if self.split not in _SPLIT_NAMES:
            raise ValueError("query split is invalid")
        if type(self.recipient_origin_id) is not str or not self.recipient_origin_id:
            raise ValueError("query recipient must be non-empty text")
        if type(self.kind) is not DonorKind:
            raise TypeError("query kind must be DonorKind")
        for value, name in (
            (self.attachment_bucket, "attachment_bucket"),
            (self.descriptor_bucket, "descriptor_bucket"),
            (self.difficulty_bucket, "difficulty_bucket"),
        ):
            if value is not None and (type(value) is not str or not value):
                raise ValueError(f"query {name} must be non-empty text or None")
        if self.limit is not None and (type(self.limit) is not int or self.limit <= 0):
            raise ValueError("query limit must be positive or None")


def _derive_edges(
    manifest: Any,
    split: str,
    donors: tuple[DonorEntry, ...],
) -> tuple[DonorEdge, ...]:
    buckets: dict[str, list[DonorEntry]] = defaultdict(list)
    for donor in donors:
        buckets[donor.bucket_id].append(donor)
    edges = []
    for bucket_id, bucket_donors in sorted(buckets.items()):
        for recipient, donor in permutations(
            sorted(bucket_donors, key=lambda item: item.donor_id), 2
        ):
            if recipient.donor_origin_id == donor.donor_origin_id:
                continue
            edge_split = _require_same_manifest_split(
                manifest,
                recipient.donor_origin_id,
                donor.donor_origin_id,
            )
            if edge_split != split:
                raise DonorPoolError(
                    "CROSS_SPLIT_DONOR_EDGE",
                    "derived donor edge does not belong to the pool split",
                )
            edges.append(
                DonorEdge(
                    edge_id=f"{recipient.donor_origin_id}->{donor.donor_id}",
                    split=split,
                    recipient_origin_id=recipient.donor_origin_id,
                    donor_id=donor.donor_id,
                    donor_origin_id=donor.donor_origin_id,
                    kind=donor.kind,
                    bucket_id=bucket_id,
                )
            )
    return tuple(sorted(edges, key=lambda item: item.edge_id))


@dataclass(frozen=True, slots=True, init=False)
class SplitDonorPool:
    """One immutable donor pool bound to one verified manifest and split."""

    manifest: Any = field(repr=False, compare=False)
    manifest_sha256: str
    dataset_version: str
    split_seed: int
    split: str
    donors: tuple[DonorEntry, ...]
    edges: tuple[DonorEdge, ...]
    format_version: str
    schema_version: str

    def __init__(
        self,
        *,
        manifest: VerifiedSplitManifest,
        split: str,
        donors: Iterable[DonorEntry],
    ) -> None:
        verified = _require_verified_manifest(manifest)
        if split not in _SPLIT_NAMES:
            raise ValueError("pool split is invalid")
        try:
            ordered = tuple(sorted(donors, key=lambda item: item.donor_id))
        except (AttributeError, TypeError) as error:
            raise TypeError("donors must be iterable DonorEntry values") from error
        if any(type(item) is not DonorEntry for item in ordered):
            raise TypeError("donors must contain DonorEntry values")
        donor_ids = tuple(item.donor_id for item in ordered)
        if len(donor_ids) != len(set(donor_ids)):
            raise ValueError("donor IDs must be unique")
        for donor in ordered:
            manifest_split = _manifest_split(verified, donor.donor_origin_id)
            if donor.split != split or manifest_split != split:
                raise DonorPoolError(
                    "CROSS_SPLIT_DONOR",
                    "donor does not belong to the pool's frozen split",
                    evidence={
                        "donor_origin_id": donor.donor_origin_id,
                        "donor_split": donor.split,
                        "manifest_split": manifest_split,
                        "pool_split": split,
                    },
                )
        edges = _derive_edges(verified, split, ordered)
        object.__setattr__(self, "manifest", verified)
        object.__setattr__(self, "manifest_sha256", verified.manifest_sha256)
        object.__setattr__(self, "dataset_version", verified.dataset_version)
        object.__setattr__(self, "split_seed", verified.split_seed)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "donors", ordered)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "format_version", DONOR_POOL_FORMAT_VERSION)
        object.__setattr__(self, "schema_version", DONOR_POOL_SCHEMA_VERSION)

    def query(self, query: SplitBoundDonorQuery) -> tuple[DonorEntry, ...]:
        if type(query) is not SplitBoundDonorQuery:
            raise TypeError("query must be SplitBoundDonorQuery")
        if query.manifest_sha256 != self.manifest_sha256 or query.split != self.split:
            raise DonorPoolError(
                "QUERY_MANIFEST_SPLIT_MISMATCH",
                "query is not bound to this manifest/split pool",
            )
        if _manifest_split(self.manifest, query.recipient_origin_id) != self.split:
            raise DonorPoolError(
                "QUERY_RECIPIENT_SPLIT_MISMATCH",
                "query recipient does not belong to this pool split",
            )
        donor_by_id = {item.donor_id: item for item in self.donors}
        values = []
        for edge in self.edges:
            if (
                edge.recipient_origin_id != query.recipient_origin_id
                or edge.kind is not query.kind
            ):
                continue
            donor = donor_by_id[edge.donor_id]
            if (
                query.attachment_bucket is not None
                and donor.attachment_bucket != query.attachment_bucket
            ):
                continue
            if (
                query.descriptor_bucket is not None
                and donor.descriptor_bucket != query.descriptor_bucket
            ):
                continue
            if (
                query.difficulty_bucket is not None
                and donor.difficulty_bucket != query.difficulty_bucket
            ):
                continue
            values.append(donor)
        ordered = tuple(sorted(values, key=lambda item: item.donor_id))
        return ordered if query.limit is None else ordered[: query.limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "split_manifest_sha256": self.manifest_sha256,
            "split_seed": self.split_seed,
            "split": self.split,
            "summary": {
                "donor_count": len(self.donors),
                "edge_count": len(self.edges),
                "bucket_count": len({item.bucket_id for item in self.donors}),
                "donors_by_kind": {
                    kind.value: sum(item.kind is kind for item in self.donors)
                    for kind in DonorKind
                },
            },
            "donors": [item.to_dict() for item in self.donors],
            "edges": [item.to_dict() for item in self.edges],
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())


def _source_atomic_numbers(truth: EditTruth) -> Mapping[AtomReference, int]:
    molecule = Chem.MolFromSmiles(truth.source_smiles)
    if molecule is None:
        raise DonorPoolError(
            "SOURCE_PARSE_FAILED",
            "T015-passing source could not be parsed for donor attachments",
        )
    values = {
        AtomReference(AtomReferenceNamespace.SOURCE_MAP, atom.GetAtomMapNum()): (
            atom.GetAtomicNum()
        )
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    return MappingProxyType(values)


def _attachment_atomic_numbers(
    truth: EditTruth,
    fragment: FragmentSpec,
) -> tuple[int, ...]:
    source_numbers = _source_atomic_numbers(truth)
    product_numbers = {item.reference: item.atomic_number for item in truth.added_atoms}
    values = []
    for reference in fragment.attachment_atoms:
        if reference.namespace is AtomReferenceNamespace.SOURCE_MAP:
            lookup = source_numbers
        else:
            lookup = product_numbers
        try:
            values.append(lookup[reference])
        except KeyError as error:
            raise DonorPoolError(
                "ATTACHMENT_IDENTITY_MISMATCH",
                "fragment attachment atom is absent from T013 identity maps",
            ) from error
    return tuple(sorted(values))


def _donor_entry(
    *,
    truth: EditTruth,
    kind: DonorKind,
    split: str,
    difficulty_bucket: str,
    fragment: FragmentSpec | None,
) -> DonorEntry:
    descriptors = (
        truth.product_descriptors if fragment is None else fragment.descriptors
    )
    canonical_smiles = (
        truth.canonical_gt_smiles if fragment is None else fragment.canonical_smiles
    )
    return DonorEntry(
        donor_id=f"{kind.value}:{truth.anonymous_sample_id}",
        donor_origin_id=truth.anonymous_sample_id,
        kind=kind,
        split=split,
        canonical_smiles=canonical_smiles,
        heavy_atom_count=descriptors.heavy_atom_count,
        ring_count=descriptors.ring_count,
        formal_charge=descriptors.formal_charge,
        heteroatom_counts=descriptors.heteroatom_counts,
        attachment_atomic_numbers=(
            () if fragment is None else _attachment_atomic_numbers(truth, fragment)
        ),
        boundary_bond_types=(
            ()
            if fragment is None
            else tuple(bond.bond_type.value for bond in fragment.boundary_bonds)
        ),
        difficulty_bucket=difficulty_bucket,
    )


def build_split_local_donor_pools(
    manifest: VerifiedSplitManifest,
    *,
    items: Iterable[OriginValidationInput],
    audit: OriginSplitAudit,
) -> tuple[SplitDonorPool, ...]:
    """Build three pools without invoking or mutating any split solver/manifest."""

    from molhallulens.validation.reference import OriginValidationInput

    verified = _require_verified_manifest(manifest)
    if type(audit) is not OriginSplitAudit:
        raise TypeError("audit must be OriginSplitAudit")
    values = tuple(items)
    if any(type(item) is not OriginValidationInput for item in values):
        raise TypeError("items must contain OriginValidationInput values")
    item_ids = tuple(item.edit_truth.anonymous_sample_id for item in values)
    audit_ids = tuple(record.anonymous_sample_id for record in audit.records)
    manifest_ids = _manifest_ids(verified)
    if (
        len(item_ids) != len(set(item_ids))
        or set(item_ids) != set(audit_ids)
        or set(item_ids) != set(manifest_ids)
    ):
        raise DonorPoolError(
            "DONOR_INPUT_IDENTITY_MISMATCH",
            "chemistry inputs, T026 audit, and T029 manifest must cover the same origins",
        )
    if audit.dataset_version != verified.dataset_version:
        raise DonorPoolError(
            "MANIFEST_DATASET_MISMATCH",
            "T026 audit and T029 manifest dataset versions differ",
        )
    audit_by_id = {record.anonymous_sample_id: record for record in audit.records}
    donors_by_split: dict[str, list[DonorEntry]] = {split: [] for split in _SPLIT_NAMES}
    for item in sorted(values, key=lambda value: value.edit_truth.anonymous_sample_id):
        truth = item.edit_truth
        split = _manifest_split(verified, truth.anonymous_sample_id)
        difficulty = audit_by_id[truth.anonymous_sample_id].tanimoto_quantile_bin
        donors_by_split[split].append(
            _donor_entry(
                truth=truth,
                kind=DonorKind.PRODUCT,
                split=split,
                difficulty_bucket=difficulty,
                fragment=None,
            )
        )
        if truth.add_fragment is not None:
            donors_by_split[split].append(
                _donor_entry(
                    truth=truth,
                    kind=DonorKind.FRAGMENT,
                    split=split,
                    difficulty_bucket=difficulty,
                    fragment=truth.add_fragment,
                )
            )
        if truth.remove_fragment is not None:
            donors_by_split[split].append(
                _donor_entry(
                    truth=truth,
                    kind=DonorKind.GROUP,
                    split=split,
                    difficulty_bucket=difficulty,
                    fragment=truth.remove_fragment,
                )
            )
    return tuple(
        SplitDonorPool(
            manifest=verified,
            split=split,
            donors=donors_by_split[split],
        )
        for split in _SPLIT_NAMES
    )


def _parse_donor(payload: object) -> DonorEntry:
    if not isinstance(payload, Mapping):
        raise DonorPoolError("DONOR_ARTIFACT_SCHEMA", "donor row must be an object")
    try:
        donor = DonorEntry(
            donor_id=payload["donor_id"],
            donor_origin_id=payload["donor_origin_id"],
            kind=DonorKind(payload["kind"]),
            split=payload["split"],
            canonical_smiles=payload["canonical_smiles"],
            heavy_atom_count=payload["heavy_atom_count"],
            ring_count=payload["ring_count"],
            formal_charge=payload["formal_charge"],
            heteroatom_counts=tuple(
                tuple(item) for item in payload["heteroatom_counts"]
            ),
            attachment_atomic_numbers=tuple(payload["attachment_atomic_numbers"]),
            boundary_bond_types=tuple(payload["boundary_bond_types"]),
            difficulty_bucket=payload["difficulty_bucket"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DonorPoolError(
            "DONOR_ARTIFACT_SCHEMA", "donor row failed typed validation"
        ) from error
    if dict(payload) != donor.to_dict():
        raise DonorPoolError(
            "DONOR_ARTIFACT_SCHEMA",
            "donor row contains unknown, stale, or derived-field values",
        )
    return donor


def _validate_serialized_edges(
    payloads: object,
    *,
    manifest: Any,
    split: str,
    donors: tuple[DonorEntry, ...],
) -> None:
    if not isinstance(payloads, list):
        raise DonorPoolError("DONOR_ARTIFACT_SCHEMA", "edges must be an array")
    donor_by_id = {item.donor_id: item for item in donors}
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise DonorPoolError(
                "DONOR_ARTIFACT_SCHEMA", "donor edge must be an object"
            )
        try:
            recipient = payload["recipient_origin_id"]
            donor = donor_by_id[payload["donor_id"]]
        except KeyError as error:
            raise DonorPoolError(
                "UNKNOWN_DONOR_EDGE_ENDPOINT",
                "edge refers to an unknown donor or missing endpoint",
            ) from error
        edge_split = _require_same_manifest_split(
            manifest, recipient, donor.donor_origin_id
        )
        if edge_split != split:
            raise DonorPoolError(
                "CROSS_SPLIT_DONOR_EDGE",
                "serialized donor edge is outside the artifact split",
            )


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def load_split_donor_pool(
    path: Path,
    *,
    manifest: VerifiedSplitManifest,
    expected_split: str,
) -> SplitDonorPool:
    """Strictly import one pool; mismatch never triggers split recomputation."""

    verified = _require_verified_manifest(manifest)
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if expected_split not in _SPLIT_NAMES:
        raise ValueError("expected_split is invalid")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        _DuplicateJsonKeyError,
    ) as error:
        raise DonorPoolError(
            "DONOR_ARTIFACT_PARSE", "donor artifact is not strict JSON"
        ) from error
    if not isinstance(payload, Mapping):
        raise DonorPoolError(
            "DONOR_ARTIFACT_SCHEMA", "donor artifact root must be an object"
        )
    header = {
        "format_version": DONOR_POOL_FORMAT_VERSION,
        "schema_version": DONOR_POOL_SCHEMA_VERSION,
        "dataset_version": verified.dataset_version,
        "split_manifest_sha256": verified.manifest_sha256,
        "split_seed": verified.split_seed,
        "split": expected_split,
    }
    if any(payload.get(key) != value for key, value in header.items()):
        raise DonorPoolError(
            "DONOR_MANIFEST_BINDING_MISMATCH",
            "artifact header differs from the verified manifest/split identity",
        )
    donor_payloads = payload.get("donors")
    if not isinstance(donor_payloads, list):
        raise DonorPoolError("DONOR_ARTIFACT_SCHEMA", "donors must be an array")
    donors = tuple(_parse_donor(item) for item in donor_payloads)
    _validate_serialized_edges(
        payload.get("edges"),
        manifest=verified,
        split=expected_split,
        donors=donors,
    )
    pool = SplitDonorPool(
        manifest=verified,
        split=expected_split,
        donors=donors,
    )
    if dict(payload) != pool.to_dict():
        raise DonorPoolError(
            "DONOR_ARTIFACT_CONTENT_MISMATCH",
            "artifact content is not the deterministic pool for its donor rows",
        )
    return pool


def write_split_donor_pools(
    pools: Iterable[SplitDonorPool],
    *,
    output_directory: Path = DEFAULT_DONOR_POOL_DIRECTORY,
) -> None:
    values = tuple(pools)
    if any(type(item) is not SplitDonorPool for item in values):
        raise TypeError("pools must contain SplitDonorPool values")
    if {item.split for item in values} != set(_SPLIT_NAMES) or len(values) != 3:
        raise ValueError("writer requires exactly one pool per frozen split")
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    output_directory.mkdir(parents=True, exist_ok=True)
    for pool in sorted(values, key=lambda item: item.split):
        (output_directory / f"{pool.split}.json").write_bytes(pool.to_json_bytes())


__all__ = [
    "DEFAULT_DONOR_POOL_DIRECTORY",
    "DONOR_POOL_FORMAT_VERSION",
    "DONOR_POOL_SCHEMA_VERSION",
    "DonorEdge",
    "DonorEntry",
    "DonorKind",
    "DonorPoolError",
    "SplitBoundDonorQuery",
    "SplitDonorPool",
    "build_split_local_donor_pools",
    "load_split_donor_pool",
    "write_split_donor_pools",
]
