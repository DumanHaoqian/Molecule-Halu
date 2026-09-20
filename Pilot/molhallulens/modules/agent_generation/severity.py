"""Inspectable construction severity, independent of root counts or victim output.

``product_tanimoto`` (also ``molecular.tanimoto``) is full-product Morgan radius-2,
2048-bit Tanimoto with chirality disabled. It is a similarity, not a probability;
fingerprint equality is not molecular equality. ``source_footprint_ratio`` is the
fraction of original source heavy atoms whose retention/attributes, source-source
bonds, or attached incoming-component signatures differ between clean and H.

Only ORIGINAL source maps establish atom correspondence. Generated map numbers
are discarded from rooted incoming-fragment signatures. The signatures describe
induced graphs, not standalone capped reagents. They are not an atom mapping or
an expensive MCS approximation. Attribute CIP changes are descriptor changes and
can reflect changed substituent priority, not necessarily geometric inversion.

These metrics stay separate: no root-weighted severity or weighted total score
is produced. Optional character/token fractions describe the supplied H CoT and
its labels, not per-atom truth and not counterfactual effects of individual roots.
"""

from __future__ import annotations

import difflib
import json
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .chemistry_tools import ChemistryToolError, inspect_source


def _mapped_mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.parseName = False
    return Chem.MolFromSmiles(inspect_source(smiles)["mapped_smiles"], params)


def _unmapped(mol: Chem.Mol) -> Chem.Mol:
    result = Chem.Mol(mol)
    for atom in result.GetAtoms():
        atom.SetAtomMapNum(0)
    return result


def _execution(execution: dict[str, Any], source_mapped: str) -> tuple[Chem.Mol, dict[str, Any]]:
    try:
        if not isinstance(execution, dict):
            raise ValueError("Expected an execution object")
        if "source_mapped_smiles" in execution and inspect_source(execution["source_mapped_smiles"])["mapped_smiles"] != source_mapped:
            raise ValueError("Execution source graph/maps differ from the supplied original source")
        mapped = execution["mapped_product_smiles"]
        # No implicit atom-map assignment for execution artifacts: correspondence
        # must have been recorded by the executor, not invented by this metric.
        params = Chem.SmilesParserParams()
        params.removeHs = False
        params.parseName = False
        mol = Chem.MolFromSmiles(mapped, params)
        if mol is None or any(atom.GetAtomMapNum() <= 0 for atom in mol.GetAtoms()):
            raise ValueError("Execution requires complete positive product maps")
        facts = inspect_source(mapped)
        if facts["canonical_smiles"] != inspect_source(execution["product_smiles"])["canonical_smiles"]:
            raise ValueError("Mapped and unmapped execution products disagree")
        return mol, facts
    except (KeyError, TypeError, ValueError) as error:
        raise ChemistryToolError("severity_invalid_execution", str(error)) from error


def _atom_attributes(atom: Chem.Atom) -> dict[str, Any]:
    return {
        "atomic_number": atom.GetAtomicNum(), "isotope": atom.GetIsotope(),
        "formal_charge": atom.GetFormalCharge(), "aromatic": atom.GetIsAromatic(),
        "total_hydrogens": atom.GetTotalNumHs(includeNeighbors=True),
        "cip_code": atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
    }


def _bond_attributes(bond: Chem.Bond) -> dict[str, Any]:
    return {"bond_type": str(bond.GetBondType()), "aromatic": bond.GetIsAromatic(),
            "stereo": str(bond.GetStereo())}


def _source_bonds(mol: Chem.Mol, source_maps: set[int]) -> dict[tuple[int, int], dict]:
    result = {}
    for bond in mol.GetBonds():
        maps = (bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())
        if set(maps) <= source_maps:
            result[tuple(sorted(maps))] = _bond_attributes(bond)
    return result


def _incoming_signatures(mol: Chem.Mol, source_maps: set[int]) -> dict[int, list[dict]]:
    """Describe each source-to-new attachment without treating new maps as IDs."""
    # Ordinary explicit H versus implicit H is notation, not an incoming group.
    # Isotopic hydrogens remain, so isotopic changes are not erased.
    mol = Chem.RemoveHs(mol)
    unseen = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() not in source_maps}
    plain = _unmapped(mol)
    # RDKit requires the entire molecule supplied to rootedAtAtom to have one
    # component, even when atomsToUse itself is connected. Isolate its original
    # connected component for serialization only; full-product metrics below
    # continue to include every spectator ion/component.
    component_indices: list[tuple[int, ...]] = []
    component_mols = Chem.GetMolFrags(plain, asMols=True, sanitizeFrags=False,
                                     fragsMolAtomMapping=component_indices)
    serialization_contexts = {}
    for fragment, global_indices in zip(component_mols, component_indices):
        local_indices = {index: local for local, index in enumerate(global_indices)}
        for index in global_indices:
            serialization_contexts[index] = (fragment, local_indices)
    signatures: dict[int, list[dict]] = {}
    while unseen:
        component, frontier = set(), {min(unseen)}
        while frontier:
            index = frontier.pop()
            if index in component:
                continue
            component.add(index)
            frontier.update(neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(index).GetNeighbors()
                            if neighbor.GetIdx() in unseen and neighbor.GetIdx() not in component)
        unseen -= component
        indices = sorted(component)
        attachment_edges = []
        for index in indices:
            for neighbor in mol.GetAtomWithIdx(index).GetNeighbors():
                if neighbor.GetAtomMapNum() in source_maps:
                    bond = mol.GetBondBetweenAtoms(index, neighbor.GetIdx())
                    attachment_edges.append((neighbor.GetAtomMapNum(), index, _bond_attributes(bond)))
        fragment, index_map = serialization_contexts[indices[0]]
        fragment_indices = [index_map[index] for index in indices]
        rooted = {
            index: Chem.MolFragmentToSmiles(fragment, atomsToUse=fragment_indices,
                                            rootedAtAtom=index_map[index], canonical=True, isomericSmiles=True)
            for _, index, _ in attachment_edges
        }
        # The source-anchor identifiers are real correspondence. Include every
        # boundary to distinguish incoming bridges, while never storing new maps.
        boundary = sorted([
            {"source_map": anchor, "bond": bond,
             "rooted_fragment_smiles": rooted[index]}
            for anchor, index, bond in attachment_edges
        ], key=lambda item: json.dumps(item, sort_keys=True))
        for anchor, index, bond in attachment_edges:
            signatures.setdefault(anchor, []).append({
                "bond": bond,
                "incoming_heavy_atoms": sum(mol.GetAtomWithIdx(i).GetAtomicNum() > 1 for i in component),
                "rooted_fragment_smiles": rooted[index],
                "source_attachments": boundary,
            })
    for values in signatures.values():
        values.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return signatures


def _text_metrics(clean_trace, h_trace, annotations, token_labels) -> dict[str, Any]:
    if h_trace is None:
        if clean_trace is not None or annotations is not None or token_labels is not None:
            raise ValueError("h_trace is required for text or annotation severity")
        return {}
    if not isinstance(h_trace, str) or (clean_trace is not None and not isinstance(clean_trace, str)):
        raise ValueError("Traces must be strings")
    length = len(h_trace)
    result = {"h_characters": length, "clean_characters": len(clean_trace) if clean_trace is not None else None,
              "annotated_error_characters": None, "annotated_error_character_fraction": None,
              "changed_h_characters": None, "changed_h_character_fraction": None,
              "deleted_clean_characters": None, "tokenizers": {}}
    changed_ranges = []
    if clean_trace is not None:
        opcodes = difflib.SequenceMatcher(a=clean_trace, b=h_trace, autojunk=False).get_opcodes()
        changed_ranges = [(j1, j2) for tag, _, _, j1, j2 in opcodes if tag != "equal" and j1 < j2]
        changed = sum(j2 - j1 for tag, _, _, j1, j2 in opcodes if tag != "equal")
        result.update(changed_h_characters=changed,
                      changed_h_character_fraction=changed / length if length else None,
                      deleted_clean_characters=sum(i2 - i1 for tag, i1, i2, _, _ in opcodes if tag == "delete"),
                      text_difference_policy="SequenceMatcher character insert/replace coverage in H; not semantic error labels")
    if annotations is not None:
        if not isinstance(annotations, list):
            raise ValueError("annotations must be a list")
        ranges = []
        for span in annotations:
            start, end = span.get("start"), span.get("end")
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= length:
                raise ValueError("Annotation offsets must be nonempty and inside h_trace")
            if "text" in span and span["text"] != h_trace[start:end]:
                raise ValueError("Annotation text differs from h_trace")
            ranges.append((start, end))
        covered, previous_end = 0, 0
        for start, end in sorted(ranges):
            covered += max(0, end - max(start, previous_end))
            previous_end = max(previous_end, end)
        result.update(annotated_error_characters=covered,
                      annotated_error_character_fraction=covered / length if length else None)
    if token_labels is not None:
        if not isinstance(token_labels, dict):
            raise ValueError("token_labels must map tokenizer names to standalone H token records")
        for name, records in token_labels.items():
            if not isinstance(name, str) or not isinstance(records, list):
                raise ValueError("Invalid tokenizer records")
            errors, changed_tokens = 0, 0
            for record in records:
                start, end = record.get("start"), record.get("end")
                if type(start) is not int or type(end) is not int or not 0 <= start <= end <= length:
                    raise ValueError("Token offsets must refer to the supplied standalone h_trace")
                labels = record.get("labels")
                if not isinstance(labels, list) or not labels or set(labels) - {"root_error", "propagated_error", "unchanged"}:
                    raise ValueError("Invalid token error labels")
                errors += bool(set(labels) & {"root_error", "propagated_error"})
                changed_tokens += any(start < high and end > low for low, high in changed_ranges)
            result["tokenizers"][name] = {"tokens": len(records), "error_tokens": errors,
                                           "error_token_fraction": errors / len(records) if records else None,
                                           "changed_tokens": changed_tokens if clean_trace is not None else None,
                                           "changed_token_fraction": changed_tokens / len(records) if records and clean_trace is not None else None,
                                           "changed_token_policy": "H tokens overlapping literal inserted/replaced characters; clean-only deletions counted separately as characters",
                                           "scope": "supplied standalone H token records; each token counted once"}
    return result


def measure_severity(
    source_smiles: str, clean_execution: dict[str, Any], h_execution: dict[str, Any], *,
    clean_trace: str | None = None, h_trace: str | None = None,
    annotations: list[dict] | None = None, token_labels: dict[str, list[dict]] | None = None,
) -> dict[str, Any]:
    """Compare a clean execution with H; never infer causality from root counts.

    Complete positive original source maps are the only atom correspondence.
    All components are used for molecular similarity. No execution, plan, trace,
    labels or mapping objects are modified. Missing text labels remain unknown.
    """
    source_mapped = inspect_source(source_smiles)["mapped_smiles"]
    source = _mapped_mol(source_mapped)
    clean, clean_facts = _execution(clean_execution, source_mapped)
    altered, h_facts = _execution(h_execution, source_mapped)
    source_maps = {atom.GetAtomMapNum() for atom in source.GetAtoms()}
    heavy_maps = {atom.GetAtomMapNum() for atom in source.GetAtoms() if atom.GetAtomicNum() > 1}
    clean_atoms = {atom.GetAtomMapNum(): _atom_attributes(atom) for atom in clean.GetAtoms()
                   if atom.GetAtomMapNum() in source_maps}
    h_atoms = {atom.GetAtomMapNum(): _atom_attributes(atom) for atom in altered.GetAtoms()
               if atom.GetAtomMapNum() in source_maps}
    changed_atoms = [{"atom_map": atom_map, "clean": clean_atoms.get(atom_map), "h": h_atoms.get(atom_map)}
                     for atom_map in sorted(source_maps) if clean_atoms.get(atom_map) != h_atoms.get(atom_map)]
    retention = sorted(clean_atoms.keys() ^ h_atoms.keys())
    cb, hb = _source_bonds(clean, source_maps), _source_bonds(altered, source_maps)
    changed_bonds = [{"atom_maps": list(pair), "clean": cb.get(pair), "h": hb.get(pair)}
                     for pair in sorted(cb.keys() | hb.keys()) if cb.get(pair) != hb.get(pair)]
    ca, ha = _incoming_signatures(clean, source_maps), _incoming_signatures(altered, source_maps)
    changed_attachments = [{"atom_map": atom_map, "clean": ca.get(atom_map, []), "h": ha.get(atom_map, [])}
                           for atom_map in sorted(ca.keys() | ha.keys()) if ca.get(atom_map, []) != ha.get(atom_map, [])]
    footprint = ({item["atom_map"] for item in changed_atoms + changed_attachments}
                 | {atom_map for item in changed_bonds for atom_map in item["atom_maps"]})
    heavy_footprint = sorted(footprint & heavy_maps)
    ratio = len(heavy_footprint) / len(heavy_maps) if heavy_maps else 0.0
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    tanimoto = float(DataStructs.TanimotoSimilarity(generator.GetFingerprint(Chem.RemoveHs(_unmapped(clean))),
                                                   generator.GetFingerprint(Chem.RemoveHs(_unmapped(altered)))))
    return {
        "version": "construction_severity_v1", "product_tanimoto": tanimoto,
        "source_footprint_ratio": ratio,
        "molecular": {
            "tanimoto": tanimoto, "fingerprint": {"family": "Morgan", "radius": 2, "bits": 2048, "include_chirality": False,
                                                       "scope": "complete products; all components; atom maps removed"},
            "isomeric_full_equivalent": clean_facts["canonical_smiles"] == h_facts["canonical_smiles"],
            "clean_heavy_atoms": clean_facts["heavy_atoms"], "h_heavy_atoms": h_facts["heavy_atoms"],
            "heavy_atom_difference": h_facts["heavy_atoms"] - clean_facts["heavy_atoms"],
            "clean_rings": clean_facts["rings"], "h_rings": h_facts["rings"],
            "ring_difference": h_facts["rings"] - clean_facts["rings"],
        },
        "source_edit": {
            "source_heavy_atoms": len(heavy_maps), "clean_retained_atom_maps": sorted(clean_atoms),
            "h_retained_atom_maps": sorted(h_atoms), "changed_retention_maps": retention,
            "changed_atom_maps": [item["atom_map"] for item in changed_atoms], "atom_changes": changed_atoms,
            "changed_bonds": changed_bonds,
            "changed_attachment_maps": [item["atom_map"] for item in changed_attachments],
            "attachment_changes": changed_attachments, "footprint_atom_maps": sorted(footprint),
            "footprint_heavy_atom_maps": heavy_footprint, "footprint_ratio": ratio,
            "attribute_note": "CIP changes can reflect changed ligand priority; generated map numbers are never matched across products",
        },
        "text": _text_metrics(clean_trace, h_trace, annotations, token_labels),
    }
