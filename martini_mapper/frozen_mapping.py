"""Strict local Mapper with immutable pre-reaction beads.

The legacy Mapper was written for a complete molecule and may regroup atoms
that did not react.  This module adds a separate, opt-in path for crosslinking:

* every old bead containing a reaction atom is editable;
* every other old bead is copied byte-for-byte (membership and bead type);
* frozen atoms may be present in the local context, but are never committed;
* a candidate bead crossing the boundary is trimmed at the boundary;
* one-atom beads are valid after trimming, as required by the project rule.

Only the small local context is passed through the expensive Mapper sectioning
algorithm.  The default full-molecule API is intentionally unchanged.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import io
import time
import warnings
from collections import defaultdict, deque
from typing import Any, Iterable

from rdkit import Chem

from .algorithm import map_martini_beads
from .mapping_scheme import map_molecule, parse_smiles
from .martini_3_dictionary import get_m3_dict
from .setup_mapping import connectivity_matrix, get_atom_properties


class FrozenMappingError(RuntimeError):
    """Strict local mapping could not produce a valid frozen partition."""


def _atom_set(group: dict[str, Any]) -> frozenset[int]:
    return frozenset(int(atom) for atom in group["atoms"])


def _normalize_previous_groups(payload: Any) -> list[dict[str, Any]]:
    """Accept a group list or common JSON wrapper objects used by this project."""
    if isinstance(payload, dict):
        if "groups" in payload:
            payload = payload["groups"]
        elif "pre_groups" in payload:
            payload = payload["pre_groups"]
    if not isinstance(payload, list) or not payload:
        raise FrozenMappingError(
            "previous mapping must be a non-empty group list or contain groups/pre_groups"
        )
    groups = []
    for index, group in enumerate(payload):
        if not isinstance(group, dict) or "atoms" not in group or "type" not in group:
            raise FrozenMappingError(f"invalid previous mapping group at index {index}")
        atoms = [int(atom) for atom in group["atoms"]]
        if not atoms:
            raise FrozenMappingError(f"previous mapping group {index} is empty")
        normalized = dict(group)
        normalized["atoms"] = atoms
        normalized["type"] = str(group["type"])
        groups.append(normalized)
    return groups


def _validate_partition(groups: list[dict[str, Any]], expected_atoms: set[int]) -> None:
    flat = [int(atom) for group in groups for atom in group["atoms"]]
    if len(flat) != len(set(flat)):
        raise FrozenMappingError("mapping assigns at least one atom more than once")
    if set(flat) != expected_atoms:
        missing = sorted(expected_atoms - set(flat))[:10]
        extra = sorted(set(flat) - expected_atoms)[:10]
        raise FrozenMappingError(
            f"mapping is not an exact atom partition; missing={missing}, extra={extra}"
        )


def _adapter_tokens(smiles: str) -> list[str]:
    """Represent large ring labels as integer tokens instead of invalid bare %100."""
    tokens = parse_smiles(smiles)
    positions: dict[int, list[int]] = defaultdict(list)
    for index, token in enumerate(tokens):
        if token.isdigit():
            positions[int(token)].append(index)
    next_number = max(positions, default=0) + 1
    for indices in positions.values():
        if len(indices) % 2:
            raise FrozenMappingError("local SMILES contains an unpaired ring label")
        for offset in range(2, len(indices), 2):
            tokens[indices[offset]] = tokens[indices[offset + 1]] = str(next_number)
            next_number += 1
    return tokens


def _bead_graph(
    mol: Chem.Mol,
    previous_groups: list[dict[str, Any]],
) -> tuple[dict[int, set[int]], dict[int, int]]:
    group_by_atom: dict[int, int] = {}
    for group_index, group in enumerate(previous_groups):
        for atom_id in group["atoms"]:
            group_by_atom[int(atom_id)] = group_index
    graph = {index: set() for index in range(len(previous_groups))}
    for bond in mol.GetBonds():
        left = int(bond.GetBeginAtom().GetAtomMapNum())
        right = int(bond.GetEndAtom().GetAtomMapNum())
        left_group, right_group = group_by_atom[left], group_by_atom[right]
        if left_group != right_group:
            graph[left_group].add(right_group)
            graph[right_group].add(left_group)
    return graph, group_by_atom


def _select_context_atoms(
    mol: Chem.Mol,
    previous_groups: list[dict[str, Any]],
    editable_group_indices: set[int],
    context_layers: int,
    group_graph: dict[int, set[int]],
    group_by_atom: dict[int, int],
) -> set[int]:
    """Add read-only neighboring beads and close small/fused rings for context."""
    distance = {index: 0 for index in editable_group_indices}
    queue = deque(editable_group_indices)
    while queue:
        current = queue.popleft()
        if distance[current] >= context_layers:
            continue
        for neighbor in group_graph[current]:
            if neighbor not in distance:
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    context_groups = set(distance)

    # The Mapper handles rings before non-ring sections.  If a selected bead
    # touches a small/fused ring, include the complete ring as read-only context
    # so the fragment cut does not change ring perception near the reaction.
    index_to_atom_id = {
        atom.GetIdx(): int(atom.GetAtomMapNum()) for atom in mol.GetAtoms()
    }
    changed = True
    while changed:
        changed = False
        context_atoms = set().union(
            *(_atom_set(previous_groups[index]) for index in context_groups)
        )
        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) > 8:
                continue
            ring_atom_ids = {index_to_atom_id[index] for index in ring}
            if ring_atom_ids & context_atoms:
                additions = {
                    group_by_atom[atom_id] for atom_id in ring_atom_ids
                } - context_groups
                if additions:
                    context_groups.update(additions)
                    changed = True
    return set().union(*(_atom_set(previous_groups[index]) for index in context_groups))


def _induced_submol(
    mol: Chem.Mol,
    selected_atom_ids: set[int],
) -> tuple[Chem.Mol, list[int]]:
    old_indices = [
        atom.GetIdx() for atom in mol.GetAtoms()
        if int(atom.GetAtomMapNum()) in selected_atom_ids
    ]
    old_to_new: dict[int, int] = {}
    builder = Chem.RWMol()
    for old_index in old_indices:
        atom = Chem.Atom(mol.GetAtomWithIdx(old_index))
        atom.SetAtomMapNum(0)
        old_to_new[old_index] = builder.AddAtom(atom)
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in old_to_new and right in old_to_new:
            builder.AddBond(old_to_new[left], old_to_new[right], bond.GetBondType())
            local_bond = builder.GetBondBetweenAtoms(old_to_new[left], old_to_new[right])
            local_bond.SetIsAromatic(bond.GetIsAromatic())
    submol = builder.GetMol()
    status = Chem.SanitizeMol(submol, catchErrors=True)
    if status != Chem.SanitizeFlags.SANITIZE_NONE:
        raise FrozenMappingError(f"local subgraph sanitization failed: {status}")
    return submol, old_indices


def _map_local_candidate(
    full_mol: Chem.Mol,
    context_atom_ids: set[int],
) -> tuple[list[dict[str, Any]], dict[str, float], list[str]]:
    started = time.perf_counter()
    submol, selected_old_indices = _induced_submol(full_mol, context_atom_ids)
    extracted = time.perf_counter()

    smiles = Chem.MolToSmiles(submol, canonical=False, isomericSmiles=True)
    order = ast.literal_eval(submol.GetProp("_smilesAtomOutputOrder"))
    ordered_old_indices = [selected_old_indices[index] for index in order]
    ordered_atom_ids = [
        int(full_mol.GetAtomWithIdx(index).GetAtomMapNum())
        for index in ordered_old_indices
    ]
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None or parsed.GetNumAtoms() != len(ordered_atom_ids):
        raise FrozenMappingError("local SMILES round-trip changed the atom count")
    prepared = time.perf_counter()

    # Preserve the full-molecule H counts at the cut boundary.  RDKit would
    # otherwise add implicit H atoms to the isolated local fragment.
    full_properties = get_atom_properties(full_mol)
    local_properties = [copy.deepcopy(full_properties[index]) for index in ordered_old_indices]
    mapping = map_molecule(
        _adapter_tokens(smiles),
        connectivity_matrix(parsed, parsed.GetNumAtoms()),
        local_properties,
    )
    sectioned = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        final = map_martini_beads(mapping, [""] * len(ordered_atom_ids), get_m3_dict())
    assigned = time.perf_counter()

    grouped: dict[str, list[int]] = defaultdict(list)
    for tag, atom_id in zip(final, ordered_atom_ids):
        grouped[str(tag)].append(atom_id)
    groups = [
        {"atoms": sorted(atom_ids), "type": tag[:-6].replace("+", "")}
        for tag, atom_ids in grouped.items()
    ]
    timings = {
        "extract_seconds": extracted - started,
        "prepare_seconds": prepared - extracted,
        "section_seconds": sectioned - prepared,
        "assign_seconds": assigned - sectioned,
        "local_total_seconds": assigned - started,
    }
    notices = sorted({str(item.message) for item in caught})
    return groups, timings, notices


def run_local_mapping(
    smiles: str,
    previous_mapping: Any,
    reaction_atom_ids: Iterable[int],
    *,
    context_layers: int = 1,
) -> dict[str, Any]:
    """Run strict local mapping and return a JSON-serializable audit payload.

    ``smiles`` must contain unique, positive atom-map numbers.  These persistent
    IDs must match the atom IDs in ``previous_mapping`` and
    ``reaction_atom_ids``.
    """
    total_started = time.perf_counter()
    if context_layers < 0:
        raise ValueError("context_layers must be non-negative")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise FrozenMappingError("RDKit could not parse the atom-mapped SMILES")
    atom_ids = [int(atom.GetAtomMapNum()) for atom in mol.GetAtoms()]
    if any(atom_id <= 0 for atom_id in atom_ids) or len(atom_ids) != len(set(atom_ids)):
        raise FrozenMappingError(
            "local Mapper requires a unique positive atom-map number on every atom"
        )
    atom_id_set = set(atom_ids)
    previous_groups = _normalize_previous_groups(previous_mapping)
    _validate_partition(previous_groups, atom_id_set)
    reaction_atom_ids = {int(atom) for atom in reaction_atom_ids}
    if not reaction_atom_ids or not reaction_atom_ids <= atom_id_set:
        raise FrozenMappingError(
            "reaction atom IDs must be a non-empty subset of the mapped SMILES atoms"
        )

    editable_group_indices = {
        index for index, group in enumerate(previous_groups)
        if _atom_set(group) & reaction_atom_ids
    }
    editable_atom_ids = set().union(
        *(_atom_set(previous_groups[index]) for index in editable_group_indices)
    )
    frozen_groups = [
        dict(group) for index, group in enumerate(previous_groups)
        if index not in editable_group_indices
    ]

    group_graph, group_by_atom = _bead_graph(mol, previous_groups)
    context_atom_ids = _select_context_atoms(
        mol,
        previous_groups,
        editable_group_indices,
        context_layers,
        group_graph,
        group_by_atom,
    )
    candidate_groups, timings, notices = _map_local_candidate(mol, context_atom_ids)

    # Strict boundary rule: frozen atoms are useful as read-only chemical
    # context, but they are removed from every candidate before commit.  The
    # remaining editable fragment may contain one atom; singleton beads are
    # explicitly valid for post-crosslink mapping in this project.
    editable_groups = []
    boundary_trimmed_candidates = 0
    singleton_beads = 0
    for candidate in candidate_groups:
        candidate_atoms = _atom_set(candidate)
        kept_atoms = sorted(candidate_atoms & editable_atom_ids)
        if not kept_atoms:
            continue
        if candidate_atoms - editable_atom_ids:
            boundary_trimmed_candidates += 1
        if len(kept_atoms) == 1:
            singleton_beads += 1
        editable_groups.append({"atoms": kept_atoms, "type": str(candidate["type"])})

    _validate_partition(editable_groups, editable_atom_ids)
    output_groups = frozen_groups + editable_groups
    _validate_partition(output_groups, atom_id_set)

    frozen_signatures = {
        (_atom_set(group), str(group["type"])) for group in frozen_groups
    }
    output_signatures = {
        (_atom_set(group), str(group["type"])) for group in output_groups
    }
    frozen_violations = len(frozen_signatures - output_signatures)
    if frozen_violations:
        raise FrozenMappingError("at least one frozen bead changed during local merge")

    return {
        "mode": "strict_local_mapper",
        "groups": output_groups,
        "reaction_atom_ids": sorted(reaction_atom_ids),
        "editable_atom_ids": sorted(editable_atom_ids),
        "context_atom_ids": sorted(context_atom_ids),
        "editable_pre_beads": len(editable_group_indices),
        "frozen_pre_beads": len(frozen_groups),
        "boundary_trimmed_candidates": boundary_trimmed_candidates,
        "singleton_beads": singleton_beads,
        "frozen_violations": frozen_violations,
        "warnings": notices,
        "timings": {
            **timings,
            "total_seconds": time.perf_counter() - total_started,
        },
    }
