"""Checkpoint atom conventions and geometry derived from Apache-2.0 RhoFold data.

The vocabulary and indices implement the RNA-CodeSign checkpoint interface.
Ideal coordinates are transformed from the licensed rna_template_data.json;
NuFold implementations and geometry tables are not used.
"""

import json
from pathlib import Path

import numpy as np

restypes = ["A", "G", "C", "U"]
restype_num = len(restypes)
restype_order = {letter: index for index, letter in enumerate(restypes)}
unk_restype_index = restype_num
mask_token_index = restype_num + 1
model_token_num = restype_num + 2
restypes_with_x = [*restypes, "N"]
restype_1to3 = {letter: letter for letter in restypes}
restype_3to1 = restype_1to3.copy()

# These ordered names specify the trained model's atom28/atom23 interfaces.
atom_types = [
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
    "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4", "O6", "N2", "O2", "N4", "O4", "OP3",
]
atom_type_num = len(atom_types)
atom_order = {name: index for index, name in enumerate(atom_types)}
_backbone = atom_types[:12]
residue_atoms = {
    "A": [*_backbone, "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "G": [*_backbone, "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "C": [*_backbone, "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "U": [*_backbone, "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
}
restype_name_to_atom23_names = {base: names + [""] * (23 - len(names)) for base, names in residue_atoms.items()}
residue_atom_renaming_swaps = {base: {"OP1": "OP2"} for base in restypes}
chi_angles_atoms = {
    base: [["O4'", "C1'", "N9" if base in "AG" else "N1", "C8" if base in "AG" else "C2"]]
    for base in restypes
}
chi_angles_mask = [[1.0] for _ in restypes]
chi_pi_periodic = [[0.0] for _ in restypes_with_x]

_torsions = [
    ["C2'", "C1'", "O4'", "C4'"],
    ["C1'", "O4'", "C4'", "C5'"],
    ["O4'", "C4'", "C5'", "O5'"],
    ["C4'", "C5'", "O5'", "P"],
    ["C5'", "O5'", "P", "OP1"],
    ["O4'", "C1'", "C2'", "O2'"],
    ["O4'", "C1'", "C2'", "C3'"],
    ["C1'", "C2'", "C3'", "O3'"],
]
_parents = [0, 0, 1, 2, 3, 4, 0, 0, 7, 0]
_atom_groups = {"C4'": 1, "C5'": 2, "O5'": 3, "P": 4, "OP1": 5, "OP2": 5, "O2'": 6, "C3'": 7, "O3'": 8}


def _frame(origin, axis, plane):
    """Construct an orthonormal frame from an origin and two directions."""
    x = np.asarray(axis, dtype=np.float64)
    x = x / np.linalg.norm(x)
    y = np.asarray(plane, dtype=np.float64)
    y = y - x * (x @ y)
    y = y / np.linalg.norm(y)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((x, y, np.cross(x, y)))
    result[:3, 3] = origin
    return result


def _reference_positions(data, base):
    """Expand the licensed RhoFold group coordinates into Cartesian positions."""
    atoms = data["ATOM_INFOS_PER_RESD"][base]
    positions = {name: np.asarray(xyz, dtype=np.float64) for name, group, xyz in atoms if group == 0}
    for group, (_, _, names) in enumerate(data["ANGL_INFOS_PER_RESD"][base], start=3):
        a, b, c = (positions[name] for name in names[:3])
        basis = _frame(c, c - b, a - c)
        for name, atom_group, xyz in atoms:
            if atom_group == group:
                positions[name] = basis[:3, :3] @ xyz + c
    root = _frame(positions["C1'"], positions["C2'"] - positions["C1'"], positions["O4'"] - positions["C1'"])
    return {name: root[:3, :3].T @ (xyz - root[:3, 3]) for name, xyz in positions.items()}


restype_atom28_to_rigid_group = np.zeros((5, 28), dtype=int)
restype_atom28_mask = np.zeros((5, 28), dtype=np.float32)
restype_atom28_rigid_group_positions = np.zeros((5, 28, 3), dtype=np.float32)
restype_atom23_to_rigid_group = np.zeros((5, 23), dtype=int)
restype_atom23_mask = np.zeros((5, 23), dtype=np.float32)
restype_atom23_rigid_group_positions = np.zeros((5, 23, 3), dtype=np.float32)
restype_rigid_group_default_frame = np.zeros((5, 10, 4, 4), dtype=np.float32)
REFERENCE_ATOM28 = np.zeros((5, 28, 3), dtype=np.float32)


def _initialize_templates():
    data = json.loads(Path(__file__).with_name("rna_template_data.json").read_text())
    for base, index in restype_order.items():
        positions = _reference_positions(data, base)
        # Actual frames contain reference torsions. Zero frames omit those
        # rotations so sampling can supply the model's predicted torsions.
        actual_frames = [np.eye(4)]
        zero_frames = [np.eye(4)]
        for a_name, b_name, c_name, d_name in _torsions + chi_angles_atoms[base]:
            a, b, c, d = (positions[name] for name in (a_name, b_name, c_name, d_name))
            zero_frames.append(_frame(c, c - b, a - c))
            actual_frames.append(_frame(c, c - b, d - c))
        for group, parent in enumerate(_parents):
            restype_rigid_group_default_frame[index, group] = np.linalg.inv(actual_frames[parent]) @ zero_frames[group]
        anchor = "N9" if base in "AG" else "N1"
        for compact_index, name in enumerate(residue_atoms[base]):
            group = _atom_groups.get(name, 0 if name in {"C1'", "C2'", "O4'", anchor} else 9)
            basis = actual_frames[group]
            local = basis[:3, :3].T @ (positions[name] - basis[:3, 3])
            full_index = atom_order[name]
            REFERENCE_ATOM28[index, full_index] = positions[name]
            restype_atom28_to_rigid_group[index, full_index] = group
            restype_atom28_mask[index, full_index] = 1
            restype_atom28_rigid_group_positions[index, full_index] = local
            restype_atom23_to_rigid_group[index, compact_index] = group
            restype_atom23_mask[index, compact_index] = 1
            restype_atom23_rigid_group_positions[index, compact_index] = local


_initialize_templates()
STANDARD_ATOM_MASK = restype_atom28_mask.astype(np.int32)
