import json
from pathlib import Path

import numpy as np
import torch

from data import data_transforms
from np import residue_constants as rc
from utils import feats


def test_licensed_templates_reconstruct_reference_coordinates():
    """Extracted torsions must recover the reference geometry for every RNA base."""
    aatypes = torch.arange(4).reshape(1, 4)
    positions = torch.tensor(rc.REFERENCE_ATOM28[:4]).unsqueeze(0)
    masks = torch.tensor(rc.STANDARD_ATOM_MASK[:4]).unsqueeze(0).float()
    features = {
        "aatype": aatypes,
        "all_atom_positions": positions,
        "all_atom_mask": masks,
    }
    features = data_transforms.make_atom23_masks(features)
    features = data_transforms.atom28_to_torsion_angles(features)
    reconstructed, _ = feats.to_atom28(
        torch.zeros(1, 4, 3),
        torch.eye(3)[None, None].repeat(1, 4, 1, 1),
        features["torsion_angles_sin_cos"],
        aatypes,
        features,
    )
    assert torch.allclose(reconstructed * masks[..., None], positions, atol=2e-5)
    assert torch.isfinite(reconstructed).all()


def test_template_rotations_and_checkpoint_layout():
    assert rc.restypes == ["A", "G", "C", "U"]
    assert rc.atom_order["C1'"] == 11
    assert rc.restype_atom23_rigid_group_positions.shape == (5, 23, 3)
    rotations = rc.restype_rigid_group_default_frame[:4, :, :3, :3]
    np.testing.assert_allclose(rotations @ rotations.swapaxes(-1, -2), np.broadcast_to(np.eye(3), rotations.shape), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(rotations), 1, atol=1e-6)
    assert np.count_nonzero(rc.STANDARD_ATOM_MASK[4]) == 0
    data = json.loads(Path(rc.__file__).with_name("rna_template_data.json").read_text())
    assert data["license"] == "Apache-2.0"
    assert "24ef5b9d19349bc6a3ecd8075742334e471cbd7d" in data["source"]
