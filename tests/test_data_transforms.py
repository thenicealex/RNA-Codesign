"""
Tests for data/data_transforms.py

Tests core data transformation functions including:
- Frame computations
- Torsion angle computations
- Atom position computations
"""

import pytest
import torch

from data import data_transforms

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def num_res():
    return 10


@pytest.fixture
def sample_feats(batch_size, num_res):
    """Create sample features for testing transforms."""
    # Create realistic atom positions (28 atoms per residue)
    # This is a simplified representation for testing
    all_atom_positions = torch.randn(batch_size, num_res, 28, 3)
    all_atom_mask = torch.ones(batch_size, num_res, 28)

    return {
        "aatype": torch.randint(0, 4, (batch_size, num_res)),
        "all_atom_positions": all_atom_positions,
        "all_atom_mask": all_atom_mask,
        "bb_mask": torch.ones(batch_size, num_res, 3),  # All backbone atoms present
        "res_mask": torch.ones(batch_size, num_res),
    }


# ---------------------------------------------------------------------------
# Test atom28_to_frames
# ---------------------------------------------------------------------------


class TestAtom28ToFrames:
    """Tests for atom28_to_frames transform."""

    def test_returns_correct_keys(self, sample_feats):
        result = data_transforms.atom28_to_frames(sample_feats)

        # Check for actual keys returned
        expected_keys = ["rigidgroups_gt_frames", "rigidgroups_gt_exists"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_frame_shape(self, sample_feats):
        result = data_transforms.atom28_to_frames(sample_feats)

        batch_size, num_res = sample_feats["aatype"].shape
        # Frames should have shape (B, N, 10, 4, 4) - 10 rigid groups per residue
        assert result["rigidgroups_gt_frames"].shape == (batch_size, num_res, 10, 4, 4)

    def test_exists_shape(self, sample_feats):
        result = data_transforms.atom28_to_frames(sample_feats)

        batch_size, num_res = sample_feats["aatype"].shape
        # Exists should have shape (B, N, 10) - 10 rigid groups per residue
        assert result["rigidgroups_gt_exists"].shape == (batch_size, num_res, 10)


# ---------------------------------------------------------------------------
# Test make_atom23_masks
# ---------------------------------------------------------------------------


class TestMakeAtom23Masks:
    """Tests for make_atom23_masks transform."""

    def test_returns_correct_keys(self, sample_feats):
        result = data_transforms.make_atom23_masks(sample_feats)

        # Check for actual keys returned
        expected_keys = ["atom23_atom_exists", "residx_atom23_to_atom28"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_atom_exists_shape(self, sample_feats):
        result = data_transforms.make_atom23_masks(sample_feats)

        batch_size, num_res = sample_feats["aatype"].shape
        # atom23_atom_exists should have shape (B, N, 23)
        assert result["atom23_atom_exists"].shape == (batch_size, num_res, 23)


# ---------------------------------------------------------------------------
# Test atom28_to_torsion_angles
# ---------------------------------------------------------------------------


class TestAtom28ToTorsionAngles:
    """Tests for atom28_to_torsion_angles transform."""

    def test_returns_correct_keys(self, sample_feats):
        # Need frames first
        feats = data_transforms.atom28_to_frames(sample_feats)
        feats.update(sample_feats)

        result = data_transforms.atom28_to_torsion_angles(feats)

        expected_keys = ["torsion_angles_sin_cos", "alt_torsion_angles_sin_cos", "torsion_angles_mask"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_torsion_shape(self, sample_feats):
        # Need frames first
        feats = data_transforms.atom28_to_frames(sample_feats)
        feats.update(sample_feats)

        result = data_transforms.atom28_to_torsion_angles(feats)

        batch_size, num_res = sample_feats["aatype"].shape
        # torsion_angles_sin_cos should have shape (B, N, 9, 2) - 9 angles for RNA
        assert result["torsion_angles_sin_cos"].shape == (batch_size, num_res, 9, 2)


# ---------------------------------------------------------------------------
# Test compose_transforms
# ---------------------------------------------------------------------------


class TestComposeTransforms:
    """Test that transforms can be composed."""

    def test_compose_all_transforms(self, sample_feats):
        """Test composing all three main transforms."""
        transforms = [
            data_transforms.atom28_to_frames,
            data_transforms.make_atom23_masks,
            data_transforms.atom28_to_torsion_angles,
        ]

        result = sample_feats
        for transform in transforms:
            result = transform(result)

        # Check that all expected keys are present
        expected_keys = [
            "rigidgroups_gt_frames",
            "rigidgroups_gt_exists",
            "atom23_atom_exists",
            "residx_atom23_to_atom28",
            "torsion_angles_sin_cos",
            "torsion_angles_mask",
        ]

        for key in expected_keys:
            assert key in result, f"Missing key after compose: {key}"


# ---------------------------------------------------------------------------
# Test edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases for data transforms."""

    def test_missing_backbone_atoms(self, sample_feats):
        """Test with missing backbone atoms."""
        # Set some backbone masks to 0
        sample_feats["bb_mask"][:, :5, :] = 0

        result = data_transforms.atom28_to_frames(sample_feats)

        # Should still return valid output
        assert "rigidgroups_gt_frames" in result
        assert "rigidgroups_gt_exists" in result

    def test_single_residue(self):
        """Test with single residue."""
        feats = {
            "aatype": torch.randint(0, 4, (1, 1)),
            "all_atom_positions": torch.zeros(1, 1, 28, 3),
            "all_atom_mask": torch.ones(1, 1, 28),
        }

        result = data_transforms.atom28_to_frames(feats)

        assert result["rigidgroups_gt_frames"].shape == (1, 1, 10, 4, 4)
