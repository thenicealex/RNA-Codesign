# Modified for RNA-CodeSign RNA processing and inference.
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps

import numpy as np
import torch

from np import residue_constants as rc
from utils.rigid_utils import Rigid, Rotation
from utils.tensor_utils import batched_gather, tensor_tree_map, tree_map


def cast_to_64bit_ints(feats):
    # We keep all ints as int64
    for k, v in feats.items():
        if v.dtype == torch.int32:
            feats[k] = v.type(torch.int64)

    return feats


def make_one_hot(x, num_classes):
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    x_one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return x_one_hot


def make_seq_mask(feats):
    feats["seq_mask"] = torch.ones(feats["aatype"].shape, dtype=torch.float32)
    return feats


def curry1(f):
    """Supply all arguments but the first."""

    @wraps(f)
    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


def make_all_atom_aatype(feats):
    feats["all_atom_aatype"] = feats["aatype"]
    return feats


def fix_templates_aatype(feats):
    # Map one-hot to indices
    num_templates = feats["template_aatype"].shape[0]
    if num_templates > 0:
        feats["template_aatype"] = torch.argmax(feats["template_aatype"], dim=-1)
        # Map hhsearch-aatype to our aatype.
        new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
        new_order = torch.tensor(
            new_order_list,
            dtype=torch.int64,
            device=feats["aatype"].device,
        ).expand(num_templates, -1)
        feats["template_aatype"] = torch.gather(new_order, 1, index=feats["template_aatype"])

    return feats


@curry1
def add_distillation_flag(feats, distillation):
    feats["is_distillation"] = distillation
    return feats


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Similar to
    tf.unsorted_segment_sum, but only supports 1-D indices.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The 1-D segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert len(segment_ids.shape) == 1
    assert segment_ids.shape[0] == data.shape[0]
    segment_ids = segment_ids.view(segment_ids.shape[0], *((1,) * len(data.shape[1:])))
    segment_ids = segment_ids.expand(data.shape)
    shape = [num_segments, *list(data.shape[1:])]
    tensor = torch.zeros(*shape, device=segment_ids.device).scatter_add_(0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_mask):
    """Create pseudo beta features."""
    is_prn = (aatype == rc.restype_order["A"]) + (aatype == rc.restype_order["G"])
    n9_idx = rc.atom_order["N9"]
    n1_idx = rc.atom_order["N1"]

    pseudo_beta = torch.where(
        is_prn[..., None].expand(*((-1,) * len(is_prn.shape)), 3),
        all_atom_positions[..., n9_idx, :],
        all_atom_positions[..., n1_idx, :],
    )

    if all_atom_mask is not None:
        pseudo_beta_mask = torch.where(
            is_prn,
            all_atom_mask[..., n9_idx],
            all_atom_mask[..., n1_idx],
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


@curry1
def make_pseudo_beta(feats, prefix=""):
    """Create pseudo-beta (alpha for glycine) position and mask."""
    assert prefix in ["", "template_"]
    (
        feats[prefix + "pseudo_beta"],
        feats[prefix + "pseudo_beta_mask"],
    ) = pseudo_beta_fn(
        feats["template_aatype" if prefix else "aatype"],
        feats[prefix + "all_atom_positions"],
        feats["template_all_atom_mask" if prefix else "all_atom_mask"],
    )
    return feats


def atom_pick_fn(aatype, all_atom_positions, all_atom_masks, atom):
    """this is main chain only"""
    is_prn = (aatype == rc.restype_order["A"]) + (aatype == rc.restype_order["G"])
    trg_idx = rc.atom_order[atom]

    atom_pos = torch.where(
        is_prn[..., None].expand(*((-1,) * len(is_prn.shape)), 3),
        all_atom_positions[..., trg_idx, :],
        all_atom_positions[..., trg_idx, :],
    )

    if all_atom_masks is not None:
        atom_pos_mask = torch.where(
            is_prn,
            all_atom_masks[..., trg_idx],
            all_atom_masks[..., trg_idx],
        )
        return atom_pos, atom_pos_mask
    else:
        return atom_pos_mask


def shaped_categorical(probs, epsilon=1e-10):
    ds = probs.shape
    num_classes = ds[-1]
    distribution = torch.distributions.categorical.Categorical(torch.reshape(probs + epsilon, [-1, num_classes]))
    counts = distribution.sample()
    return torch.reshape(counts, ds[:-1])


@curry1
def select_feat(feats, feature_list):
    return {k: v for k, v in feats.items() if k in feature_list}


@curry1
def crop_templates(feats, max_templates):
    for k, v in feats.items():
        if k.startswith("template_"):
            feats[k] = v[:max_templates]
    return feats


def make_atom23_masks(feats):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    restype_atom23_to_atom28 = []
    restype_atom37_to_atom14 = []
    restype_atom23_mask = []

    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom23_names[rc.restype_1to3[rt]]
        restype_atom23_to_atom28.append([(rc.atom_order[name] if name else 0) for name in atom_names])
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [(atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0) for name in rc.atom_types]
        )

        restype_atom23_mask.append([(1.0 if name else 0.0) for name in atom_names])

    # Add dummy mapping for restype 'N'
    restype_atom23_to_atom28.append([0] * 23)
    restype_atom37_to_atom14.append([0] * 28)
    restype_atom23_mask.append([0.0] * 23)

    restype_atom23_to_atom28 = torch.tensor(
        restype_atom23_to_atom28,
        dtype=torch.int32,
        device=feats["aatype"].device,
    )
    restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14,
        dtype=torch.int32,
        device=feats["aatype"].device,
    )
    restype_atom23_mask = torch.tensor(
        restype_atom23_mask,
        dtype=torch.float32,
        device=feats["aatype"].device,
    )
    nucleicacid_aatype = feats["aatype"].to(torch.long)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this feats
    residx_atom23_to_atom28 = restype_atom23_to_atom28[nucleicacid_aatype]
    residx_atom14_mask = restype_atom23_mask[nucleicacid_aatype]

    feats["atom23_atom_exists"] = residx_atom14_mask
    feats["residx_atom23_to_atom28"] = residx_atom23_to_atom28.long()

    # create the gather indices for mapping back
    residx_atom28_to_atom23 = restype_atom37_to_atom14[nucleicacid_aatype]
    feats["residx_atom28_to_atom23"] = residx_atom28_to_atom23.long()

    # create the corresponding mask
    restype_atom28_mask = torch.zeros([5, 28], dtype=torch.float32, device=feats["aatype"].device)
    for restype, restype_letter in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom_order[atom_name]
            restype_atom28_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom28_mask[nucleicacid_aatype]
    feats["atom28_atom_exists"] = residx_atom37_mask

    return feats


def make_atom23_masks_np(batch):
    batch = tree_map(lambda n: torch.tensor(n, device="cpu"), batch, np.ndarray)
    out = make_atom23_masks(batch)
    out = tensor_tree_map(lambda t: np.array(t), out)
    return out


def make_atom23_positions(feats):
    """Constructs denser atom positions (14 dimensions instead of 37)."""
    residx_atom14_mask = feats["atom23_atom_exists"]
    residx_atom23_to_atom28 = feats["residx_atom23_to_atom28"]

    # Create a mask for known ground truth positions.
    residx_atom14_gt_mask = residx_atom14_mask * batched_gather(
        feats["all_atom_mask"],
        residx_atom23_to_atom28,
        dim=-1,
        no_batch_dims=len(feats["all_atom_mask"].shape[:-1]),
    )

    # Gather the ground truth positions.
    residx_atom23_gt_positions = residx_atom14_gt_mask[..., None] * (
        batched_gather(
            feats["all_atom_positions"],
            residx_atom23_to_atom28,
            dim=-2,
            no_batch_dims=len(feats["all_atom_positions"].shape[:-2]),
        )
    )

    feats["atom23_atom_exists"] = residx_atom14_mask
    feats["atom23_gt_exists"] = residx_atom14_gt_mask
    feats["atom23_gt_positions"] = residx_atom23_gt_positions

    # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
    # alternative ground truth coordinates where the naming is swapped
    restype_3 = [rc.restype_1to3[res] for res in rc.restypes]
    restype_3 += ["N"]

    # Matrices for renaming ambiguous atoms.
    all_matrices = {
        res: torch.eye(
            23,
            dtype=feats["all_atom_mask"].dtype,
            device=feats["all_atom_mask"].device,
        )
        for res in restype_3
    }
    for resname, swap in rc.residue_atom_renaming_swaps.items():
        correspondences = torch.arange(
            23,
            device=feats["all_atom_mask"].device,
        )
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = rc.restype_name_to_atom23_names[resname].index(source_atom_swap)
            target_index = rc.restype_name_to_atom23_names[resname].index(target_atom_swap)
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            # renaming_matrix = feats["all_atom_mask"].new_zeros((14, 14))
            renaming_matrix = feats["all_atom_mask"].new_zeros((23, 23))
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.0
        all_matrices[resname] = renaming_matrix

    renaming_matrices = torch.stack([all_matrices[restype] for restype in restype_3])

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[feats["aatype"]]

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = torch.einsum("...rac,...rab->...rbc", residx_atom23_gt_positions, renaming_transform)
    feats["atom23_alt_gt_positions"] = alternative_gt_positions

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position).
    alternative_gt_mask = torch.einsum("...ra,...rab->...rb", residx_atom14_gt_mask, renaming_transform)
    feats["atom23_alt_gt_exists"] = alternative_gt_mask

    # Create an ambiguous atoms mask.  shape: (21, 14).
    # restype_atom14_is_ambiguous = feats["all_atom_mask"].new_zeros((21, 14))
    restype_atom14_is_ambiguous = feats["all_atom_mask"].new_zeros((5, 23))
    for resname, swap in rc.residue_atom_renaming_swaps.items():
        for atom_name1, atom_name2 in swap.items():
            restype = rc.restype_order[rc.restype_3to1[resname]]
            atom_idx1 = rc.restype_name_to_atom23_names[resname].index(atom_name1)
            atom_idx2 = rc.restype_name_to_atom23_names[resname].index(atom_name2)
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    # From this create an ambiguous_mask for the given sequence.
    feats["atom23 _atom_is_ambiguous"] = restype_atom14_is_ambiguous[feats["aatype"]]

    return feats


def atom28_to_frames(feats, eps=1e-8):
    aatype: torch.Tensor = feats["aatype"]
    all_atom_positions: torch.Tensor = feats["all_atom_positions"]
    all_atom_mask: torch.Tensor = feats["all_atom_mask"]

    batch_dims = len(aatype.shape[:-1])

    restype_rigidgroup_base_atom_names = np.full([5, 10, 3], "", dtype=object)
    restype_rigidgroup_base_atom_names[:, 0, :] = ["C2'", "C1'", "O4'"]  # bb ??
    restype_rigidgroup_base_atom_names[:, 1, :] = ["C1'", "O4'", "C4'"]  # angle1
    restype_rigidgroup_base_atom_names[:, 2, :] = ["O4'", "C4'", "C5'"]  # angle2
    restype_rigidgroup_base_atom_names[:, 3, :] = ["C4'", "C5'", "O5'"]  # angle3
    restype_rigidgroup_base_atom_names[:, 4, :] = ["C5'", "O5'", "P"]  # angle4
    restype_rigidgroup_base_atom_names[:, 5, :] = ["O5'", "P", "OP1"]  # angle5
    restype_rigidgroup_base_atom_names[:, 6, :] = ["C1'", "C2'", "O2'"]  # angle6
    restype_rigidgroup_base_atom_names[:, 7, :] = ["C1'", "C2'", "C3'"]  # angle7
    restype_rigidgroup_base_atom_names[:, 8, :] = ["C2'", "C3'", "O3'"]  # angle8

    for restype, restype_letter in enumerate(rc.restypes):
        resname = rc.restype_1to3[restype_letter]
        for chi_idx in range(1):
            if rc.chi_angles_mask[restype][chi_idx]:
                names = rc.chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom_names[restype, 9, :] = names[1:]

    restype_rigidgroup_mask = all_atom_mask.new_zeros(
        (*aatype.shape[:-1], 5, 10),
    )
    restype_rigidgroup_mask[..., 0:9] = 1
    restype_rigidgroup_mask[..., :4, 9:] = all_atom_mask.new_tensor(rc.chi_angles_mask)

    lookuptable = rc.atom_order.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    restype_rigidgroup_base_atom28_idx = lookup(
        restype_rigidgroup_base_atom_names,
    )
    restype_rigidgroup_base_atom28_idx = aatype.new_tensor(
        restype_rigidgroup_base_atom28_idx,
    )
    restype_rigidgroup_base_atom28_idx = restype_rigidgroup_base_atom28_idx.view(
        *((1,) * batch_dims), *restype_rigidgroup_base_atom28_idx.shape
    )

    residx_rigidgroup_base_atom28_idx = batched_gather(
        restype_rigidgroup_base_atom28_idx,
        aatype,
        dim=-3,
        no_batch_dims=batch_dims,
    )

    base_atom_pos = batched_gather(
        all_atom_positions,
        residx_rigidgroup_base_atom28_idx,
        dim=-2,
        no_batch_dims=len(all_atom_positions.shape[:-2]),
    )

    gt_frames = Rigid.from_3_points(
        p_neg_x_axis=base_atom_pos[..., 0, :],
        origin=base_atom_pos[..., 1, :],
        p_xy_plane=base_atom_pos[..., 2, :],
        eps=eps,
    )

    group_exists = batched_gather(
        restype_rigidgroup_mask,
        aatype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    gt_atoms_exist = batched_gather(
        all_atom_mask,
        residx_rigidgroup_base_atom28_idx,
        dim=-1,
        no_batch_dims=len(all_atom_mask.shape[:-1]),
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists

    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
    rots = torch.tile(rots, (*((1,) * batch_dims), 10, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1
    rots = Rotation(rot_mats=rots)

    gt_frames = gt_frames.compose(Rigid(rots, None))

    restype_rigidgroup_is_ambiguous = all_atom_mask.new_zeros(*((1,) * batch_dims), 5, 10)
    restype_rigidgroup_rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
    restype_rigidgroup_rots = torch.tile(
        restype_rigidgroup_rots,
        (*((1,) * batch_dims), 5, 10, 1, 1),
    )

    for resname, _ in rc.residue_atom_renaming_swaps.items():
        restype = rc.restype_order[rc.restype_3to1[resname]]
        chi_idx = int(sum(rc.chi_angles_mask[restype]) - 1)
        restype_rigidgroup_is_ambiguous[..., restype, chi_idx + 9] = 1
        restype_rigidgroup_rots[..., restype, chi_idx + 9, 1, 1] = -1
        restype_rigidgroup_rots[..., restype, chi_idx + 9, 2, 2] = -1

    residx_rigidgroup_is_ambiguous = batched_gather(
        restype_rigidgroup_is_ambiguous,
        aatype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = batched_gather(
        restype_rigidgroup_rots,
        aatype,
        dim=-4,
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = Rotation(rot_mats=residx_rigidgroup_ambiguity_rot)
    alt_gt_frames = gt_frames.compose(Rigid(residx_rigidgroup_ambiguity_rot, None))

    gt_frames_tensor = gt_frames.to_tensor_4x4()
    alt_gt_frames_tensor = alt_gt_frames.to_tensor_4x4()

    feats["rigidgroups_gt_frames"] = gt_frames_tensor
    feats["rigidgroups_gt_exists"] = gt_exists
    feats["rigidgroups_group_exists"] = group_exists
    feats["rigidgroups_group_is_ambiguous"] = residx_rigidgroup_is_ambiguous
    feats["rigidgroups_alt_gt_frames"] = alt_gt_frames_tensor

    return feats


def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=4, chis=1, atoms=4]. The residue types are
      in the order specified in rc.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in rc.restypes:
        residue_name = rc.restype_1to3[residue_name]
        residue_chi_angles = rc.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([rc.atom_order[atom] for atom in chi_angle])
        for _ in range(1 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 1)  # For UNKNOWN residue.

    return chi_atom_indices


def atom28_to_torsion_angles(
    feats,
    prefix: str = "",
) -> object:
    """
    Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        Dict containing:
            * (prefix)aatype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 28, 3] atom positions (in atom28 format)
            * (prefix)all_atom_mask:
                [*, N_res, 28] atom position mask
    Returns:
        The same dictionary updated with the following features:

        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)alt_torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Alternate torsion angles (accounting for 180-degree symmetry)
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """
    aatype: torch.Tensor = feats[prefix + "aatype"]
    all_atom_positions: torch.Tensor = feats[prefix + "all_atom_positions"]
    all_atom_mask: torch.Tensor = feats[prefix + "all_atom_mask"]

    aatype = torch.clamp(aatype, max=4)

    pad = all_atom_positions.new_zeros([*all_atom_positions.shape[:-3], 1, 28, 3])  # [0 1 28 3]
    prev_all_atom_positions = torch.cat([pad, all_atom_positions[..., :-1, :, :]], dim=-3)  # [0 L 28 3]
    fol_all_atom_positions = torch.cat(
        [
            all_atom_positions[..., 1:, :, :],
            pad,
        ],
        dim=-3,
    )  # [0 L 28 3]

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 28])  # [0 1 28]
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)  # [0 L 28]
    fol_all_atom_mask = torch.cat(
        [
            all_atom_mask[..., 1:, :],
            pad,
        ],
        dim=-2,
    )  # [0 L 28]

    # main chain angles
    angle1_atom_pos = torch.cat(
        [
            all_atom_positions[..., 9:10, :],  # C2'
            all_atom_positions[..., 11:12, :],  # C1'
            all_atom_positions[..., 6:7, :],  # O4'
            all_atom_positions[..., 5:6, :],  # C4'
        ],
        dim=-2,
    )
    angle1_mask = (
        torch.prod(all_atom_mask[..., 9:10], dim=-1)
        * torch.prod(all_atom_mask[..., 11:12], dim=-1)
        * torch.prod(all_atom_mask[..., 6:7], dim=-1)
        * torch.prod(all_atom_mask[..., 5:6], dim=-1)
    )

    angle2_atom_pos = torch.cat(
        [
            all_atom_positions[..., 11:12, :],  # C1'
            all_atom_positions[..., 6:7, :],  # O4'
            all_atom_positions[..., 5:6, :],  # C4'
            all_atom_positions[..., 4:5, :],  # C5'
        ],
        dim=-2,
    )
    angle2_mask = (
        torch.prod(all_atom_mask[..., 11:12], dim=-1)
        * torch.prod(all_atom_mask[..., 6:7], dim=-1)
        * torch.prod(all_atom_mask[..., 5:6], dim=-1)
        * torch.prod(all_atom_mask[..., 4:5], dim=-1)
    )

    angle3_atom_pos = torch.cat(
        [
            all_atom_positions[..., 6:7, :],  # O4'
            all_atom_positions[..., 5:6, :],  # C4'
            all_atom_positions[..., 4:5, :],  # C5'
            all_atom_positions[..., 3:4, :],  # O5'
        ],
        dim=-2,
    )
    angle3_mask = (
        torch.prod(all_atom_mask[..., 6:7], dim=-1)
        * torch.prod(all_atom_mask[..., 5:6], dim=-1)
        * torch.prod(all_atom_mask[..., 4:5], dim=-1)
        * torch.prod(all_atom_mask[..., 3:4], dim=-1)
    )

    angle4_atom_pos = torch.cat(
        [
            all_atom_positions[..., 5:6, :],  # C4'
            all_atom_positions[..., 4:5, :],  # C5'
            all_atom_positions[..., 3:4, :],  # O5'
            all_atom_positions[..., 0:1, :],  # P
        ],
        dim=-2,
    )
    angle4_mask = (
        torch.prod(all_atom_mask[..., 5:6], dim=-1)
        * torch.prod(all_atom_mask[..., 4:5], dim=-1)
        * torch.prod(all_atom_mask[..., 3:4], dim=-1)
        * torch.prod(all_atom_mask[..., 0:1], dim=-1)
    )

    angle5_atom_pos = torch.cat(
        [
            all_atom_positions[..., 4:5, :],  # C5'
            all_atom_positions[..., 3:4, :],  # O5'
            all_atom_positions[..., 0:1, :],  # P
            all_atom_positions[..., 1:2, :],  # OP1
        ],
        dim=-2,
    )
    angle5_mask = (
        torch.prod(all_atom_mask[..., 4:5], dim=-1)
        * torch.prod(all_atom_mask[..., 3:4], dim=-1)
        * torch.prod(all_atom_mask[..., 0:1], dim=-1)
        * torch.prod(all_atom_mask[..., 1:2], dim=-1)
    )

    angle6_atom_pos = torch.cat(
        [
            all_atom_positions[..., 6:7, :],  # O4'
            all_atom_positions[..., 11:12, :],  # C1'
            all_atom_positions[..., 9:10, :],  # C2'
            all_atom_positions[..., 10:11, :],  # O2'
        ],
        dim=-2,
    )
    angle6_mask = (
        torch.prod(all_atom_mask[..., 6:7], dim=-1)
        * torch.prod(all_atom_mask[..., 11:12], dim=-1)
        * torch.prod(all_atom_mask[..., 9:10], dim=-1)
        * torch.prod(all_atom_mask[..., 10:11], dim=-1)
    )

    angle7_atom_pos = torch.cat(
        [
            all_atom_positions[..., 6:7, :],  # O4'
            all_atom_positions[..., 11:12, :],  # C1'
            all_atom_positions[..., 9:10, :],  # C2'
            all_atom_positions[..., 7:8, :],  # C3'
        ],
        dim=-2,
    )
    angle7_mask = (
        torch.prod(all_atom_mask[..., 6:7], dim=-1)
        * torch.prod(all_atom_mask[..., 11:12], dim=-1)
        * torch.prod(all_atom_mask[..., 9:10], dim=-1)
        * torch.prod(all_atom_mask[..., 7:8], dim=-1)
    )

    angle8_atom_pos = torch.cat(
        [
            all_atom_positions[..., 11:12, :],  # C1'
            all_atom_positions[..., 9:10, :],  # C2'
            all_atom_positions[..., 7:8, :],  # C3'
            all_atom_positions[..., 8:9, :],  # O3'
        ],
        dim=-2,
    )
    angle8_mask = (
        torch.prod(all_atom_mask[..., 11:12], dim=-1)
        * torch.prod(all_atom_mask[..., 9:10], dim=-1)
        * torch.prod(all_atom_mask[..., 7:8], dim=-1)
        * torch.prod(all_atom_mask[..., 8:9], dim=-1)
    )

    chi_atom_indices = torch.as_tensor(get_chi_atom_indices(), device=aatype.device)

    atom_indices = chi_atom_indices[..., aatype, :, :]
    chis_atom_pos = batched_gather(all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2]))

    chi_angles_mask = list(rc.chi_angles_mask)
    chi_angles_mask.append([0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[aatype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype)
    chis_mask = chis_mask * chi_angle_atoms_mask

    torsions_atom_pos = torch.cat(
        [
            angle1_atom_pos[..., None, :, :],
            angle2_atom_pos[..., None, :, :],
            angle3_atom_pos[..., None, :, :],
            angle4_atom_pos[..., None, :, :],
            angle5_atom_pos[..., None, :, :],
            angle6_atom_pos[..., None, :, :],
            angle7_atom_pos[..., None, :, :],
            angle8_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_angles_mask = torch.cat(
        [
            angle1_mask[..., None],
            angle2_mask[..., None],
            angle3_mask[..., None],
            angle4_mask[..., None],
            angle5_mask[..., None],
            angle6_mask[..., None],
            angle7_mask[..., None],
            angle8_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    torsion_frames = Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(torsions_atom_pos[..., 3, :])

    torsion_angles_sin_cos = torch.stack([fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1)

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom
    torsion_angles_sin_cos = (
        torsion_angles_sin_cos
        * all_atom_mask.new_tensor(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # what is this ????
        )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]
    )

    chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(
        rc.chi_pi_periodic,
    )[aatype, ...]

    mirror_torsion_angles = torch.cat(
        [
            all_atom_mask.new_ones(*aatype.shape, 8),
            1.0 - 2.0 * chi_is_ambiguous,
        ],
        dim=-1,
    )
    # logger.info("torsion_angles_sin_cos", torsion_angles_sin_cos.shape)
    # logger.info("mirror_torsion_angles", mirror_torsion_angles.shape)
    alt_torsion_angles_sin_cos = torsion_angles_sin_cos * mirror_torsion_angles[..., None]

    feats[prefix + "torsion_angles_sin_cos"] = torsion_angles_sin_cos
    feats[prefix + "alt_torsion_angles_sin_cos"] = alt_torsion_angles_sin_cos
    feats[prefix + "torsion_angles_mask"] = torsion_angles_mask

    return feats


def get_backbone_frames(feats):
    # DISCREPANCY: AlphaFold uses tensor_7s here. I don't know why.
    feats["backbone_rigid_tensor"] = feats["rigidgroups_gt_frames"][..., 0, :, :]
    feats["backbone_rigid_mask"] = feats["rigidgroups_gt_exists"][..., 0]

    return feats


def get_chi_angles(feats):
    dtype = feats["all_atom_mask"].dtype
    feats["chi_angles_sin_cos"] = (feats["torsion_angles_sin_cos"][..., 8:, :]).to(dtype)
    feats["chi_mask"] = feats["torsion_angles_mask"][..., 8:].to(dtype)

    return feats
