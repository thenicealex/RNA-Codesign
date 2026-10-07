# Copyright 2022 Y.K, Kihara Lab
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for the RNA-CodeSign inference atom representation.

"""Utilities for calculating all atom representations."""

import torch

from data import data_transforms
from np import residue_constants
from utils import data_utils, rigid_utils

Rigid = rigid_utils.Rigid
Rotation = rigid_utils.Rotation

# Residue Constants from OpenFold/AlphaFold2.
IDEALIZED_POS = torch.tensor(residue_constants.restype_atom23_rigid_group_positions)
DEFAULT_FRAMES = torch.tensor(residue_constants.restype_rigid_group_default_frame)
ATOM_MASK = torch.tensor(residue_constants.restype_atom23_mask)
GROUP_IDX = torch.tensor(residue_constants.restype_atom23_to_rigid_group)


def to_atom28(trans, rots, psi_torsions=None, aatype=None, chain_feats=None):
    num_batch, num_res, _ = trans.shape
    psi_torsions = (
        psi_torsions if psi_torsions is not None else torch.zeros((num_batch, num_res, 9, 2), device=trans.device)
    )
    aatype = (
        aatype.long()
        if aatype is not None
        else torch.full((num_batch, num_res), 3, device=trans.device, dtype=torch.long)
    )
    # final_atom28 = compute_backbone(
    #     du.create_rigid(rots, trans),
    #     psi_torsions,
    #     aatype,
    # )[0]

    final_atom28, _, final_atom14 = compute_all_atom(
        data_utils.create_rigid(rots, trans),
        psi_torsions,
        aatype,
        chain_feats=chain_feats,
    )
    return final_atom28, final_atom14


def torsion_angles_to_frames(
    r: Rigid,  # type: ignore [valid-type]
    alpha: torch.Tensor,
    aatype: torch.Tensor,
):
    """Conversion method of torsion angles to frames provided the backbone.

    Args:
        r: Backbone rigid groups.
        alpha: Torsion angles.
        aatype: residue types.

    Returns:
        All 8 frames corresponding to each torsion frame.

    """
    # [*, N, 8, 4, 4]
    with torch.no_grad():
        default_4x4 = DEFAULT_FRAMES.to(aatype.device)[aatype, ...]  # type: ignore [attr-defined]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)  # type: ignore [attr-defined]

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    g1_frame_to_bb = all_frames[..., 1]
    g2_frame_to_frame = all_frames[..., 2]
    g2_frame_to_bb = g1_frame_to_bb.compose(g2_frame_to_frame)
    g3_frame_to_frame = all_frames[..., 3]
    g3_frame_to_bb = g2_frame_to_bb.compose(g3_frame_to_frame)
    g4_frame_to_frame = all_frames[..., 4]
    g4_frame_to_bb = g3_frame_to_bb.compose(g4_frame_to_frame)
    g5_frame_to_frame = all_frames[..., 5]
    g5_frame_to_bb = g4_frame_to_bb.compose(g5_frame_to_frame)
    g6_frame_to_bb = all_frames[..., 6]
    g7_frame_to_bb = all_frames[..., 7]
    g8_frame_to_frame = all_frames[..., 8]
    g8_frame_to_bb = g7_frame_to_bb.compose(g8_frame_to_frame)
    g9_frame_to_bb = all_frames[..., 9]

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :1],
            g1_frame_to_bb.unsqueeze(-1),
            g2_frame_to_bb.unsqueeze(-1),
            g3_frame_to_bb.unsqueeze(-1),
            g4_frame_to_bb.unsqueeze(-1),
            g5_frame_to_bb.unsqueeze(-1),
            g6_frame_to_bb.unsqueeze(-1),
            g7_frame_to_bb.unsqueeze(-1),
            g8_frame_to_bb.unsqueeze(-1),
            g9_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def prot_to_torsion_angles(aatype, atom37, atom37_mask):
    """Calculate torsion angle features from nucleic acid features."""
    prot_feats = {
        "aatype": aatype,
        "all_atom_positions": atom37,
        "all_atom_mask": atom37_mask,
    }
    torsion_angles_feats = data_transforms.atom28_to_torsion_angles()(prot_feats)
    torsion_angles = torsion_angles_feats["torsion_angles_sin_cos"]
    torsion_mask = torsion_angles_feats["torsion_angles_mask"]
    return torsion_angles, torsion_mask


def frames_to_atom23_pos(
    r: Rigid,  # type: ignore [valid-type]
    aatype: torch.Tensor,
):
    """Convert frames to their idealized all atom representation.

    Args:
        r: All rigid groups. [..., N, 8, 3]
        aatype: Residue types. [..., N]

    Returns:

    """
    with torch.no_grad():
        group_mask = GROUP_IDX.to(aatype.device)[aatype, ...]
        group_mask = torch.nn.functional.one_hot(
            group_mask,
            num_classes=DEFAULT_FRAMES.shape[-3],
        )
        frame_atom_mask = ATOM_MASK.to(aatype.device)[aatype, ...].unsqueeze(-1)  # type: ignore [attr-defined]
        frame_null_pos = IDEALIZED_POS.to(aatype.device)[aatype, ...]  # type: ignore [attr-defined]

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask  # type: ignore [index]

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 14, 3]
    pred_positions = t_atoms_to_global.apply(frame_null_pos)
    pred_positions = pred_positions * frame_atom_mask

    return pred_positions


def compute_backbone(bb_rigids, psi_torsions, aatype):
    torsion_angles = psi_torsions
    # torsion_angles = torch.tile(
    #     psi_torsions[..., None, :],
    #     tuple([1 for _ in range(len(bb_rigids.shape))]) + (7, 1),
    # )
    # aatype = torch.zeros(bb_rigids.shape, device=bb_rigids.device).long()
    # aatype = torch.zeros(bb_rigids.shape).long().to(bb_rigids.device)
    all_frames = torsion_angles_to_frames(
        bb_rigids,
        torsion_angles,
        aatype,
    )
    atom14_pos = frames_to_atom23_pos(all_frames, aatype)
    atom37_bb_pos = torch.zeros((*bb_rigids.shape, 28, 3), device=bb_rigids.device)
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
    # atom37_bb_pos[..., :3, :] = atom14_pos[..., :3, :]
    # atom37_bb_pos[..., 3, :] = atom14_pos[..., 4, :]
    # atom37_bb_pos[..., 4, :] = atom14_pos[..., 3, :]
    # atom23
    # ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'","N9", ...]
    # atom28
    # ["P","OP1","OP2","O5'","C5'","C4'","O4'","C3'","O3'","C2'","O2'","C1'","N9", ...]
    #                               5     6           8           10    11    12
    atom37_bb_pos[:, :, :12, :] = atom37_bb_pos[:, :, :12, :]
    atom37_bb_pos[:, :, 18, :] = atom37_bb_pos[:, :, 18, :]
    atom37_mask = torch.any(atom37_bb_pos, axis=-1)
    return atom37_bb_pos, atom37_mask, aatype, atom14_pos


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def atom23_to_atom28(atom14, residx_atom28_to_atom23, atom28_atom_exists):
    atom28 = batched_gather(
        atom14,
        residx_atom28_to_atom23,
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )
    atom28 = atom28 * atom28_atom_exists[..., None]
    return atom28


def compute_all_atom(bb_rigids, psi_torsions, aatype, chain_feats):
    torsion_angles = psi_torsions
    # torsion_angles = torch.tile(
    #     psi_torsions[..., None, :],
    #     tuple([1 for _ in range(len(bb_rigids.shape))]) + (7, 1),
    # )
    # aatype = torch.zeros(bb_rigids.shape, device=bb_rigids.device).long()
    # aatype = torch.zeros(bb_rigids.shape).long().to(bb_rigids.device)
    all_frames = torsion_angles_to_frames(
        bb_rigids,
        torsion_angles,
        aatype,
    )
    atom23_pos = frames_to_atom23_pos(all_frames, aatype)
    atom28_pos = atom23_to_atom28(
        atom23_pos,
        chain_feats["residx_atom28_to_atom23"],
        chain_feats["atom28_atom_exists"],
    )
    atom37_mask = torch.any(atom28_pos, axis=-1)
    return atom28_pos, atom37_mask, atom23_pos


def atom28_from_trans_rot(trans, rots, aatype, psi_torsions, chain_feats, res_mask=None):
    num_batch, num_res = trans.shape[:2]
    if res_mask is None:
        res_mask = torch.ones([*trans.shape[:-1]], device=trans.device)
    rigids = data_utils.create_rigid(rots, trans)
    psi_torsions = (
        psi_torsions if psi_torsions is not None else torch.zeros((num_batch, num_res, 9, 2), device=trans.device)
    )
    # aatype = torch.full((num_batch, num_res), 3, device=trans.device, dtype=torch.long)
    atom28, _, atom23 = compute_all_atom(rigids, psi_torsions, aatype, chain_feats)
    batch_atom28 = []
    batch_atom23 = []
    num_batch = res_mask.shape[0]
    for i in range(num_batch):
        batch_atom23.append(atom23[i])
        batch_atom28.append(atom28[i])
    return torch.stack(batch_atom28), torch.stack(batch_atom23)
