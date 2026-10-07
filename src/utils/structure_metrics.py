"""Structure alignment used by flow sampling."""

import torch
from torch import Tensor
from torch_scatter import scatter, scatter_add


def _center_sparse_batch(coordinates: Tensor, batch_indices: Tensor) -> Tensor:
    means = scatter(coordinates, batch_indices, dim=0, reduce="mean")
    return coordinates - means[batch_indices]


@torch.no_grad()
def align_sparse_batch(
    mobile: Tensor,
    batch_indices: Tensor,
    fixed: Tensor,
    broadcast_fixed: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Align sparse batched coordinates and return centered coordinates."""
    if mobile.shape[0] != fixed.shape[0]:
        if broadcast_fixed:
            num_batches = int(torch.max(batch_indices) + 1)
            fixed = fixed.repeat(num_batches, 1)
        else:
            raise ValueError("Mismatch in sparse batch dimensions")

    mobile = _center_sparse_batch(mobile, batch_indices)
    fixed = _center_sparse_batch(fixed, batch_indices)
    covariance = scatter_add(
        mobile[:, None, :] * fixed[:, :, None],
        batch_indices,
        dim=0,
    )
    left, _, right_t = torch.linalg.svd(covariance)
    left_t = left.transpose(1, 2)
    right = right_t.transpose(1, 2)
    sign = torch.sign(torch.linalg.det(torch.bmm(right, left_t)))
    left_t[:, 2, :] = left_t[:, 2, :] * sign[:, None]
    rotations = torch.bmm(right, left_t)
    aligned_mobile = torch.bmm(mobile[:, None, :], rotations[batch_indices]).squeeze(1)
    return aligned_mobile, fixed, rotations


def batch_align_structures(pos_1, pos_2, mask=None):
    """Align dense [B, N, 3] structures through the sparse batch kernel."""
    if pos_1.shape != pos_2.shape:
        raise ValueError("pos_1 and pos_2 must have the same shape")
    if pos_1.ndim != 3:
        raise ValueError("Expected inputs with shape [B, N, 3]")

    num_batch = pos_1.shape[0]
    batch_indices = torch.arange(num_batch, device=pos_1.device)[:, None].expand(pos_1.shape[:2])
    flat_pos_1 = pos_1.reshape(-1, 3)
    flat_pos_2 = pos_2.reshape(-1, 3)
    flat_batch_indices = batch_indices.reshape(-1)
    if mask is None:
        aligned_1, aligned_2, rotations = align_sparse_batch(flat_pos_1, flat_batch_indices, flat_pos_2)
        return (
            aligned_1.reshape(num_batch, -1, 3),
            aligned_2.reshape(num_batch, -1, 3),
            rotations,
        )

    flat_mask = mask.reshape(-1).bool()
    masked_batch_indices = flat_batch_indices[flat_mask]
    _, _, rotations = align_sparse_batch(
        flat_pos_1[flat_mask],
        masked_batch_indices,
        flat_pos_2[flat_mask],
    )
    return torch.bmm(pos_1, rotations), pos_2, rotations

