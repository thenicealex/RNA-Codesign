import math

import torch


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(min_bin, max_bin, num_bins, device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size // 2, device=indices.device)
    pos_embedding_sin = torch.sin(indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))).to(
        indices.device
    )
    pos_embedding_cos = torch.cos(indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))).to(
        indices.device
    )
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    """Sinusoidal features with the frequency spacing used by existing checkpoints."""
    frequency_count = embedding_dim // 2
    log_step = -math.log(max_positions) / (frequency_count - 1)
    frequencies = torch.exp(torch.arange(frequency_count, device=timesteps.device, dtype=torch.float32) * log_step)
    phases = (timesteps * max_positions).float().unsqueeze(-1) * frequencies
    features = torch.cat((phases.sin(), phases.cos()), dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        features = torch.cat((features, features.new_zeros((len(timesteps), 1))), dim=-1)
    return features
