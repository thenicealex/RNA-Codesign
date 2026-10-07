"""Distance-conditioned RNA sequence-span distributions."""

import math
from pathlib import Path

import torch

DEFAULT_NUM_DISTANCE_BINS = 80
DEFAULT_NUM_SPANS = 256
DEFAULT_DISTANCE_MAX = 40.0
DEFAULT_DISTANCE_BIN_WIDTH = 0.5
INVALID_SPAN = -1


def _distance_bin_width(num_distance_bins: int, distance_max: float) -> float:
    if num_distance_bins <= 0:
        raise ValueError("num_distance_bins must be positive")
    if distance_max <= 0:
        raise ValueError("distance_max must be positive")
    return float(distance_max) / float(num_distance_bins)


def _validate_structure_inputs(
    d_map: torch.Tensor,
    l_map: torch.Tensor,
    node_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if d_map.ndim != 2 or d_map.shape[0] != d_map.shape[1]:
        raise ValueError("d_map must be a square [N, N] tensor")
    if l_map.shape != d_map.shape:
        raise ValueError("l_map must have the same shape as d_map")
    if not d_map.is_floating_point():
        raise TypeError("d_map must be a floating point tensor")
    if l_map.is_floating_point():
        raise TypeError("l_map must be an integer tensor")

    mask = None
    if node_mask is not None:
        if node_mask.ndim != 1 or node_mask.shape[0] != d_map.shape[0]:
            raise ValueError("node_mask must be a [N] tensor")
        mask = node_mask.to(device=d_map.device, dtype=torch.bool)

    n = d_map.shape[0]
    indices = torch.arange(n, device=d_map.device)
    expected = (indices[:, None] - indices[None, :]).abs()
    if mask is None:
        compare_mask = torch.ones_like(expected, dtype=torch.bool)
    else:
        compare_mask = mask[:, None] & mask[None, :]
    actual_spans = l_map.to(device=d_map.device).long()
    if not torch.equal(actual_spans[compare_mask], expected[compare_mask]):
        raise ValueError("l_map must equal abs(i - j) for non-padding positions")
    return mask


def build_structure_distribution(
    d_map: torch.Tensor,
    l_map: torch.Tensor,
    node_mask: torch.Tensor | None = None,
    num_distance_bins: int = DEFAULT_NUM_DISTANCE_BINS,
    num_spans: int = DEFAULT_NUM_SPANS,
    distance_max: float = DEFAULT_DISTANCE_MAX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map one RNA structure to ``P(span | distance_bin)``.

    Only upper-triangle non-padding pairs with finite distances in
    ``[0, distance_max)`` and spans in ``[1, num_spans)`` are counted.
    """
    if num_spans <= 1:
        raise ValueError("num_spans must be greater than 1")
    mask = _validate_structure_inputs(d_map, l_map, node_mask)
    bin_width = _distance_bin_width(num_distance_bins, distance_max)

    n = d_map.shape[0]
    device = d_map.device
    count_dtype = d_map.dtype

    upper = torch.triu(torch.ones((n, n), device=device, dtype=torch.bool), diagonal=1)
    valid = (
        upper
        & torch.isfinite(d_map)
        & (d_map >= 0)
        & (d_map < distance_max)
        & (l_map.to(device=device) >= 1)
        & (l_map.to(device=device) < num_spans)
    )
    if mask is not None:
        valid = valid & mask[:, None] & mask[None, :]

    distances = d_map[valid]
    spans = l_map.to(device=device).long()[valid]
    counts = torch.zeros(
        num_distance_bins * num_spans,
        dtype=count_dtype,
        device=device,
    )
    if distances.numel() > 0:
        distance_bins = torch.floor(distances / bin_width).long()
        linear_index = distance_bins * num_spans + spans
        counts.scatter_add_(
            0,
            linear_index,
            torch.ones(linear_index.shape, dtype=count_dtype, device=device),
        )
    counts = counts.reshape(num_distance_bins, num_spans)

    bin_pair_count = counts.sum(dim=1).long()
    valid_bin_mask = bin_pair_count > 0
    prob = torch.zeros_like(counts)
    if valid_bin_mask.any():
        prob[valid_bin_mask] = counts[valid_bin_mask] / bin_pair_count[valid_bin_mask].to(dtype=count_dtype)[:, None]
    prob[:, 0] = 0
    return prob, valid_bin_mask, bin_pair_count


def update_dataset_distribution(
    prob_sum: torch.Tensor,
    support_count: torch.Tensor,
    structure_prob: torch.Tensor,
    valid_bin_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate structure-equally weighted conditional distributions."""
    if prob_sum.ndim != 2:
        raise ValueError("prob_sum must have shape [K, S]")
    if support_count.shape != prob_sum.shape[:1]:
        raise ValueError("support_count must have shape [K]")

    if structure_prob.ndim == 2:
        structure_prob = structure_prob.unsqueeze(0)
        valid_bin_mask = valid_bin_mask.unsqueeze(0)
    elif structure_prob.ndim != 3:
        raise ValueError("structure_prob must have shape [K, S] or [B, K, S]")

    if structure_prob.shape[1:] != prob_sum.shape:
        raise ValueError("structure_prob shape must match prob_sum")
    if valid_bin_mask.shape != structure_prob.shape[:2]:
        raise ValueError("valid_bin_mask must have shape [K] or [B, K]")

    weights = valid_bin_mask.to(device=prob_sum.device, dtype=prob_sum.dtype)
    prob_sum += (structure_prob.to(device=prob_sum.device, dtype=prob_sum.dtype) * weights[:, :, None]).sum(dim=0)
    support_count += weights.sum(dim=0).to(dtype=support_count.dtype)
    return prob_sum, support_count


def finalize_dataset_distribution(
    prob_sum: torch.Tensor,
    support_count: torch.Tensor,
) -> torch.Tensor:
    """Return the masked mean ``P_dataset[distance_bin, span]``."""
    if prob_sum.ndim != 2:
        raise ValueError("prob_sum must have shape [K, S]")
    if support_count.shape != prob_sum.shape[:1]:
        raise ValueError("support_count must have shape [K]")

    dataset_prob = prob_sum / support_count.clamp_min(1).to(dtype=prob_sum.dtype)[:, None]
    dataset_prob = dataset_prob.clone()
    dataset_prob[support_count <= 0] = 0
    dataset_prob[:, 0] = 0
    return dataset_prob


def _nearest_supported_bins(
    row_ids: torch.Tensor,
    supported_bins: torch.Tensor,
) -> torch.Tensor:
    if supported_bins.numel() == 0:
        return torch.full_like(row_ids, -1)
    distances = (row_ids[:, None] - supported_bins[None, :]).abs()
    nearest = distances.argmin(dim=1)
    return supported_bins[nearest]


def load_distance_span_distribution(path_value: str | Path) -> dict:
    """Load and validate a saved distance-span distribution."""
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Distance-span distribution does not exist: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Distance-span distribution must be a torch-saved dictionary")

    required = {"dataset_prob", "support_count", "distance_bin_width", "distance_max"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Distance-span distribution is missing keys: {missing}")

    dataset_prob = payload["dataset_prob"]
    support_count = payload["support_count"]
    if not isinstance(dataset_prob, torch.Tensor) or dataset_prob.ndim != 2:
        raise ValueError("dataset_prob must be a [K, S] tensor")
    if not isinstance(support_count, torch.Tensor) or support_count.shape != dataset_prob.shape[:1]:
        raise ValueError("support_count must have shape [K]")
    if dataset_prob.shape[1] < 2:
        raise ValueError("dataset_prob must provide spans 0 and 1 or greater")
    if not torch.isfinite(dataset_prob).all() or (dataset_prob < 0).any():
        raise ValueError("dataset_prob must be finite and non-negative")
    if not torch.equal(dataset_prob[:, 0], torch.zeros_like(dataset_prob[:, 0])):
        raise ValueError("dataset_prob[:, 0] must be zero")

    distance_bin_width = float(payload["distance_bin_width"])
    distance_max = float(payload["distance_max"])
    if distance_bin_width <= 0 or distance_max <= 0:
        raise ValueError("Distance-span distribution metadata must be positive")
    if not math.isclose(
        distance_max,
        dataset_prob.shape[0] * distance_bin_width,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError("distance_max must equal K * distance_bin_width")

    return {
        "path": str(path),
        "dataset_prob": dataset_prob.float(),
        "distance_bin_width": distance_bin_width,
        "distance_max": distance_max,
    }


def span_probabilities(
    distance: float | torch.Tensor,
    dataset_prob: torch.Tensor,
    min_span: int = 1,
    max_span: int = DEFAULT_NUM_SPANS - 1,
    temperature: float = 1.0,
    distance_bin_width: float | None = None,
    distance_max: float | None = None,
) -> torch.Tensor:
    """Return ``P(span | distance)`` after range and temperature filtering.

    Invalid distances return all-zero rows.
    Empty distance bins fall back to the nearest bin with positive support after
    applying the requested span range.
    """
    if dataset_prob.ndim != 2:
        raise ValueError("dataset_prob must have shape [K, S]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    num_distance_bins, num_spans = dataset_prob.shape
    if distance_bin_width is None:
        if distance_max is None:
            distance_bin_width = DEFAULT_DISTANCE_BIN_WIDTH
        else:
            distance_bin_width = float(distance_max) / float(num_distance_bins)
    if distance_bin_width <= 0:
        raise ValueError("distance_bin_width must be positive")
    if distance_max is None:
        distance_max = num_distance_bins * distance_bin_width
    if distance_max <= 0:
        raise ValueError("distance_max must be positive")
    expected_distance_max = num_distance_bins * distance_bin_width
    if not math.isclose(
        float(distance_max),
        float(expected_distance_max),
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError("distance_max must equal dataset_prob.shape[0] * distance_bin_width")
    if max_span == DEFAULT_NUM_SPANS - 1:
        max_span = min(max_span, num_spans - 1)
    if min_span < 1 or max_span >= num_spans or min_span > max_span:
        raise ValueError("span range must satisfy 1 <= min_span <= max_span < num_spans")

    distance_tensor = torch.as_tensor(
        distance,
        dtype=dataset_prob.dtype,
        device=dataset_prob.device,
    )
    original_shape = distance_tensor.shape
    flat_distance = distance_tensor.reshape(-1)
    probabilities = torch.zeros(
        (flat_distance.shape[0], num_spans),
        dtype=dataset_prob.dtype,
        device=dataset_prob.device,
    )

    in_range = torch.isfinite(flat_distance) & (flat_distance >= 0) & (flat_distance < distance_max)
    if not in_range.any():
        return probabilities.reshape(*original_shape, num_spans)

    row_ids = torch.floor(flat_distance[in_range] / distance_bin_width).long()
    row_ids = row_ids.clamp(max=num_distance_bins - 1)
    span_mask = torch.zeros(num_spans, dtype=torch.bool, device=dataset_prob.device)
    span_mask[min_span : max_span + 1] = True
    supported_bins = torch.nonzero(
        (dataset_prob[:, span_mask] > 0).any(dim=1),
        as_tuple=False,
    ).flatten()
    selected_rows = _nearest_supported_bins(row_ids, supported_bins)
    has_row = selected_rows >= 0
    if not has_row.any():
        return probabilities.reshape(*original_shape, num_spans)

    candidate_probs = dataset_prob[selected_rows[has_row]].clone()
    candidate_probs[:, ~span_mask] = 0
    positive = candidate_probs > 0
    logits = torch.full_like(candidate_probs, -torch.inf)
    logits[positive] = torch.log(candidate_probs[positive]) / temperature
    normalized = torch.softmax(logits, dim=1)

    in_range_indices = torch.nonzero(in_range, as_tuple=False).flatten()
    probabilities[in_range_indices[has_row]] = normalized
    return probabilities.reshape(*original_shape, num_spans)


def sample_span(
    distance: float | torch.Tensor,
    dataset_prob: torch.Tensor,
    min_span: int = 1,
    max_span: int = DEFAULT_NUM_SPANS - 1,
    temperature: float = 1.0,
    distance_bin_width: float | None = None,
    distance_max: float | None = None,
) -> torch.Tensor:
    """Sample sequence span from ``P_dataset`` for each input distance.

    Invalid distances or unsupported span constraints return ``INVALID_SPAN``.
    """
    probabilities = span_probabilities(
        distance,
        dataset_prob,
        min_span=min_span,
        max_span=max_span,
        temperature=temperature,
        distance_bin_width=distance_bin_width,
        distance_max=distance_max,
    )
    original_shape = probabilities.shape[:-1]
    flat_probabilities = probabilities.reshape(-1, probabilities.shape[-1])
    valid = flat_probabilities.sum(dim=1) > 0
    sampled = torch.full(
        (flat_probabilities.shape[0],),
        INVALID_SPAN,
        dtype=torch.long,
        device=dataset_prob.device,
    )
    if valid.any():
        sampled[valid] = torch.multinomial(flat_probabilities[valid], num_samples=1).squeeze(1)
    return sampled.reshape(original_shape)


def build_distribution_payload(
    dataset_prob: torch.Tensor,
    support_count: torch.Tensor,
    distance_bin_width: float = DEFAULT_DISTANCE_BIN_WIDTH,
    distance_max: float | None = None,
    span_definition: str = "abs(i - j)",
) -> dict:
    """Build a self-describing payload for ``torch.save``."""
    if dataset_prob.ndim != 2:
        raise ValueError("dataset_prob must have shape [K, S]")
    if support_count.shape != dataset_prob.shape[:1]:
        raise ValueError("support_count must have shape [K]")
    if distance_bin_width <= 0:
        raise ValueError("distance_bin_width must be positive")

    num_distance_bins, num_spans = dataset_prob.shape
    if distance_max is None:
        distance_max = num_distance_bins * distance_bin_width
    distance_edges = torch.linspace(
        0.0,
        float(distance_max),
        steps=num_distance_bins + 1,
        dtype=dataset_prob.dtype,
        device=dataset_prob.device,
    )
    return {
        "dataset_prob": dataset_prob,
        "support_count": support_count,
        "distance_edges": distance_edges,
        "distance_bin_width": float(distance_bin_width),
        "distance_max": float(distance_max),
        "num_spans": int(num_spans),
        "span_definition": span_definition,
    }


def save_dataset_distribution(
    path: str | Path,
    dataset_prob: torch.Tensor,
    support_count: torch.Tensor,
    distance_bin_width: float = DEFAULT_DISTANCE_BIN_WIDTH,
    distance_max: float | None = None,
    span_definition: str = "abs(i - j)",
) -> None:
    """Save a distance-span distribution with index metadata."""
    payload = build_distribution_payload(
        dataset_prob=dataset_prob,
        support_count=support_count,
        distance_bin_width=distance_bin_width,
        distance_max=distance_max,
        span_definition=span_definition,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
