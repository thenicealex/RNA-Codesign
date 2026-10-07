"""Motif scaffolding dataset with RFdiffusion-style contig syntax.

Contig syntax (passed as ``inference.scaffolding.contigs``):

    "5-15/A10-25/30-40"

Each ``/``-separated token is one of:
    ``5-15``    scaffold region — generate 5-15 residues  (diffuse_mask=1)
    ``10``      scaffold region — generate exactly 10 residues
    ``A10-25``  motif region   — fix residues 10-25 from chain A of the input PDB
                                 (diffuse_mask=0); chain letter is case-insensitive
    ``auto``    scaffold region — sample its length from P(span | endpoint distance)
                 when ``distance_span_distribution`` is set
    ``auto:3-20`` sample from the distribution if configured, otherwise uniformly from 3-20
    ``0``       unsupported — scaffolding currently accepts one chain only

Usage in config::

    scaffolding:
      input_pdb: ./motifs/1zo3.pdb
      contigs: "5-15/A10-25/30-40"
      length: null          # null = sample each generated range freely
                            # "55-55" = exactly 55 residues
                            # "50-70" = total must be in [50, 70]
"""

import dataclasses
import logging

logger = logging.getLogger(__name__)
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from Bio import PDB
from torch.utils.data import Dataset

from data import data_transforms, parsers
from data.distance_span_distribution import load_distance_span_distribution, span_probabilities
from np import residue_constants
from utils import rigid_utils

# ---------------------------------------------------------------------------
# Segment types
# ---------------------------------------------------------------------------


@dataclass
class ScaffoldSegment:
    """Region to be generated (diffuse_mask=1)."""

    min_len: int
    max_len: int


@dataclass
class AutoLinkerSegment:
    """Generated linker with optional distance-conditioned length sampling."""

    min_len: int | None = None
    max_len: int | None = None
    distance: float | None = None


@dataclass
class TerminalSegment:
    """Optional generated 5' or 3' terminal region."""

    end: str
    min_len: int
    max_len: int


@dataclass
class MotifSegment:
    """Fixed residues from the input PDB (diffuse_mask=0)."""

    chain: str
    res_start: int  # PDB residue number, inclusive
    res_end: int  # PDB residue number, inclusive
    trans: torch.Tensor | None = field(default=None, repr=False)
    rotmats: torch.Tensor | None = field(default=None, repr=False)
    aatypes: torch.Tensor | None = field(default=None, repr=False)
    atom_positions: torch.Tensor | None = field(default=None, repr=False)
    atom_mask: torch.Tensor | None = field(default=None, repr=False)

    @property
    def length(self) -> int:
        if self.trans is not None:
            return self.trans.shape[0]
        return self.res_end - self.res_start + 1


# ---------------------------------------------------------------------------
# Contig parsing
# ---------------------------------------------------------------------------


def _parse_contig(contig_str: str) -> list:
    """Parse a contig string into a list of segments.

    >>> _parse_contig("5-15/A10-25/30-40")
    [ScaffoldSegment(5,15), MotifSegment('A',10,25,...), ScaffoldSegment(30,40)]
    """
    segments = []
    for part in contig_str.strip("[] \t").split("/"):
        part = part.strip()
        if not part:
            continue
        if part == "0":
            raise ValueError("Chain breaks are not supported; RNA scaffolding requires a single-chain contig")
        elif part.lower().startswith("auto"):
            if part.lower() == "auto":
                segments.append(AutoLinkerSegment())
                continue
            if not part.lower().startswith("auto:"):
                raise ValueError(f"Invalid auto linker token: {part}")
            bounds = part.split(":", 1)[1]
            try:
                if "-" in bounds:
                    lo, hi = map(int, bounds.split("-"))
                else:
                    lo = hi = int(bounds)
            except ValueError as exc:
                raise ValueError(f"Invalid auto linker length constraint: {part}") from exc
            if lo < 0 or hi < lo:
                raise ValueError(f"Invalid auto linker length constraint: {part}")
            segments.append(AutoLinkerSegment(min_len=lo, max_len=hi))
        elif part[0].isalpha():
            # Motif: A10-25 or A10
            chain = part[0].upper()
            nums = part[1:].split("-")
            start = int(nums[0])
            end = int(nums[1]) if len(nums) == 2 else start
            segments.append(MotifSegment(chain=chain, res_start=start, res_end=end))
        else:
            # Scaffold: 5-15 or 10
            if "-" in part:
                lo, hi = map(int, part.split("-"))
            else:
                lo = hi = int(part)
            segments.append(ScaffoldSegment(min_len=lo, max_len=hi))
    return segments


def _set_auto_linker_distances(segments: list) -> None:
    """Populate each auto linker's C1' distance from its fixed motif flanks."""
    c1_index = residue_constants.atom_order["C1'"]
    for index, segment in enumerate(segments):
        if not isinstance(segment, AutoLinkerSegment):
            continue
        if (
            index == 0
            or index == len(segments) - 1
            or not isinstance(segments[index - 1], MotifSegment)
            or not isinstance(segments[index + 1], MotifSegment)
        ):
            raise ValueError("Each auto linker must be directly flanked by two motif segments")
        left = segments[index - 1]
        right = segments[index + 1]
        if (
            left.atom_positions is None
            or right.atom_positions is None
            or left.atom_mask is None
            or right.atom_mask is None
        ):
            raise ValueError("Motif features must be loaded before resolving auto linkers")
        if not bool(left.atom_mask[-1, c1_index]) or not bool(right.atom_mask[0, c1_index]):
            raise ValueError("Auto linker flank is missing a C1' atom")
        left_c1 = left.atom_positions[-1, c1_index]
        right_c1 = right.atom_positions[0, c1_index]
        segment.distance = float(torch.linalg.vector_norm(left_c1 - right_c1))


def _parse_length_range(value, field_name: str) -> tuple[int, int]:
    parts = str(value).split("-")
    try:
        min_len, max_len = int(parts[0]), int(parts[-1])
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name}: {value}") from exc
    if min_len < 0 or max_len < min_len:
        raise ValueError(f"Invalid {field_name}: {value}")
    return min_len, max_len


def _add_terminal_segments(segments: list, cfg) -> list:
    """Add configured variable-length terminal regions around a contig."""
    result = list(segments)
    if getattr(cfg, "generate_5prime", False):
        bounds = _parse_length_range(
            getattr(cfg, "terminal_5prime_length", "0-10"),
            "terminal_5prime_length",
        )
        result.insert(0, TerminalSegment("5prime", *bounds))
    if getattr(cfg, "generate_3prime", False):
        bounds = _parse_length_range(
            getattr(cfg, "terminal_3prime_length", "0-10"),
            "terminal_3prime_length",
        )
        result.append(TerminalSegment("3prime", *bounds))
    return result


# ---------------------------------------------------------------------------
# PDB loading
# ---------------------------------------------------------------------------


def _center_atom_positions(
    atom_positions: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    """Center all-atom coordinates using the masked C1' center."""
    c1_index = residue_constants.atom_order["C1'"]
    c1_mask = atom_mask[:, c1_index].float()
    if torch.sum(c1_mask) < 1:
        raise ValueError("No valid C1' atoms in input PDB")
    c1_coords = atom_positions[:, c1_index]
    center = (c1_coords * c1_mask[:, None]).sum(dim=0) / (c1_mask.sum() + 1e-5)
    centered = atom_positions - center[None, None, :]
    return centered * atom_mask[..., None].float()


def _backbone_atom_mask_like(atom_mask: torch.Tensor) -> torch.Tensor:
    """Return an atom mask restricted to RNA backbone/sugar atoms."""
    backbone_atoms = [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "OP3",
    ]
    keep = torch.zeros(atom_mask.shape[-1], dtype=torch.bool, device=atom_mask.device)
    for atom_name in backbone_atoms:
        keep[residue_constants.atom_order[atom_name]] = True
    return atom_mask & keep


def _load_pdb_chain_features(pdb_path: str) -> dict:
    """Load PDB and return per-chain backbone features indexed by residue number.

    Returns:
        dict: {chain_id (str) →
                  {"res_nums": List[int],
                   "aatype":   LongTensor [N],
                   "trans":    FloatTensor [N, 3],
                   "rotmats":  FloatTensor [N, 3, 3]}}
    """
    pdb_parser = PDB.PDBParser(QUIET=True)
    structure = pdb_parser.get_structure("motif", pdb_path)

    chains = []
    for chain in structure.get_chains():
        residues = list(chain.get_residues())
        if residues:
            chains.append((chain, residues))
    if len(chains) != 1:
        raise ValueError(
            f"Expected exactly one chain in {pdb_path}, found {len(chains)}. Multi-chain structures are not supported."
        )

    chain_features = {}
    for chain, residues in chains:
        chain_id = chain.id.upper()
        res_nums = [res.id[1] for res in residues]

        # Extract atom-level features via existing parser
        chain_prot = parsers.process_chain(chain, chain_id=0)
        chain_dict = dataclasses.asdict(chain_prot)

        # Compute backbone frames: atom_positions → rigidgroups → trans/rotmats
        feats = {
            "aatype": torch.tensor(chain_dict["aatype"]).long(),
            "all_atom_positions": torch.tensor(chain_dict["atom_positions"]).double(),
            "all_atom_mask": torch.tensor(chain_dict["atom_mask"]).double(),
        }
        feats["all_atom_positions"] = _center_atom_positions(
            feats["all_atom_positions"],
            feats["all_atom_mask"],
        )
        feats = data_transforms.atom28_to_frames(feats)
        rigids = rigid_utils.Rigid.from_tensor_4x4(feats["rigidgroups_gt_frames"])[:, 0]
        chain_features[chain_id] = {
            "res_nums": res_nums,
            "aatype": feats["aatype"],
            "trans": rigids.get_trans().float(),
            "rotmats": rigids.get_rots().get_rot_mats().float(),
            "atom_positions": feats["all_atom_positions"].float(),
            "atom_mask": feats["all_atom_mask"].bool(),
        }

    return chain_features


def _fill_motif_features(segments: list, chain_features: dict) -> list:
    """Populate each MotifSegment with trans/rotmats/aatypes from the loaded PDB."""
    for seg in segments:
        if not isinstance(seg, MotifSegment):
            continue

        cid = seg.chain.upper()
        if cid not in chain_features:
            available = list(chain_features.keys())
            raise ValueError(f"Chain '{cid}' not found in PDB. Available: {available}")

        chain = chain_features[cid]
        indices = [i for i, rn in enumerate(chain["res_nums"]) if seg.res_start <= rn <= seg.res_end]
        if not indices:
            rn_min = min(chain["res_nums"])
            rn_max = max(chain["res_nums"])
            raise ValueError(
                f"No residues in range [{seg.res_start}, {seg.res_end}] for chain {cid} (available: {rn_min}-{rn_max})."
            )

        idx = torch.tensor(indices)
        seg.trans = chain["trans"][idx]
        seg.rotmats = chain["rotmats"][idx]
        seg.aatypes = chain["aatype"][idx]
        seg.atom_positions = chain["atom_positions"][idx]
        seg.atom_mask = chain["atom_mask"][idx]

    return segments


# ---------------------------------------------------------------------------
# Fixed-length feasibility
# ---------------------------------------------------------------------------


def _compute_length_configs(
    segments: list,
    length_str=None,
    generated_length_sums: Sequence[int] | None = None,
) -> list[dict]:
    """Return the fixed motif total when some generated-length sum is feasible.

    Args:
        segments: parsed contig segment list
        length_str: optional total-length constraint, e.g. "55-55" or "50-70"

    Returns:
        A single base config; generated lengths are sampled at item creation.
    """
    motif_total = sum(s.length for s in segments if isinstance(s, MotifSegment))

    # Parse total-length constraint
    if length_str:
        parts = str(length_str).split("-")
        len_min, len_max = int(parts[0]), int(parts[-1])
    else:
        len_min = len_max = None

    if len_min is not None:
        if generated_length_sums is None:
            feasible = len_min <= motif_total <= len_max
        else:
            feasible = any(len_min <= motif_total + generated_sum <= len_max for generated_sum in generated_length_sums)
        if not feasible:
            return []
    return [{"total": motif_total}]


def _build_auto_suffix_masses(
    distributions: Sequence[Sequence[tuple[int, float]]],
) -> list[dict[int, float]]:
    """Return suffix probabilities indexed by total linker length."""
    suffix_masses: list[dict[int, float]] = [dict() for _ in range(len(distributions) + 1)]
    suffix_masses[-1] = {0: 1.0}
    for index in range(len(distributions) - 1, -1, -1):
        masses: dict[int, float] = {}
        for length, probability in distributions[index]:
            for suffix_sum, suffix_probability in suffix_masses[index + 1].items():
                total = length + suffix_sum
                masses[total] = masses.get(total, 0.0) + probability * suffix_probability
        suffix_masses[index] = masses
    return suffix_masses


def _weighted_choice(options: Sequence[tuple[int, float]], rng: random.Random) -> int:
    total_weight = sum(weight for _, weight in options)
    if total_weight <= 0:
        raise ValueError("Cannot sample from an empty probability distribution")
    draw = rng.random() * total_weight
    cumulative = 0.0
    for value, weight in options:
        cumulative += weight
        if draw <= cumulative:
            return value
    return options[-1][0]


def _sample_auto_lengths(
    distributions: Sequence[Sequence[tuple[int, float]]],
    suffix_masses: Sequence[dict[int, float]],
    rng: random.Random,
    min_total: int | None = None,
    max_total: int | None = None,
) -> list[int]:
    """Sample linker lengths, conditioned on their total lying in a range."""
    valid_totals = [
        (total, mass)
        for total, mass in suffix_masses[0].items()
        if (min_total is None or total >= min_total) and (max_total is None or total <= max_total)
    ]
    if not valid_totals:
        raise ValueError("No auto linker lengths satisfy the total length constraint")

    remaining = _weighted_choice(valid_totals, rng)
    sampled = []
    for index, distribution in enumerate(distributions):
        choices = [
            (length, probability * suffix_masses[index + 1][remaining - length])
            for length, probability in distribution
            if remaining - length in suffix_masses[index + 1]
        ]
        length = _weighted_choice(choices, rng)
        sampled.append(length)
        remaining -= length
    return sampled


# ---------------------------------------------------------------------------
# Static plan and sample instantiation
# ---------------------------------------------------------------------------


GeneratedSegment = AutoLinkerSegment | TerminalSegment | ScaffoldSegment


def _is_generated_segment(segment) -> bool:
    return isinstance(segment, (AutoLinkerSegment, TerminalSegment, ScaffoldSegment))


def _build_length_distributions(
    variable_segments: Sequence[GeneratedSegment],
    distance_span_distribution: dict | None,
    temperature: float = 1.0,
) -> tuple[tuple[tuple[int, float], ...], ...]:
    distributions = []
    for segment in variable_segments:
        if isinstance(segment, AutoLinkerSegment) and distance_span_distribution is not None:
            dataset_prob = distance_span_distribution["dataset_prob"]
            min_span = (segment.min_len or 0) + 1
            requested_max_span = dataset_prob.shape[1] - 1 if segment.max_len is None else segment.max_len + 1
            max_span = min(requested_max_span, dataset_prob.shape[1] - 1)
            if min_span > max_span:
                raise ValueError("No distance-span probability remains after applying auto linker length constraints")
            probabilities = span_probabilities(
                segment.distance,
                dataset_prob,
                min_span=min_span,
                max_span=max_span,
                temperature=temperature,
                distance_bin_width=distance_span_distribution["distance_bin_width"],
                distance_max=distance_span_distribution["distance_max"],
            )
            distribution = [
                (span - 1, float(probability))
                for span, probability in enumerate(probabilities.tolist())
                if span >= 1 and probability > 0
            ]
            if not distribution:
                raise ValueError(
                    "No distance-span probability is available for the auto linker endpoint distance and length range"
                )
        else:
            if segment.min_len is None or segment.max_len is None:
                raise ValueError(
                    "Auto linkers without scaffolding.distance_span_distribution must specify a length range, "
                    "e.g. auto:3-30"
                )
            count = segment.max_len - segment.min_len + 1
            distribution = tuple((length, 1.0 / count) for length in range(segment.min_len, segment.max_len + 1))
        distributions.append(tuple(distribution))
    return tuple(distributions)


def _build_linker_metadata(
    variable_segments: Sequence[GeneratedSegment],
) -> tuple[torch.Tensor, torch.Tensor]:
    kinds = []
    distances = []
    for segment in variable_segments:
        if isinstance(segment, AutoLinkerSegment):
            kinds.append(0)
            distances.append(segment.distance if segment.distance is not None else float("nan"))
        elif isinstance(segment, TerminalSegment) and segment.end == "5prime":
            kinds.append(1)
            distances.append(float("nan"))
        elif isinstance(segment, TerminalSegment):
            kinds.append(2)
            distances.append(float("nan"))
        else:
            kinds.append(3)
            distances.append(float("nan"))
    return (
        torch.tensor(kinds, dtype=torch.long),
        torch.tensor(distances, dtype=torch.float32),
    )


@dataclass
class ScaffoldPlan:
    """Static motif, contig, and linker data shared by all sampled items."""

    pdb_stem: str
    segments: tuple
    length_distributions: tuple[tuple[tuple[int, float], ...], ...]
    suffix_masses: tuple[dict[int, float], ...]
    length_configs: tuple[dict, ...]
    length_range: tuple[int, int] | None
    linker_kinds: torch.Tensor
    linker_distances: torch.Tensor

    @classmethod
    def from_config(cls, cfg) -> "ScaffoldPlan":
        pdb_path = cfg.input_pdb
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"input_pdb not found: {pdb_path}")

        chain_features = _load_pdb_chain_features(pdb_path)
        logger.info("Loaded PDB %s — chains: %s", pdb_path, list(chain_features.keys()))

        segments = _add_terminal_segments(_parse_contig(cfg.contigs), cfg)
        segments = _fill_motif_features(segments, chain_features)
        distribution_path = getattr(cfg, "distance_span_distribution", None)
        distance_span_distribution = load_distance_span_distribution(distribution_path) if distribution_path else None
        if distance_span_distribution is not None:
            _set_auto_linker_distances(segments)

        variable_segments = tuple(segment for segment in segments if _is_generated_segment(segment))
        distributions = _build_length_distributions(
            variable_segments,
            distance_span_distribution,
            temperature=float(getattr(cfg, "distance_span_temperature", 1.0)),
        )
        suffix_masses = tuple(_build_auto_suffix_masses(distributions))

        length_value = getattr(cfg, "length", None)
        length_range = _parse_length_range(length_value, "length") if length_value is not None else None
        generated_length_sums = tuple(suffix_masses[0]) if variable_segments else None
        length_configs = tuple(
            _compute_length_configs(
                segments,
                length_value,
                generated_length_sums=generated_length_sums,
            )
        )
        if not length_configs:
            raise ValueError("No valid length configurations found. Check contigs ranges and length constraint.")

        motif_segments = [segment for segment in segments if isinstance(segment, MotifSegment)]
        logger.info(
            "Contig '%s' → %d motif segment(s), %d fixed residues",
            cfg.contigs,
            len(motif_segments),
            sum(segment.length for segment in motif_segments),
        )
        if variable_segments:
            logger.info(
                "%d base length configuration(s), %d variable generated region(s)",
                len(length_configs),
                len(variable_segments),
            )
        else:
            logger.info(
                "%d length configuration(s), total range: %d-%d",
                len(length_configs),
                length_configs[0]["total"],
                length_configs[-1]["total"],
            )

        linker_kinds, linker_distances = _build_linker_metadata(variable_segments)
        return cls(
            pdb_stem=Path(pdb_path).stem,
            segments=tuple(segments),
            length_distributions=distributions,
            suffix_masses=suffix_masses,
            length_configs=length_configs,
            length_range=length_range,
            linker_kinds=linker_kinds,
            linker_distances=linker_distances,
        )

    def instantiate(self, idx: int, seed: int, samples_per_length: int) -> dict:
        """Sample generated lengths and assemble one model-ready condition."""
        config = self.length_configs[idx // samples_per_length]
        generated_lengths = []
        if self.length_distributions:
            min_generated_total = max_generated_total = None
            if self.length_range is not None:
                min_generated_total = self.length_range[0] - config["total"]
                max_generated_total = self.length_range[1] - config["total"]
            generated_lengths = _sample_auto_lengths(
                self.length_distributions,
                self.suffix_masses,
                random.Random(seed + idx),
                min_total=min_generated_total,
                max_total=max_generated_total,
            )

        generated_iter = iter(generated_lengths)
        assigned_lengths = tuple(
            next(generated_iter) if _is_generated_segment(segment) else segment.length for segment in self.segments
        )
        total_length = sum(assigned_lengths)
        return _assemble_scaffold_sample(
            self,
            idx=idx,
            total_length=total_length,
            assigned_lengths=assigned_lengths,
            generated_lengths=generated_lengths,
        )


def _assemble_scaffold_sample(
    plan: ScaffoldPlan,
    *,
    idx: int,
    total_length: int,
    assigned_lengths: Sequence[int],
    generated_lengths: Sequence[int],
) -> dict:
    """Build tensors for one sampled length assignment."""
    trans_1 = torch.zeros(total_length, 3)
    rotmats_1 = torch.eye(3).unsqueeze(0).expand(total_length, -1, -1).clone()
    aatypes_1 = torch.zeros(total_length, dtype=torch.long)
    diffuse_mask = torch.ones(total_length)
    aatype_diffuse_mask = torch.ones(total_length)
    aatype_known_mask = torch.zeros(total_length)
    trans_mask = torch.zeros(total_length)
    rot_mask = torch.zeros(total_length)
    motif_mask = torch.zeros(total_length)
    atom28_1 = torch.zeros(total_length, 28, 3)
    atom28_fixed_mask = torch.zeros(total_length, 28, dtype=torch.bool)

    position = 0
    for segment, length in zip(plan.segments, assigned_lengths, strict=True):
        if length == 0:
            continue
        end = position + length
        if isinstance(segment, MotifSegment):
            trans_1[position:end] = segment.trans
            rotmats_1[position:end] = segment.rotmats
            aatypes_1[position:end] = segment.aatypes
            diffuse_mask[position:end] = 0.0
            aatype_diffuse_mask[position:end] = (segment.aatypes >= residue_constants.restype_num).float()
            aatype_known_mask[position:end] = (segment.aatypes < residue_constants.restype_num).float()
            trans_mask[position:end] = segment.atom_mask[:, residue_constants.atom_order["C1'"]]
            frame_indices = [residue_constants.atom_order[name] for name in ("C2'", "C1'", "O4'")]
            rot_mask[position:end] = torch.min(segment.atom_mask[:, frame_indices], dim=-1).values
            motif_mask[position:end] = 1.0
            atom28_1[position:end] = segment.atom_positions
            fixed_mask = segment.atom_mask.clone()
            unknown_residues = segment.aatypes >= residue_constants.restype_num
            if torch.any(unknown_residues):
                fixed_mask[unknown_residues] = _backbone_atom_mask_like(fixed_mask[unknown_residues])
            atom28_fixed_mask[position:end] = fixed_mask
        elif not _is_generated_segment(segment):
            raise TypeError(f"Unsupported contig segment: {type(segment).__name__}")
        position = end

    return {
        "trans_1": trans_1,
        "rotmats_1": rotmats_1,
        "aatypes_1": aatypes_1,
        "res_mask": torch.ones(total_length, dtype=torch.int),
        "diffuse_mask": diffuse_mask,
        "aatype_diffuse_mask": aatype_diffuse_mask,
        "aatype_known_mask": aatype_known_mask,
        "trans_mask": trans_mask,
        "rot_mask": rot_mask,
        "motif_mask": motif_mask,
        "atom28_1": atom28_1,
        "atom28_fixed_mask": atom28_fixed_mask,
        "chain_idx": torch.ones(total_length, dtype=torch.int),
        "res_idx": torch.arange(total_length, dtype=torch.float32),
        "gt_torsions": torch.zeros(total_length, 9, 2),
        "torsion_angles_mask": torch.zeros(total_length, 9),
        "csv_idx": torch.ones(1, dtype=torch.long) * idx,
        "num_res": torch.tensor(total_length),
        "pdb_name": f"{plan.pdb_stem}_scaffold_{idx}_len{total_length}",
        "linker_lengths": torch.tensor(generated_lengths, dtype=torch.long),
        "linker_distances": plan.linker_distances.clone(),
        "linker_kinds": plan.linker_kinds.clone(),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ScaffoldDataset(Dataset):
    """Thin dataset wrapper around a reusable :class:`ScaffoldPlan`."""

    def __init__(self, cfg):
        self.plan = ScaffoldPlan.from_config(cfg.scaffolding)
        self._seed = int(getattr(cfg, "seed", 0))
        self._samples_per_length = int(cfg.samples.samples_per_length)
        self.num_samples = len(self.plan.length_configs) * self._samples_per_length
        logger.info("Total samples: %d", self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.plan.instantiate(idx, self._seed, self._samples_per_length)
