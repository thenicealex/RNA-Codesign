import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from data import data_transforms
from utils import data_utils, feats, so3_utils, structure_metrics

# ---------------------------------------------------------------------------
# Module-level noise / prior samplers
# ---------------------------------------------------------------------------


def _centered_gaussian(num_batch, num_res, device):
    """Sample (num_batch, num_res, 3) Gaussian noise with zero mean per batch."""
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdim=True)


def _uniform_so3(num_batch, num_res, device):
    """Sample (num_batch, num_res, 3, 3) rotation matrices uniformly from SO(3)."""
    return torch.tensor(
        Rotation.random(num_batch * num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)


def _masked_categorical(num_batch, num_res, device):
    """Return an all-MASK token tensor of shape (num_batch, num_res)."""
    return torch.full(
        (num_batch, num_res),
        data_utils.MASK_TOKEN_INDEX,
        device=device,
        dtype=torch.long,
    )


# ---------------------------------------------------------------------------
# Diffuse-mask blending helpers
# Blend noisy state (trans_t / rotmats_t / aatypes_t) with the clean target
# (trans_1 / rotmats_1 / aatypes_1) according to diffuse masks:
#   output = noisy * diffuse_mask + clean * (1 - diffuse_mask)
# Positions where diffuse_mask=0 (e.g. motif regions) are kept at ground truth.
# ---------------------------------------------------------------------------


def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])


def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return rotmats_t * diffuse_mask[..., None, None] + rotmats_1 * (1 - diffuse_mask[..., None, None])


def _aatypes_diffuse_mask(aatypes_t, aatypes_1, diffuse_mask):
    return torch.where(diffuse_mask.bool(), aatypes_t, aatypes_1).long()


@dataclass(frozen=True)
class SamplingCondition:
    """Optional initial state and conditioning for reverse sampling."""

    task: str = "codesign"
    trans_0: torch.Tensor | None = None
    rotmats_0: torch.Tensor | None = None
    aatypes_0: torch.Tensor | None = None
    trans_1: torch.Tensor | None = None
    rotmats_1: torch.Tensor | None = None
    aatypes_1: torch.Tensor | None = None
    res_mask: torch.Tensor | None = None
    rot_mask: torch.Tensor | None = None
    diffuse_mask: torch.Tensor | None = None
    aatype_diffuse_mask: torch.Tensor | None = None
    chain_idx: torch.Tensor | None = None
    res_idx: torch.Tensor | None = None
    fixed_atom28: torch.Tensor | None = None
    fixed_atom28_mask: torch.Tensor | None = None

    def __post_init__(self):
        if self.task not in {"codesign", "forward_folding", "inverse_folding"}:
            raise ValueError(f"Unknown sampling task {self.task}")


# ---------------------------------------------------------------------------
# Interpolant class
# ---------------------------------------------------------------------------


class Interpolant:
    """
    SE(3) x categorical flow-matching interpolant for RNA co-design.

    Handles three modalities jointly:
      - R3 translations (backbone C-alpha / P positions)
      - SO(3) rotations (backbone frames)
      - Categorical nucleotide types (masking or uniform interpolant)

    Training: corrupt_batch() adds noise at a random t ∈ [min_t, 1].
    Inference: sample() runs an Euler ODE from t=min_t to t=1.
    """

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._aatypes_cfg = cfg.aatypes
        self._sample_cfg = cfg.sampling
        self._igso3 = None  # lazy-loaded; see igso3 property

        # Masking uses [A,G,C,U,UNK,MASK]. UNK is never sampled as a clean
        # state; it is retained only in raw reference data.
        self.num_tokens = data_utils.NUM_MODEL_TOKENS

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def igso3(self):
        """Lazy-load the isotropic Gaussian distribution on SO(3)."""
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=".cache")
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        """Uniform t in [min_t, 1-min_t], shape (num_batch,)."""
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2 * self._cfg.min_t) + self._cfg.min_t

    # ------------------------------------------------------------------
    # Corruption (forward process): t=0 → t
    # ------------------------------------------------------------------

    def corrupt_batch(self, batch):
        """
        Add noise to a clean batch at a random timestep t.

        When codesign_separate_t is True, different t values are used for
        structure (SO3 / R3) and sequence (categorical) to simulate
        forward-folding and inverse-folding conditioning during training.

        Returns a noisy copy of the batch with keys:
            trans_t, rotmats_t, aatypes_t  — corrupted modalities
            so3_t, r3_t, cat_t              — per-batch timesteps fed to model
            trans_sc, aatypes_sc            — self-conditioning placeholders (zeros)
        """
        noisy_batch = copy.deepcopy(batch)

        trans_1 = batch["trans_1"]  # [B, N, 3], Angstrom
        rotmats_1 = batch["rotmats_1"]  # [B, N, 3, 3]
        aatypes_1 = batch["aatypes_1"]  # [B, N]
        res_mask = batch["res_mask"]  # [B, N]
        diffuse_mask = batch["diffuse_mask"]  # [B, N]
        aatype_diffuse_mask = batch.get("aatype_diffuse_mask", diffuse_mask)
        aatype_diffuse_mask = torch.maximum(
            aatype_diffuse_mask.to(dtype=res_mask.dtype),
            (aatypes_1 == data_utils.UNK_TOKEN_INDEX).to(res_mask.dtype) * res_mask,
        )
        noisy_batch["aatype_diffuse_mask"] = aatype_diffuse_mask
        target_rot_mask = batch.get("rot_mask", res_mask)
        target_trans_mask = batch.get("trans_mask", res_mask)
        num_batch, _N = diffuse_mask.shape

        if self._cfg.codesign_separate_t:
            # Sample a task selector u ∈ [0, 1) per batch element, then assign
            # each sample to one of three modes:
            #   forward folding  (u < fwd_prop):           t_struct=rand, t_seq=1
            #   inverse folding  (fwd_prop ≤ u < fwd+inv): t_struct=1,    t_seq=rand
            #   codesign         (u ≥ fwd+inv):            t_struct=rand, t_seq=rand
            u = torch.rand((num_batch,), device=self._device)
            fwd_prop = self._cfg.codesign_forward_fold_prop
            inv_prop = self._cfg.codesign_inverse_fold_prop

            forward_fold_mask = (u < fwd_prop).float()
            inverse_fold_mask = ((u >= fwd_prop) & (u < fwd_prop + inv_prop)).float()

            normal_structure_t = self.sample_t(num_batch)
            normal_cat_t = self.sample_t(num_batch)
            ones = torch.ones((num_batch,), device=self._device)

            # Forward folding: fix sequence (t_seq=1), diffuse structure
            cat_t = forward_fold_mask * ones + (1 - forward_fold_mask) * normal_cat_t
            # Inverse folding: fix structure (t_struct=1), diffuse sequence
            structure_t = inverse_fold_mask * ones + (1 - inverse_fold_mask) * normal_structure_t

            so3_t = structure_t[:, None]
            r3_t = structure_t[:, None]
            cat_t = cat_t[:, None]
        else:
            # Shared t for all modalities
            t = self.sample_t(num_batch)[:, None]
            so3_t = r3_t = cat_t = t

        noisy_batch["so3_t"] = so3_t
        noisy_batch["r3_t"] = r3_t
        noisy_batch["cat_t"] = cat_t

        # Apply per-modality corruption
        if self._trans_cfg.corrupt:
            trans_t = self._corrupt_trans(
                trans_1,
                r3_t,
                res_mask,
                diffuse_mask,
                target_trans_mask=target_trans_mask,
            )
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError("NaN in trans_t during corruption")
        noisy_batch["trans_t"] = trans_t

        if self._rots_cfg.corrupt:
            rotmats_t = self._corrupt_rotmats(rotmats_1, so3_t, res_mask, diffuse_mask)
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError("NaN in rotmats_t during corruption")
        noisy_batch["rotmats_t"] = rotmats_t

        if self._aatypes_cfg.corrupt:
            aatypes_t = self._corrupt_aatypes(aatypes_1, cat_t, res_mask, aatype_diffuse_mask)
        else:
            aatypes_t = aatypes_1
        noisy_batch["aatypes_t"] = aatypes_t
        # During training, missing target frames remain sequence-only nodes and
        # are excluded from IPA/geometry edges. Reverse sampling starts from a
        # valid random frame, so sample() activates generated positions instead.
        noisy_batch["frame_mask"] = target_rot_mask * res_mask

        # Self-conditioning placeholders (filled in by model during inference)
        noisy_batch["trans_sc"] = torch.zeros_like(trans_1)
        noisy_batch["aatypes_sc"] = torch.zeros_like(aatypes_1)[..., None].repeat(1, 1, self.num_tokens)
        return noisy_batch

    # ------------------------------------------------------------------
    # Inference (reverse process): Euler ODE from t=min_t to t=1
    # ------------------------------------------------------------------

    def sample(
        self,
        num_batch,
        num_res,
        model,
        condition=None,
        *,
        num_timesteps=None,
        t_nn=None,
        separate_t=False,
        return_aux=False,
        return_trajectories=True,
    ):
        """
        Run the reverse Euler ODE from t=min_t to t=1.

        Args:
            num_batch, num_res: Batch and sequence dimensions.
            model: Callable that takes a feature dict and returns predictions.
            num_timesteps: Number of integration steps (default from config).
            condition: Initial state, conditioning targets, masks, indices, and task.
            t_nn: Optional learned time-embedding; if None uses raw t.
            separate_t: Use task-specific t values matching corrupt_batch logic.
            return_aux: Also return terminal model embeddings and confidence logits.
            return_trajectories: Keep all Euler-step atom trajectories. When false,
                return only the terminal frame for each trajectory.

        Returns:
            noisy_traj: List of (atom28, atom23, aatypes) at every noisy step.
            x0_traj:    List of (atom28, atom23, aatypes) from model x0-predictions.
            aux:        Optional terminal model outputs when return_aux is true.
        """
        condition = condition or SamplingCondition()
        trans_0 = condition.trans_0
        rotmats_0 = condition.rotmats_0
        aatypes_0 = condition.aatypes_0
        trans_1 = condition.trans_1
        rotmats_1 = condition.rotmats_1
        aatypes_1 = condition.aatypes_1
        res_mask = condition.res_mask
        target_rot_mask = condition.rot_mask
        diffuse_mask = condition.diffuse_mask
        aatype_diffuse_mask = condition.aatype_diffuse_mask
        chain_idx = condition.chain_idx
        res_idx = condition.res_idx
        fixed_atom28 = condition.fixed_atom28
        fixed_atom28_mask = condition.fixed_atom28_mask
        forward_folding = condition.task == "forward_folding"
        inverse_folding = condition.task == "inverse_folding"

        expected_mask_shape = (num_batch, num_res)
        if res_mask is None:
            res_mask = torch.ones(expected_mask_shape, device=self._device)
        else:
            if res_mask.shape != expected_mask_shape:
                raise ValueError(f"res_mask shape {tuple(res_mask.shape)} != {expected_mask_shape}")
            res_mask = res_mask.to(device=self._device, dtype=torch.float32)
        if target_rot_mask is None:
            target_rot_mask = res_mask
        else:
            if target_rot_mask.shape != expected_mask_shape:
                raise ValueError(f"rot_mask shape {tuple(target_rot_mask.shape)} != {expected_mask_shape}")
            target_rot_mask = target_rot_mask.to(device=self._device, dtype=res_mask.dtype) * res_mask

        # ---- Initialize prior samples ----
        if trans_0 is None:
            trans_0 = _centered_gaussian(num_batch, num_res, self._device) * data_utils.NM_TO_ANG_SCALE
        if rotmats_0 is None:
            rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        if aatypes_0 is None:
            if self._aatypes_cfg.interpolant_type == "masking":
                aatypes_0 = _masked_categorical(num_batch, num_res, self._device)
            elif self._aatypes_cfg.interpolant_type == "uniform":
                aatypes_0 = torch.randint_like(
                    res_mask,
                    low=0,
                    high=data_utils.NUM_TOKENS,
                    dtype=torch.long,
                )
            else:
                raise ValueError(f"Unknown aatypes interpolant type {self._aatypes_cfg.interpolant_type}")

        # ---- Default index and mask tensors ----
        if res_idx is None:
            res_idx = torch.arange(num_res, device=self._device, dtype=torch.float32)[None].repeat(num_batch, 1)
        if chain_idx is None:
            chain_idx = res_mask
        if diffuse_mask is None:
            diffuse_mask = res_mask
        else:
            if diffuse_mask.shape != expected_mask_shape:
                raise ValueError(f"diffuse_mask shape {tuple(diffuse_mask.shape)} != {expected_mask_shape}")
            diffuse_mask = diffuse_mask.to(device=self._device, dtype=res_mask.dtype) * res_mask
        if aatype_diffuse_mask is None:
            aatype_diffuse_mask = diffuse_mask
        else:
            if aatype_diffuse_mask.shape != expected_mask_shape:
                raise ValueError(
                    f"aatype_diffuse_mask shape {tuple(aatype_diffuse_mask.shape)} != {expected_mask_shape}"
                )
            aatype_diffuse_mask = aatype_diffuse_mask.to(device=self._device, dtype=res_mask.dtype) * res_mask
        if aatypes_1 is not None:
            # Unknown reference identities cannot be fixed sequence conditions.
            aatype_diffuse_mask = torch.maximum(
                aatype_diffuse_mask,
                (aatypes_1.to(self._device) == data_utils.UNK_TOKEN_INDEX).to(res_mask.dtype) * res_mask,
            )

        # ---- Conditioning ground-truth targets (used by folding tasks) ----
        if trans_1 is None:
            trans_1 = torch.zeros(num_batch, num_res, 3, device=self._device)
        if rotmats_1 is None:
            rotmats_1 = torch.eye(3, device=self._device)[None, None].repeat(num_batch, num_res, 1, 1)
        if aatypes_1 is None:
            aatypes_1 = torch.zeros((num_batch, num_res), device=self._device).long()

        # One-hot encoding of target sequence (used for self-conditioning)
        logits_1 = F.one_hot(aatypes_1, num_classes=self.num_tokens).float()

        # ---- Task-specific overrides for prior ----
        # Forward folding: sequence is known, so start from the true sequence.
        if forward_folding:
            assert self._aatypes_cfg.noise == 0
            if separate_t:
                aatypes_0 = _aatypes_diffuse_mask(aatypes_0, aatypes_1, aatype_diffuse_mask)
        # Inverse folding: structure is known, so start from the true frames.
        if inverse_folding:
            if separate_t:
                trans_0 = trans_1
                rotmats_0 = rotmats_1

        # ---- Assemble the shared batch dict (mutated in-place each step) ----
        sampling_frame_mask = (
            target_rot_mask if inverse_folding else torch.maximum(diffuse_mask, target_rot_mask) * res_mask
        )
        batch = {
            "res_mask": res_mask,
            "diffuse_mask": diffuse_mask,
            "aatype_diffuse_mask": aatype_diffuse_mask,
            "frame_mask": sampling_frame_mask,
            "chain_idx": chain_idx,
            "res_idx": res_idx,
            "trans_sc": torch.zeros(num_batch, num_res, 3, device=self._device),
            "aatypes_sc": torch.zeros(num_batch, num_res, self.num_tokens, device=self._device),
        }

        # ---- Inner helpers ----

        def trans_rot_to_atom28(x, y, aatype, psi_torsions=None):
            """Convert backbone frames (trans, rotmats) to atom28/atom23 coordinates."""
            # Structural templates exist only for A/G/C/U. Intermediate UNK or
            # MASK tokens use an A template for visualization; this never changes
            # the categorical state or creates a supervision target.
            structural_aatype = torch.where(
                aatype < data_utils.NUM_TOKENS,
                aatype,
                torch.zeros_like(aatype),
            ).long()
            chain_feats = data_transforms.make_atom23_masks({"aatype": structural_aatype})
            atom28_pos, _ = feats.atom28_from_trans_rot(x, y, structural_aatype, psi_torsions, chain_feats, res_mask)
            if fixed_atom28 is not None and fixed_atom28_mask is not None:
                atom28_pos = torch.where(
                    fixed_atom28_mask[..., None].bool(),
                    fixed_atom28,
                    atom28_pos,
                )
            return atom28_pos

        def set_batch_model_inputs(t, trans_t, rotmats_t, aatypes_t):
            """Populate the batch dict with noisy states and timestep scalars."""
            # Use noisy modality if corruption is enabled; otherwise use clean target.
            batch["trans_t"] = trans_t if self._trans_cfg.corrupt else trans_1
            batch["rotmats_t"] = rotmats_t if self._rots_cfg.corrupt else rotmats_1
            batch["aatypes_t"] = aatypes_t if self._aatypes_cfg.corrupt else aatypes_1

            # Timestep encoding
            if t_nn is not None:
                batch["r3_t"], batch["so3_t"], batch["cat_t"] = torch.split(t_nn(t), -1)
            else:
                batch["so3_t"] = self.rot_sample_kappa(t) if self._cfg.provide_kappa else t
                batch["r3_t"] = t
                batch["cat_t"] = t

            # Override t for task-specific modes (separate_t training)
            if forward_folding and separate_t:
                # Sequence is always at t=1 (fully revealed) during forward folding
                batch["cat_t"] = (1 - self._cfg.min_t) * torch.ones_like(batch["cat_t"])
            if inverse_folding and separate_t:
                # Structure is always at t=1 (fully revealed) during inverse folding
                batch["r3_t"] = (1 - self._cfg.min_t) * torch.ones_like(batch["r3_t"])
                batch["so3_t"] = (1 - self._cfg.min_t) * torch.ones_like(batch["so3_t"])

        # ---- ODE integration: Euler steps from t=min_t to t=1 ----
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)

        # Initialize trajectories at t=min_t.
        # Apply diffuse_mask so motif positions (diffuse_mask=0) start from ground
        # truth rather than noise — consistent with how corrupt_batch behaves.
        trans_0 = _trans_diffuse_mask(trans_0, trans_1, diffuse_mask)
        rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_1, diffuse_mask)
        aatypes_0 = _aatypes_diffuse_mask(aatypes_0, aatypes_1, aatype_diffuse_mask)
        trans_t_1, rotmats_t_1, aatypes_t_1 = trans_0, rotmats_0, aatypes_0

        noisy_traj = []
        clean_traj = []
        if return_trajectories:
            noisy_traj.append((trans_rot_to_atom28(trans_t_1, rotmats_t_1, aatypes_t_1), aatypes_t_1.detach().cpu()))

        t_1 = ts[0]
        for t_2 in ts[1:]:
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            set_batch_model_inputs(t, trans_t_1, rotmats_t_1, aatypes_t_1)
            d_t = t_2 - t_1

            with torch.no_grad():
                model_out = model(batch)

            # Unpack model x0-predictions
            pred_trans_1 = model_out["pred_trans"]
            pred_rotmats_1 = model_out["pred_rotmats"]
            pred_aatypes_1 = model_out["pred_aatypes"]
            pred_torsion_1 = model_out["pred_torsions"]
            pred_logits_1 = model_out["pred_logits"]

            if return_trajectories:
                atom28_pos_clean = trans_rot_to_atom28(pred_trans_1, pred_rotmats_1, pred_aatypes_1, pred_torsion_1)
                clean_traj.append((atom28_pos_clean, pred_aatypes_1.detach().cpu()))

            # Override predictions with conditioning targets for folding tasks
            if forward_folding:
                pred_logits_1 = torch.where(
                    aatype_diffuse_mask[..., None].bool(),
                    pred_logits_1,
                    100.0 * logits_1,
                )
            if inverse_folding:
                pred_trans_1 = trans_1
                pred_rotmats_1 = rotmats_1

            # Self-conditioning: feed current predictions back as context
            if self._cfg.self_condition:
                batch["trans_sc"] = _trans_diffuse_mask(pred_trans_1, trans_1, diffuse_mask)
                batch["aatypes_sc"] = _trans_diffuse_mask(
                    pred_logits_1,
                    logits_1,
                    aatype_diffuse_mask,
                )

            # Euler step for each modality
            trans_t_2 = self._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)

            use_purity = self._aatypes_cfg.do_purity and (t_1 >= self._aatypes_cfg.purity_switch_t)
            if use_purity:
                aatypes_t_2 = self._aatypes_euler_step_purity(d_t, t_1, pred_logits_1, aatypes_t_1)
            else:
                aatypes_t_2 = self._aatypes_euler_step(d_t, t_1, pred_logits_1, aatypes_t_1)

            # Pin fixed regions (diffuse_mask=0) to ground truth
            trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_1, diffuse_mask)
            rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_1, diffuse_mask)
            aatypes_t_2 = _aatypes_diffuse_mask(aatypes_t_2, aatypes_1, aatype_diffuse_mask)

            trans_t_1, rotmats_t_1, aatypes_t_1 = trans_t_2, rotmats_t_2, aatypes_t_2
            if return_trajectories:
                atom28_pos_noisy = trans_rot_to_atom28(trans_t_2, rotmats_t_2, aatypes_t_2, pred_torsion_1)
                noisy_traj.append((atom28_pos_noisy, aatypes_t_2.detach().cpu()))

            t_1 = t_2

        # ---- Final model call at t=1 to obtain the terminal x0-prediction ----
        t = torch.ones((num_batch, 1), device=self._device) * ts[-1]
        set_batch_model_inputs(t, trans_t_1, rotmats_t_1, aatypes_t_1)
        with torch.no_grad():
            model_out = model(batch)

        pred_trans_1 = model_out["pred_trans"]
        pred_rotmats_1 = model_out["pred_rotmats"]
        pred_aatypes_1 = model_out["pred_aatypes"]
        pred_torsion_1 = model_out["pred_torsions"]

        if inverse_folding:
            pred_trans_1 = trans_1
            pred_rotmats_1 = rotmats_1

        # Fixed residues must remain exact in the returned terminal prediction,
        # not only in the evolving noisy state.
        pred_trans_1 = _trans_diffuse_mask(pred_trans_1, trans_1, diffuse_mask)
        pred_rotmats_1 = _rots_diffuse_mask(pred_rotmats_1, rotmats_1, diffuse_mask)
        pred_aatypes_1 = _aatypes_diffuse_mask(pred_aatypes_1, aatypes_1, aatype_diffuse_mask)

        atom28_pos_final = trans_rot_to_atom28(pred_trans_1, pred_rotmats_1, pred_aatypes_1, pred_torsion_1)
        terminal_frame = (atom28_pos_final, pred_aatypes_1.detach().cpu())
        clean_traj.append(terminal_frame)
        noisy_traj.append(terminal_frame)

        if return_aux:
            return (
                noisy_traj,
                clean_traj,
                {
                    "node_embed": model_out.get("node_embed"),
                },
            )
        return noisy_traj, clean_traj

    # ------------------------------------------------------------------
    # Forward-process corruption helpers
    # ------------------------------------------------------------------

    def _corrupt_trans(self, trans_1, t, res_mask, diffuse_mask, target_trans_mask=None):
        """Interpolate translations from Gaussian noise to trans_1 at time t."""
        trans_0 = _centered_gaussian(*res_mask.shape, self._device) * data_utils.NM_TO_ANG_SCALE
        if self._trans_cfg.batch_ot:
            # Align noise to ground truth via optimal transport to reduce crossing paths.
            ot_mask = diffuse_mask if target_trans_mask is None else diffuse_mask * target_trans_mask
            if torch.all(ot_mask.sum(dim=-1) >= 3):
                trans_0 = self._batch_ot(trans_0, trans_1, ot_mask)

        if self._trans_cfg.train_schedule == "cosine":
            # Cosine schedule concentrates training steps in the informative mid-t range.
            t_cos = 1 - torch.cos(t * torch.pi / 2)
            trans_t = (1 - t_cos[..., None]) * trans_0 + t_cos[..., None] * trans_1
        elif self._trans_cfg.train_schedule == "linear":
            trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        else:
            raise ValueError(f"Unknown trans schedule {self._trans_cfg.train_schedule}")

        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * res_mask[..., None]

    def _batch_ot(self, trans_0, trans_1, res_mask):
        """
        Batch optimal transport: reorder noise samples to minimize transport cost.

        Aligns every (noise, target) pair, computes a cost matrix, then solves
        the linear assignment problem to find the permutation of noise samples
        that minimises total displacement.
        """
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(torch.ones(num_batch, num_batch))

        # Align all (noise, target) combinations
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = structure_metrics.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        )
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)

        # Cost = mean L2 distance over residues after alignment
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1) / torch.sum(
            batch_mask, dim=-1
        )

        noise_perm, gt_perm = linear_sum_assignment(data_utils.to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]

    def _corrupt_rotmats(self, rotmats_1, t, res_mask, diffuse_mask):
        """Interpolate rotations from uniform-SO3 noise to rotmats_1 via geodesic."""
        num_batch, num_res = res_mask.shape

        # Sample a large-sigma IGSO3 perturbation as the prior rotation
        noisy_rotmats = (
            self.igso3.sample(torch.tensor([1.5]), num_batch * num_res)
            .to(self._device)
            .reshape(num_batch, num_res, 3, 3)
        )

        # Compose: rotmats_0 = rotmats_1 @ noisy_rotmats (right-multiply)
        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)

        so3_schedule = self._rots_cfg.train_schedule
        if so3_schedule == "cosine":
            so3_t = 1 - torch.cos(t * torch.pi / 2)
        elif so3_schedule == "exp":
            so3_t = 1 - torch.exp(-t * self._rots_cfg.exp_rate)
        elif so3_schedule == "linear":
            so3_t = t
        else:
            raise ValueError(f"Invalid schedule: {so3_schedule}")

        rotmats_t = so3_utils.geodesic_t(so3_t[..., None], rotmats_1, rotmats_0)

        # Pad masked positions with identity
        identity = torch.eye(3, device=self._device)
        rotmats_t = rotmats_t * res_mask[..., None, None] + identity[None, None] * (1 - res_mask[..., None, None])
        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)

    def _corrupt_aatypes(self, aatypes_1, t, res_mask, diffuse_mask):
        """
        Corrupt nucleotide types at time t.

        masking:  Each token is independently replaced by MASK with prob (1-t).
        uniform:  Each token is independently replaced by a random token with prob (1-t).
        """
        num_batch, num_res = res_mask.shape
        assert aatypes_1.shape == (num_batch, num_res)
        assert t.shape == (num_batch, 1)

        u = torch.rand(num_batch, num_res, device=self._device)
        aatypes_t = aatypes_1.clone()
        corruption_mask = u < (1 - t)  # positions to corrupt, shape (B, N)

        if self._aatypes_cfg.interpolant_type == "masking":
            aatypes_t[corruption_mask] = data_utils.MASK_TOKEN_INDEX
        elif self._aatypes_cfg.interpolant_type == "uniform":
            uniform_sample = torch.randint_like(aatypes_t, low=0, high=data_utils.NUM_TOKENS)
            aatypes_t[corruption_mask] = uniform_sample[corruption_mask]
        else:
            raise ValueError(f"Unknown aatypes interpolant type {self._aatypes_cfg.interpolant_type}")

        # Apply residue mask: non-existent positions become MASK
        # UNK is missing supervision, not a clean flow endpoint. Present it to
        # the model as MASK at every t and exclude it from CE through the label mask.
        aatypes_t = torch.where(
            aatypes_1 == data_utils.UNK_TOKEN_INDEX,
            torch.full_like(aatypes_t, data_utils.MASK_TOKEN_INDEX),
            aatypes_t,
        )
        aatypes_t = aatypes_t * res_mask + data_utils.MASK_TOKEN_INDEX * (1 - res_mask)
        return _aatypes_diffuse_mask(aatypes_t, aatypes_1, diffuse_mask)

    # ------------------------------------------------------------------
    # Reverse-process Euler step helpers
    # ------------------------------------------------------------------

    def rot_sample_kappa(self, t):
        """Map linear t to the schedule-specific kappa used for SO(3) sampling."""
        if self._rots_cfg.sample_schedule == "cosine":
            return 1 - torch.cos(t * torch.pi / 2)
        elif self._rots_cfg.sample_schedule == "exp":
            return 1 - torch.exp(-t * self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == "linear":
            return t
        else:
            raise ValueError(f"Invalid schedule: {self._rots_cfg.sample_schedule}")

    def _trans_vector_field(self, t, trans_1, trans_t):
        """
        Compute the R3 velocity field v(t, x_t) = dx/dt pointing toward trans_1.

        linear:  v = (x1 - xt) / (1 - t)
        cosine:  v = (x1 - xt) / (1 - cos(π t/2)) * sin(π t/2) * π/2
        vpsde:   Score-based SDE drift; see VP-SDE formulation in the literature.
        """
        if self._trans_cfg.sample_schedule == "cosine":
            t_cos = 1 - torch.cos(t * torch.pi / 2)
            dt_cos_dt = torch.sin(t * torch.pi / 2) * (torch.pi / 2)
            return (trans_1 - trans_t) / (1 - t_cos + 1e-8) * dt_cos_dt
        elif self._trans_cfg.sample_schedule == "linear":
            return (trans_1 - trans_t) / (1 - t)
        elif self._trans_cfg.sample_schedule == "vpsde":
            bmin = self._trans_cfg.vpsde_bmin
            bmax = self._trans_cfg.vpsde_bmax
            bt = bmin + (bmax - bmin) * (1 - t)
            alpha_t = torch.exp(-bmin * (1 - t) - 0.5 * (1 - t) ** 2 * (bmax - bmin))
            return 0.5 * bt * trans_t + 0.5 * bt * (torch.sqrt(alpha_t) * trans_1 - trans_t) / (1 - alpha_t)
        else:
            raise ValueError(f"Invalid sample schedule: {self._trans_cfg.sample_schedule}")

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        """Euler step for R3 translations."""
        assert d_t >= 0
        return trans_t + self._trans_vector_field(t, trans_1, trans_t) * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        """
        Euler step on SO(3) via geodesic interpolation.

        scaling maps d_t to the effective step size in the chosen schedule.
        """
        if self._rots_cfg.sample_schedule == "cosine":
            scaling = torch.sin(t * torch.pi / 2) * (torch.pi / 2) / (1 - t + 1e-8)
        elif self._rots_cfg.sample_schedule == "linear":
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == "exp":
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(f"Unknown sample schedule {self._rots_cfg.sample_schedule}")

        return so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)

    def _regularize_step_probs(self, step_probs, aatypes_t):
        """
        Ensure step_probs is a valid probability distribution over tokens.

        Clamps values to [0, 1], zeros out the probability of staying at the
        current token, then sets that slot to (1 - sum of all other probs).
        """
        batch_size, num_res, _S = step_probs.shape
        device = step_probs.device
        assert aatypes_t.shape == (batch_size, num_res)

        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

        # Zero out prob of transitioning to the current token
        zeros = torch.zeros(batch_size, num_res, 1, device=device)
        step_probs.scatter_(dim=2, index=aatypes_t.long().unsqueeze(-1), src=zeros)

        # Fill remaining probability mass back to current token
        remaining = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(0.0, 1.0)
        step_probs.scatter_(dim=2, index=aatypes_t.long().unsqueeze(-1), src=remaining)

        return torch.clamp(step_probs, min=0.0, max=1.0)

    def _aatypes_euler_step(self, d_t, t, logits_1, aatypes_t):
        """
        Stochastic Euler step for discrete sequence (masking or uniform interpolant).

        Computes per-token transition probabilities for one step of the reverse
        CTMC (continuous-time Markov chain), then samples the next token.

        masking interpolant (S=6):
            Each masked position transitions to a real token at rate
            proportional to the model's predicted p(x1) and the noise level.

        uniform interpolant (S=4):
            Each position may transition to any token; the noise term adds
            a small probability of transitioning to the current token.
        """
        batch_size, num_res, S = logits_1.shape
        assert aatypes_t.shape == (batch_size, num_res)

        if self._aatypes_cfg.interpolant_type == "masking":
            assert S == data_utils.NUM_MODEL_TOKENS
            device = logits_1.device

            # Neither UNK nor MASK is a clean output token.
            logits_1[:, :, data_utils.UNK_TOKEN_INDEX] = -1e9
            logits_1[:, :, data_utils.MASK_TOKEN_INDEX] = -1e9
            pt_x1_probs = F.softmax(logits_1 / self._aatypes_cfg.temp, dim=-1)

            # Indicator: which positions are currently masked?
            aatypes_t_is_mask = (aatypes_t == data_utils.MASK_TOKEN_INDEX).float().unsqueeze(-1)
            mask_one_hot = torch.zeros((S,), device=device)
            mask_one_hot[data_utils.MASK_TOKEN_INDEX] = 1.0

            # Transition probabilities for unmasking (first term) and re-masking (second term)
            step_probs = d_t * pt_x1_probs * ((1 + self._aatypes_cfg.noise * t) / (1 - t))
            step_probs += d_t * (1 - aatypes_t_is_mask) * mask_one_hot.view(1, 1, -1) * self._aatypes_cfg.noise

        elif self._aatypes_cfg.interpolant_type == "uniform":
            assert S in {data_utils.NUM_TOKENS, data_utils.NUM_MODEL_TOKENS}
            device = logits_1.device
            N = self._aatypes_cfg.noise

            if S == data_utils.NUM_MODEL_TOKENS:
                logits_1[:, :, data_utils.UNK_TOKEN_INDEX] = -1e9
                logits_1[:, :, data_utils.MASK_TOKEN_INDEX] = -1e9
            pt_x1_probs = F.softmax(logits_1 / self._aatypes_cfg.temp, dim=-1)
            pt_x1_eq_xt_prob = torch.gather(pt_x1_probs, dim=-1, index=aatypes_t.long().unsqueeze(-1))

            # Uniform CTMC transition rate; see Austin et al. (2021)
            real_S = data_utils.NUM_TOKENS
            step_probs = d_t * (pt_x1_probs * ((1 + N + N * (real_S - 1) * t) / (1 - t)) + N * pt_x1_eq_xt_prob)

        else:
            raise ValueError(f"Unknown aatypes interpolant type {self._aatypes_cfg.interpolant_type}")

        step_probs = self._regularize_step_probs(step_probs, aatypes_t)
        return torch.multinomial(step_probs.view(-1, S), num_samples=1).view(batch_size, num_res)

    def _aatypes_euler_step_purity(self, d_t, t, logits_1, aatypes_t):
        """
        Purity-based unmasking step (masking interpolant only).

        Instead of sampling each position independently, this selects the
        top-k masked positions ranked by model confidence (max log-prob) and
        unmasks them in one shot. The number k is drawn from a Binomial
        distribution parameterised by the current step size d_t and noise level.

        After unmasking, a small fraction of positions are re-masked to maintain
        exploration.
        """
        batch_size, num_res, S = logits_1.shape
        assert aatypes_t.shape == (batch_size, num_res)
        assert S == data_utils.NUM_MODEL_TOKENS
        assert self._aatypes_cfg.interpolant_type == "masking"
        device = logits_1.device

        # Compute confidence over the four real bases only.
        pt_x1_probs = F.softmax(
            logits_1[:, :, : data_utils.NUM_TOKENS] / self._aatypes_cfg.temp,
            dim=-1,
        )
        max_logprob = torch.max(torch.log(pt_x1_probs), dim=-1)[0]  # (B, N)

        # Bias: only currently masked positions are eligible for unmasking
        max_logprob = max_logprob - (aatypes_t != data_utils.MASK_TOKEN_INDEX).float() * 1e9

        # Rank masked positions by confidence (descending)
        sorted_idcs = torch.argsort(max_logprob, dim=-1, descending=True)  # (B, N)

        # How many positions to unmask this step?
        unmask_prob = (d_t * ((1 + self._aatypes_cfg.noise * t) / (1 - t)).to(device)).clamp(max=1)
        num_masked = torch.count_nonzero(aatypes_t == data_utils.MASK_TOKEN_INDEX, dim=-1).float()
        number_to_unmask = torch.binomial(count=num_masked, prob=unmask_prob)

        # Sample which token to assign to each unmasked position
        unmasked_samples = torch.multinomial(pt_x1_probs.reshape(-1, data_utils.NUM_TOKENS), num_samples=1).view(
            batch_size, num_res
        )

        # Build a binary mask for the top-k positions to unmask (vectorised)
        D_grid = torch.arange(num_res, device=device).view(1, -1).repeat(batch_size, 1)
        select_mask = (D_grid < number_to_unmask.view(-1, 1)).float()

        # Handle the edge case where number_to_unmask == 0 by replicating first index
        fallback_idcs = sorted_idcs[:, 0].view(-1, 1).repeat(1, num_res)
        effective_idcs = (select_mask * sorted_idcs + (1 - select_mask) * fallback_idcs).long()

        # Scatter the selection mask back to residue positions
        unmask_positions = torch.zeros((batch_size, num_res), device=device)
        unmask_positions.scatter_(
            dim=1,
            index=effective_idcs,
            src=torch.ones((batch_size, num_res), device=device),
        )
        # Zero out the scatter contribution when number_to_unmask == 0
        unmask_positions *= 1 - (number_to_unmask == 0).view(-1, 1).float()

        # Apply unmasking
        aatypes_t = aatypes_t * (1 - unmask_positions) + unmasked_samples * unmask_positions

        # Re-mask a small fraction for exploration
        u = torch.rand(batch_size, num_res, device=self._device)
        re_mask = (u < d_t * self._aatypes_cfg.noise).float()
        aatypes_t = aatypes_t * (1 - re_mask) + data_utils.MASK_TOKEN_INDEX * re_mask

        return aatypes_t
