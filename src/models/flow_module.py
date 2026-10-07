"""Inference-only Lightning wrapper; model state keys match research checkpoints."""

import logging
import os

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule

from data.interpolant import Interpolant
from models.flow_model import FlowModel
from utils.checkpoint_bundle import config_sha256
from utils.inference_output import InferenceOutputBatch, InferenceOutputWriter, convert_trajectories
from utils.inference_task import build_inference_conditioning, writes_reference

logger = logging.getLogger(__name__)


class FlowModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self._interpolant_cfg = cfg.interpolant
        self._infer_cfg = cfg.inference
        self.model = FlowModel(cfg.model)
        self._inference_dir = None
        self._inference_checkpoint_reference = None

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._get_inference_dir()]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._get_inference_dir()
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir

    def _get_inference_dir(self):
        inference_dir = OmegaConf.select(self._cfg, "inference.samples_dir")
        if inference_dir is None:
            raise ValueError("Set inference.samples_dir before prediction.")
        return str(inference_dir)

    def on_predict_start(self):
        infer_cfg = getattr(self, "_infer_cfg", None)
        if infer_cfg is None:
            return
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        config_path = os.path.join(self.inference_dir, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(config=self._cfg, f=f.name)
        logger.info("Saving inference config to %s", config_path)

        if self._inference_checkpoint_reference is not None:
            manifest = {
                "checkpoint": self._inference_checkpoint_reference,
                "runtime": {
                    "task": OmegaConf.select(self._cfg, "inference.task"),
                    "seed": OmegaConf.select(self._cfg, "experiment.seed"),
                    "devices": OmegaConf.select(self._cfg, "experiment.trainer.devices"),
                    "weight_variant": getattr(self, "_inference_weight_variant", "raw"),
                },
                "effective_config_sha256": config_sha256(self._cfg),
            }
            manifest_path = os.path.join(self.inference_dir, "manifest.yaml")
            OmegaConf.save(config=OmegaConf.create(manifest), f=manifest_path)
            logger.info("Saving inference manifest to %s", manifest_path)

    def predict_step(self, batch, batch_idx):
        del batch_idx  # Unused
        if "sample_seed" in batch:
            sample_seeds = batch["sample_seed"].reshape(-1)
            if sample_seeds.numel() != 1:
                raise ValueError("Per-candidate inference seeds require batch_size=1")
            sample_seed = int(sample_seeds[0].item())
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
        device = self.device
        interpolant = Interpolant(self._interpolant_cfg)
        interpolant.set_device(device)
        output_writer = InferenceOutputWriter(self._infer_cfg, self.inference_dir)

        if "sample_id" in batch:
            sample_ids = batch["sample_id"].squeeze().tolist()
        else:
            sample_ids = [0]
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids
        num_batch = len(sample_ids)

        conditioning = build_inference_conditioning(self._infer_cfg.task, batch)
        sample_length = conditioning.sample_length

        sample_dirs = output_writer.sample_dirs(batch, sample_length, sample_ids)
        if writes_reference(self._infer_cfg.task):
            output_writer.write_forward_folding_reference(batch, sample_dirs[0])

        noisy_traj, clean_traj = interpolant.sample(
            num_batch,
            sample_length,
            self.model,
            conditioning.sampling,
            separate_t=self._interpolant_cfg.codesign_separate_t,
            return_trajectories=bool(self._infer_cfg.write_sample_trajectories),
        )

        output = InferenceOutputBatch(
            batch=batch,
            noisy_traj=noisy_traj,
            clean_traj=clean_traj,
            sample_ids=sample_ids,
            sample_dirs=sample_dirs,
            sample_length=sample_length,
            num_batch=num_batch,
            diffuse_mask=conditioning.sampling.diffuse_mask,
            true_aatypes=conditioning.true_aatypes,
        )
        trajectories = convert_trajectories(output)
        written_samples = output_writer.write(output, trajectories)
        return written_samples

