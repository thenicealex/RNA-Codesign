"""Run unconditional generation, forward/inverse folding, or motif scaffolding."""

import logging
import re

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from models.flow_module import FlowModule
from utils.checkpoint_bundle import build_inference_config, load_checkpoint_bundle
from utils.inference_task import build_inference_dataset, validate_inference_task

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def run(cfg: DictConfig) -> None:
    validate_inference_task(cfg.inference.task)
    if not cfg.inference.ckpt_path:
        raise ValueError("Set inference.ckpt_path to a schema-v2 model checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("The public inference CLI requires a CUDA GPU")
    pl.seed_everything(cfg.experiment.seed, workers=True)
    bundle = load_checkpoint_bundle(cfg.inference.ckpt_path)
    if cfg.inference.ckpt_id is None:
        from pathlib import Path

        cfg.inference.ckpt_id = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(bundle.path).stem)
    effective_cfg = build_inference_config(bundle.training_config, cfg)
    # Retain only settings consumed by inference in output provenance.
    effective_cfg = OmegaConf.create(
        {
            key: OmegaConf.to_container(effective_cfg[key], resolve=True)
            if OmegaConf.is_config(effective_cfg[key])
            else effective_cfg[key]
            for key in ("model", "data", "interpolant", "inference", "paths", "experiment", "task_name")
        }
    )
    effective_cfg.experiment = OmegaConf.create(OmegaConf.to_container(cfg.experiment, resolve=True))
    module = FlowModule(effective_cfg)
    module.load_state_dict(bundle.get_state_dict(cfg.inference.weight_variant), strict=True)
    module._inference_weight_variant = cfg.inference.weight_variant
    module._inference_checkpoint_reference = bundle.checkpoint_reference_for(cfg.inference.weight_variant)
    module.eval()
    dataset = build_inference_dataset(
        cfg.inference.task, cfg.inference, cfg.inference.samples, cfg.data.dataset
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=cfg.data.loader.num_workers
    )
    trainer = instantiate(cfg.experiment.trainer)
    trainer.predict(module, dataloaders=dataloader, return_predictions=False)
    logger.info("Samples saved to %s", effective_cfg.inference.samples_dir)


if __name__ == "__main__":
    run()
