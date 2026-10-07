import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

ARTIFACT_KEY = "rna_codesign_artifact"
EMA_STATE_KEY = "ema_state"
ARTIFACT_SCHEMA_VERSION = 2
_MISSING = object()


def _resolved_container(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    return OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)


def config_sha256(cfg: DictConfig | dict[str, Any]) -> str:
    payload = json.dumps(
        _resolved_container(cfg),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_contract_sha256(cfg: DictConfig | dict[str, Any]) -> str:
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    model_cfg = OmegaConf.select(cfg, "model")
    if model_cfg is None:
        raise ValueError("Checkpoint config is missing model")
    return config_sha256({"model": _resolved_container(model_cfg)})


def add_checkpoint_artifact(
    checkpoint: dict[str, Any],
    cfg: DictConfig,
    lineage: dict[str, Any] | None = None,
) -> None:
    training_config = _resolved_container(cfg)
    weight_variants = ["raw"]
    ema_state = checkpoint.get(EMA_STATE_KEY)
    if ema_state is not None and bool(ema_state.get("initialized", False)):
        weight_variants.append("ema")
    checkpoint[ARTIFACT_KEY] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "training_config": training_config,
        "config_sha256": config_sha256(training_config),
        "model_contract_sha256": model_contract_sha256(training_config),
        "lineage": lineage or {},
        "weight_variants": weight_variants,
        "default_inference_weight": "ema" if "ema" in weight_variants else "raw",
    }


@dataclass(frozen=True)
class CheckpointBundle:
    path: str
    checkpoint: dict[str, Any]
    training_config: DictConfig
    metadata: dict[str, Any]

    @property
    def state_dict(self) -> dict[str, Any]:
        return self.checkpoint["state_dict"]

    @property
    def weight_variants(self) -> tuple[str, ...]:
        return tuple(self.metadata["weight_variants"])

    def get_state_dict(self, weight_variant: str) -> OrderedDict:
        if weight_variant == "raw":
            return self.checkpoint["state_dict"]
        if weight_variant != "ema":
            raise ValueError(f"Unknown checkpoint weight variant: {weight_variant}")
        if "ema" not in self.weight_variants:
            raise ValueError("Checkpoint has no initialized EMA weights; use inference.weight_variant=raw")

        ema_state = self.checkpoint.get(EMA_STATE_KEY)
        if ema_state is None or not ema_state.get("initialized", False):
            raise ValueError("Checkpoint EMA metadata and EMA state are inconsistent")

        raw_state = self.checkpoint["state_dict"]
        state = OrderedDict(raw_state)
        if hasattr(raw_state, "_metadata"):
            state._metadata = raw_state._metadata
        for name, value in ema_state["params"].items():
            model_name = f"model.{name}"
            if model_name not in state:
                raise ValueError(f"EMA parameter is absent from model state: {model_name}")
            state[model_name] = value
        return state

    @property
    def config_sha256(self) -> str:
        return self.metadata["config_sha256"]

    @property
    def model_contract_sha256(self) -> str:
        return self.metadata["model_contract_sha256"]

    @property
    def checkpoint_reference(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "config_sha256": self.config_sha256,
            "model_contract_sha256": self.model_contract_sha256,
            "epoch": self.checkpoint.get("epoch"),
            "global_step": self.checkpoint.get("global_step"),
            "weight_variants": list(self.weight_variants),
        }

    def checkpoint_reference_for(self, weight_variant: str) -> dict[str, Any]:
        if weight_variant not in self.weight_variants:
            self.get_state_dict(weight_variant)
        reference = self.checkpoint_reference
        reference["weight_variant"] = weight_variant
        return reference


def load_checkpoint_bundle(path: str) -> CheckpointBundle:
    checkpoint_path = os.path.abspath(os.path.expanduser(path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no state_dict: {checkpoint_path}")

    metadata = checkpoint.get(ARTIFACT_KEY)
    if metadata is None:
        raise ValueError("Checkpoint has no embedded rna-codesign artifact metadata")

    schema_version = metadata.get("schema_version")
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema version: {schema_version}")
    weight_variants = metadata.get("weight_variants")
    if not isinstance(weight_variants, list) or "raw" not in weight_variants:
        raise ValueError("Checkpoint has invalid weight variant metadata")
    if "ema" in weight_variants:
        ema_state = checkpoint.get(EMA_STATE_KEY)
        if ema_state is None or not bool(ema_state.get("initialized", False)):
            raise ValueError("Checkpoint advertises EMA weights without initialized EMA state")
    training_config = OmegaConf.create(metadata["training_config"])
    actual_hash = config_sha256(training_config)
    if actual_hash != metadata.get("config_sha256"):
        raise ValueError("Checkpoint training config hash mismatch")
    actual_contract_hash = model_contract_sha256(training_config)
    if actual_contract_hash != metadata.get("model_contract_sha256"):
        raise ValueError("Checkpoint model contract hash mismatch")

    return CheckpointBundle(
        path=checkpoint_path,
        checkpoint=checkpoint,
        training_config=training_config,
        metadata=metadata,
    )


def _copy_value(target: DictConfig, source: DictConfig, path: str) -> None:
    value = OmegaConf.select(source, path, default=_MISSING)
    if value is _MISSING:
        return
    OmegaConf.update(
        target,
        path,
        OmegaConf.create(OmegaConf.to_container(value, resolve=True)) if OmegaConf.is_config(value) else value,
        merge=False,
        force_add=True,
    )


def build_inference_config(
    training_cfg: DictConfig,
    runtime_cfg: DictConfig,
) -> DictConfig:
    effective_cfg = OmegaConf.create(_resolved_container(training_cfg))
    for path in (
        "task_name",
        "tags",
        "data",
        "folding",
        "interpolant",
        "evaluation",
        "inference",
        "paths",
        "experiment.seed",
        "experiment.trainer",
    ):
        _copy_value(effective_cfg, runtime_cfg, path)
    return effective_cfg
