"""Export schema-v2 research checkpoints as inference-only weight bundles."""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from utils.checkpoint_bundle import EMA_STATE_KEY, add_checkpoint_artifact, load_checkpoint_bundle


def export_checkpoint(input_path: str, output_path: str) -> None:
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    bundle = load_checkpoint_bundle(input_path)
    state = bundle.get_state_dict("raw")
    if any(not name.startswith("model.") for name in state):
        raise ValueError("Checkpoint state contains non-model tensors; review before exporting")
    exported = {"state_dict": state}
    if "ema" in bundle.weight_variants:
        exported[EMA_STATE_KEY] = {
            "initialized": True,
            "params": bundle.checkpoint[EMA_STATE_KEY]["params"],
        }
    model_cfg = OmegaConf.create(
        {"model": OmegaConf.to_container(bundle.training_config.model, resolve=True)}
    )
    add_checkpoint_artifact(exported, model_cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
