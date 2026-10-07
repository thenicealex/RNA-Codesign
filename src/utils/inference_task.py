"""Fixed inference-task helpers."""

from dataclasses import dataclass
from typing import Any

from data.interpolant import SamplingCondition
from utils import data_utils

SUPPORTED_INFERENCE_TASKS = (
    "unconditional",
    "forward_folding",
    "inverse_folding",
    "scaffolding",
)


@dataclass(frozen=True)
class InferenceConditioning:
    sample_length: int
    sampling: SamplingCondition
    true_aatypes: Any = None


def validate_inference_task(task: str) -> None:
    if task not in SUPPORTED_INFERENCE_TASKS:
        raise ValueError(f"Unknown task {task}")


def build_inference_dataset(task: str, infer_cfg, samples_cfg, dataset_cfg):
    validate_inference_task(task)
    if task == "unconditional":
        from data.datasets import LengthDataset

        return LengthDataset(samples_cfg)
    if task in {"forward_folding", "inverse_folding"}:
        from data.datasets import PdbDataset

        return PdbDataset(dataset_cfg)

    from data.scaffold_datasets import ScaffoldDataset

    return ScaffoldDataset(infer_cfg)


def build_inference_conditioning(task: str, batch) -> InferenceConditioning:
    validate_inference_task(task)
    if task == "unconditional":
        return InferenceConditioning(
            sample_length=batch["num_res"].item(),
            sampling=SamplingCondition(
                res_mask=batch.get("res_mask"),
                rot_mask=batch.get("rot_mask"),
                aatype_diffuse_mask=batch.get("aatype_diffuse_mask"),
            ),
        )
    if task == "forward_folding":
        aatype_diffuse_mask = batch.get("aatype_diffuse_mask")
        if aatype_diffuse_mask is None:
            aatype_known_mask = batch.get("aatype_known_mask", batch["aatypes_1"] < data_utils.NUM_TOKENS)
            aatype_diffuse_mask = (~aatype_known_mask.bool()).to(dtype=batch["res_mask"].dtype)
        return InferenceConditioning(
            sample_length=batch["trans_1"].shape[1],
            sampling=SamplingCondition(
                task="forward_folding",
                aatypes_1=batch["aatypes_1"],
                res_mask=batch.get("res_mask"),
                rot_mask=batch.get("rot_mask"),
                aatype_diffuse_mask=aatype_diffuse_mask,
            ),
        )
    if task == "inverse_folding":
        return InferenceConditioning(
            sample_length=batch["trans_1"].shape[1],
            sampling=SamplingCondition(
                task="inverse_folding",
                trans_1=batch["trans_1"],
                rotmats_1=batch["rotmats_1"],
                res_mask=batch.get("res_mask"),
                rot_mask=batch.get("rot_mask"),
                aatype_diffuse_mask=batch.get("aatype_diffuse_mask"),
            ),
            true_aatypes=batch["aatypes_1"],
        )
    return InferenceConditioning(
        sample_length=batch["trans_1"].shape[1],
        sampling=SamplingCondition(
            trans_1=batch["trans_1"],
            rotmats_1=batch["rotmats_1"],
            aatypes_1=batch["aatypes_1"],
            res_mask=batch.get("res_mask"),
            rot_mask=batch.get("rot_mask"),
            diffuse_mask=batch["diffuse_mask"],
            aatype_diffuse_mask=batch.get("aatype_diffuse_mask"),
            fixed_atom28=batch.get("atom28_1"),
            fixed_atom28_mask=batch.get("atom28_fixed_mask"),
        ),
    )


def sample_names(task: str, batch, sample_ids: list[int]) -> list[str]:
    validate_inference_task(task)
    if task == "unconditional":
        return [f"sample_{sample_id}" for sample_id in sample_ids]
    if task == "forward_folding":
        return [batch["pdb_name"][0]]
    if task == "inverse_folding":
        return [batch["chain_name"][0]]

    pdb_names = batch["pdb_name"]
    if hasattr(pdb_names, "tolist"):
        pdb_names = pdb_names.tolist()
    elif not isinstance(pdb_names, list):
        pdb_names = [pdb_names] * len(sample_ids)
    return [f"{pdb_names[index]}_sample_{sample_ids[index]}" for index in range(len(sample_ids))]


def writes_reference(task: str) -> bool:
    validate_inference_task(task)
    return task == "forward_folding"


def is_scaffolding(task: str) -> bool:
    validate_inference_task(task)
    return task == "scaffolding"
