import pytest
import torch

from utils.inference_task import (
    build_inference_conditioning,
    is_scaffolding,
    sample_names,
    validate_inference_task,
    writes_reference,
)


def test_conditioning_preserves_task_specific_inputs():
    batch = {
        "trans_1": torch.zeros(1, 3, 3),
        "rotmats_1": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 3, 1, 1),
        "aatypes_1": torch.tensor([[0, 1, 2]]),
        "res_mask": torch.ones(1, 3),
        "diffuse_mask": torch.tensor([[0.0, 1.0, 1.0]]),
    }

    forward = build_inference_conditioning("forward_folding", batch)
    inverse = build_inference_conditioning("inverse_folding", batch)
    scaffolding = build_inference_conditioning("scaffolding", batch)
    unconditional = build_inference_conditioning("unconditional", {"num_res": torch.tensor(7)})

    assert forward.sampling.task == "forward_folding"
    assert inverse.sampling.task == "inverse_folding"
    assert inverse.true_aatypes is batch["aatypes_1"]
    assert scaffolding.sampling.fixed_atom28 is None
    assert unconditional.sample_length == 7


@pytest.mark.parametrize(
    ("task", "batch", "sample_ids", "expected"),
    [
        ("unconditional", {}, [3], ["sample_3"]),
        ("forward_folding", {"pdb_name": ["rna"]}, [3], ["rna"]),
        ("inverse_folding", {"chain_name": ["rna_A"]}, [3], ["rna_A"]),
        ("scaffolding", {"pdb_name": ["motif"]}, [3], ["motif_sample_3"]),
    ],
)
def test_sample_names(task, batch, sample_ids, expected):
    assert sample_names(task, batch, sample_ids) == expected


def test_task_flags_and_validation():
    assert writes_reference("forward_folding") is True
    assert is_scaffolding("scaffolding") is True
    with pytest.raises(ValueError, match="Unknown task"):
        validate_inference_task("circular_permutation")
