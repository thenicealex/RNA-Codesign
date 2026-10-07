import json
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from biotite.sequence.io import fasta

from np import residue_constants as rc
from utils import data_utils
from utils.inference_task import is_scaffolding, sample_names
from utils.pdb_io import save_traj, write_prot_to_pdb

logger = logging.getLogger(__name__)


@dataclass
class InferenceOutputBatch:
    batch: Any
    noisy_traj: Any
    clean_traj: Any
    sample_ids: list
    sample_dirs: list[str]
    sample_length: int
    num_batch: int
    diffuse_mask: Any
    true_aatypes: Any


@dataclass(frozen=True)
class ConvertedTrajectories:
    atom28_clean: np.ndarray
    atom28_noisy: np.ndarray
    aatypes_noisy: np.ndarray
    aatypes_clean: np.ndarray


@dataclass(frozen=True)
class WrittenSample:
    sample_dir: str
    sample_path: str | None
    codesign_seq: str


def convert_trajectories(output: InferenceOutputBatch) -> ConvertedTrajectories:
    """Convert model trajectories to validated batch-first NumPy arrays."""
    atom28_noisy = data_utils.to_numpy(torch.stack([x[0] for x in output.noisy_traj], dim=0).transpose(0, 1))
    atom28_clean = data_utils.to_numpy(torch.stack([x[0] for x in output.clean_traj], dim=0).transpose(0, 1))
    aatypes_noisy = data_utils.to_numpy(torch.stack([x[1] for x in output.noisy_traj], dim=0).transpose(0, 1).long())
    aatypes_clean = data_utils.to_numpy(torch.stack([x[1] for x in output.clean_traj], dim=0).transpose(0, 1).long())

    noisy_steps = atom28_noisy.shape[1]
    clean_steps = atom28_clean.shape[1]
    assert atom28_noisy.shape == (output.num_batch, noisy_steps, output.sample_length, 28, 3)
    assert atom28_clean.shape == (output.num_batch, clean_steps, output.sample_length, 28, 3)
    assert aatypes_noisy.shape == (output.num_batch, noisy_steps, output.sample_length)
    assert aatypes_clean.shape == (output.num_batch, clean_steps, output.sample_length)

    for batch_index in range(output.num_batch):
        for residue_index in range(output.sample_length):
            token = aatypes_noisy[batch_index, -1, residue_index]
            if token in {data_utils.UNK_TOKEN_INDEX, data_utils.MASK_TOKEN_INDEX}:
                raise ValueError(
                    f"Sample {output.sample_ids[batch_index]}, residue {residue_index} ended with "
                    f"non-clean sequence token {token}; clean samples must contain only A/G/C/U"
                )

    return ConvertedTrajectories(
        atom28_clean=atom28_clean,
        atom28_noisy=atom28_noisy,
        aatypes_noisy=aatypes_noisy,
        aatypes_clean=aatypes_clean,
    )


class InferenceOutputWriter:
    """Persist inference samples and already-computed result tables."""

    def __init__(self, infer_cfg, inference_dir):
        self._infer_cfg = infer_cfg
        self._inference_dir = inference_dir
        self._task = infer_cfg.task

    def sample_dirs(self, batch, sample_length, sample_ids) -> list[str]:
        length_dir = os.path.join(self._inference_dir, f"length_{sample_length}")
        return [os.path.join(length_dir, sample_name) for sample_name in sample_names(self._task, batch, sample_ids)]

    @staticmethod
    def write_forward_folding_reference(batch, sample_dir) -> None:
        atom28 = batch["atom28_gt_positions"]
        os.makedirs(sample_dir, exist_ok=True)
        write_prot_to_pdb(
            prot_pos=atom28[0].cpu().detach().numpy(),
            file_path=os.path.join(sample_dir, batch["pdb_name"][0] + "_gt.pdb"),
            aatype=batch["aatypes_1"][0].cpu().detach().numpy(),
        )

    def write(
        self,
        output: InferenceOutputBatch,
        trajectories: ConvertedTrajectories,
    ) -> list[WrittenSample]:
        self._write_scaffolding_metadata(output)
        diffuse_mask = output.diffuse_mask
        if diffuse_mask is None:
            diffuse_mask = torch.ones(1, output.sample_length)
        diffuse_mask = diffuse_mask.cpu()
        true_aatypes = output.true_aatypes
        if true_aatypes is not None:
            true_aatypes = true_aatypes.cpu()

        return [
            self._write_sample(
                trajectories.atom28_clean[index],
                trajectories.atom28_noisy[index],
                trajectories.aatypes_noisy[index],
                trajectories.aatypes_clean[index],
                true_aatypes,
                diffuse_mask,
                output.sample_length,
                output.sample_dirs[index],
            )
            for index in range(output.num_batch)
        ]

    def _write_scaffolding_metadata(self, output: InferenceOutputBatch) -> None:
        if not is_scaffolding(self._task):
            return

        for index, sample_dir in enumerate(output.sample_dirs):
            os.makedirs(sample_dir, exist_ok=True)
            if getattr(self._infer_cfg, "copy_source_pdb", True) and "source_pdb_path" in output.batch:
                shutil.copy2(
                    output.batch["source_pdb_path"][index],
                    os.path.join(sample_dir, "original_input.pdb"),
                )
            np.savetxt(
                os.path.join(sample_dir, "motif_mask.txt"),
                data_utils.to_numpy(output.batch["motif_mask"])[index],
                fmt="%d",
            )
            if "linker_lengths" in output.batch:
                linker_lengths = data_utils.to_numpy(output.batch["linker_lengths"])[index]
                linker_plan = {
                    "kind": data_utils.to_numpy(output.batch["linker_kinds"])[index],
                    "spatial_distance": data_utils.to_numpy(output.batch["linker_distances"])[index],
                    "sampled_length": linker_lengths,
                }
                if "linker_sampling_strategy" in output.batch:
                    linker_plan.update(
                        {
                            "minimum_length": np.full(len(linker_lengths), int(output.batch["linker_length_min"][index])),
                            "maximum_length": np.full(len(linker_lengths), int(output.batch["linker_length_max"][index])),
                            "sampling_strategy": output.batch["linker_sampling_strategy"][index],
                            "distance_span_temperature": float(output.batch["distance_span_temperature"][index]),
                            "distance_span_distribution": output.batch["distance_span_distribution_path"][index],
                        }
                    )
                pd.DataFrame(linker_plan).to_csv(os.path.join(sample_dir, "linker_plan.csv"), index=False)
            if "source_residue_index" in output.batch:
                source_indices = data_utils.to_numpy(output.batch["source_residue_index"])[index]
                source_numbers = data_utils.to_numpy(output.batch["source_residue_number"])[index]
                pd.DataFrame(
                    {
                        "output_residue_index": np.arange(1, len(source_indices) + 1),
                        "source_residue_index": source_indices,
                        "source_residue_number": source_numbers,
                        "is_generated_linker": source_indices == 0,
                    }
                ).to_csv(os.path.join(sample_dir, "residue_mapping.csv"), index=False)
                metadata = {
                    "candidate_index": int(data_utils.to_numpy(output.batch["candidate_index"])[index]),
                    "source_manifest_index": int(data_utils.to_numpy(output.batch["source_manifest_index"])[index]),
                    "sample_seed": int(data_utils.to_numpy(output.batch["sample_seed"])[index]),
                    "cut_after_source_index": int(data_utils.to_numpy(output.batch["cut_after_source_index"])[index]),
                    "cut_after_source_residue_number": int(
                        data_utils.to_numpy(output.batch["cut_after_source_residue_number"])[index]
                    ),
                    "new_5prime_output_index": 1,
                    "new_3prime_output_index": len(source_indices),
                }
                with open(os.path.join(sample_dir, "cutpoint.json"), "w") as handle:
                    json.dump(metadata, handle, indent=2)

    def _write_sample(
        self,
        clean_traj,
        noisy_traj,
        aa_traj,
        x0_aa_traj,
        true_aa,
        diffuse_mask,
        sample_length,
        sample_dir,
    ) -> WrittenSample:
        noisy_steps = noisy_traj.shape[0]
        clean_steps = clean_traj.shape[0]
        assert noisy_traj.shape == (noisy_steps, sample_length, 28, 3)
        assert clean_traj.shape == (clean_steps, sample_length, 28, 3)
        assert aa_traj.shape == (noisy_steps, sample_length)
        assert x0_aa_traj.shape == (clean_steps, sample_length)

        os.makedirs(sample_dir, exist_ok=True)
        traj_paths = save_traj(
            noisy_traj[-1],
            noisy_traj,
            np.flip(clean_traj, axis=0),
            data_utils.to_numpy(diffuse_mask)[0],
            output_dir=sample_dir,
            aa_traj=aa_traj,
            x0_aa_traj=x0_aa_traj,
            write_trajectories=self._infer_cfg.write_sample_trajectories,
        )
        codesign_seqs_dir = os.path.join(sample_dir, "codesign_seqs")
        os.makedirs(codesign_seqs_dir, exist_ok=True)
        codesign_seq = "".join(rc.restypes[index] for index in aa_traj[-1])
        self._write_fasta(
            os.path.join(codesign_seqs_dir, "codesign.fa"),
            "codesign_seq_1",
            codesign_seq,
        )
        if true_aa is not None:
            assert true_aa.shape == (1, sample_length)
            true_seq = "".join(rc.restypes_with_x[index] for index in true_aa[0])
            self._write_fasta(
                os.path.join(codesign_seqs_dir, "true_aa.fa"),
                "seq_1",
                true_seq,
            )
        return WrittenSample(
            sample_dir=sample_dir,
            sample_path=traj_paths.get("sample_path"),
            codesign_seq=codesign_seq,
        )

    @staticmethod
    def _write_fasta(path: str, name: str, sequence: str) -> None:
        fasta_file = fasta.FastaFile()
        fasta_file[name] = sequence
        fasta_file.write(path)

    @staticmethod
    def write_metrics(path: str, metrics: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame([metrics]).to_csv(path, index=False)
