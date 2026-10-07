"""Input datasets for the four public inference tasks."""

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import data_utils


def _process_csv_row(file_path):
    """Load a pre-processed rna3db feature pickle and return a training-ready dict.

    The pickle must contain:
        trans_1       [N, 3]       backbone C1' translations (centred, Angstrom)
        rotmats_1     [N, 3, 3]    backbone rotation matrices
        aatype        [N]          nucleotide types 0-3, 4=UNK
        residue_mask  [N]          real residue vs padding
        trans_mask    [N]          C1' translation validity
        rot_mask      [N]          C2'-C1'-O4' frame validity
        torsion_angles [N, 9, 2]  (sin, cos) torsion angle pairs  (optional)
        torsion_mask  [N, 9]       torsion validity mask           (optional)
    """
    feats = data_utils.read_pkl(file_path)
    required_schema_fields = {
        "residue_mask",
        "trans_mask",
        "rot_mask",
        "aatype_known_mask",
        "atom28_gt_positions",
        "atom28_gt_mask",
    }
    missing_fields = sorted(required_schema_fields.difference(feats))
    if missing_fields:
        raise ValueError(
            f"{file_path} uses an unsupported processed-data schema; missing {missing_fields}. "
            "Regenerate the processed pickle with the current preprocessing pipeline."
        )

    trans_1 = torch.tensor(feats["trans_1"], dtype=torch.float32)
    rotmats_1 = torch.tensor(feats["rotmats_1"], dtype=torch.float32)
    aatypes_1 = torch.tensor(feats["aatype"], dtype=torch.long)
    N = len(aatypes_1)
    residue_mask = torch.tensor(feats["residue_mask"], dtype=torch.float32)
    trans_mask = torch.tensor(feats["trans_mask"], dtype=torch.float32)
    rot_mask = torch.tensor(feats["rot_mask"], dtype=torch.float32)
    aatype_known_mask = torch.tensor(feats["aatype_known_mask"], dtype=torch.float32)
    atom23_atom_exists = torch.tensor(feats["atom23_atom_exists"], dtype=torch.float32)
    atom28_atom_exists = torch.tensor(feats["atom28_atom_exists"], dtype=torch.float32)
    residx_atom23_to_atom28 = torch.tensor(feats["residx_atom23_to_atom28"], dtype=torch.long)
    residx_atom28_to_atom23 = torch.tensor(feats["residx_atom28_to_atom23"], dtype=torch.long)

    # Single-chain sequential indexing (rna3db chains are pre-merged to one chain)
    res_idx = torch.arange(1, N + 1, dtype=torch.long)
    chain_idx = torch.zeros(N, dtype=torch.float32)

    gt_torsions = torch.tensor(feats["torsion_angles"], dtype=torch.float32) if "torsion_angles" in feats else None
    torsion_angles_mask = torch.tensor(feats["torsion_mask"], dtype=torch.float32) if "torsion_mask" in feats else None
    processed = {
        "aatypes_1": aatypes_1,
        "rotmats_1": rotmats_1,
        "trans_1": trans_1,
        "res_mask": residue_mask,
        "residue_mask": residue_mask,
        "trans_mask": trans_mask,
        "rot_mask": rot_mask,
        "aatype_known_mask": aatype_known_mask,
        "chain_idx": chain_idx,
        "res_idx": res_idx,
        "gt_torsions": gt_torsions,
        "torsion_angles_mask": torsion_angles_mask,
        "atom23_atom_exists": atom23_atom_exists,
        "atom28_atom_exists": atom28_atom_exists,
        "residx_atom23_to_atom28": residx_atom23_to_atom28,
        "residx_atom28_to_atom23": residx_atom28_to_atom23,
        "atom28_gt_positions": torch.tensor(feats["atom28_gt_positions"], dtype=torch.float32),
        "atom28_gt_mask": torch.tensor(feats["atom28_gt_mask"], dtype=torch.float32),
    }
    return processed


class LengthDataset(Dataset):
    """Synthetic length requests used by unconditional inference."""

    def __init__(self, samples_cfg):
        all_sample_lengths = range(
            samples_cfg.min_length,
            samples_cfg.max_length + 1,
            samples_cfg.length_step,
        )
        if samples_cfg.length_subset is not None:
            all_sample_lengths = [int(length) for length in samples_cfg.length_subset]

        num_batch = samples_cfg.num_batch
        if samples_cfg.samples_per_length % num_batch != 0:
            raise ValueError("samples_per_length must be divisible by num_batch")
        samples_per_batch = samples_cfg.samples_per_length // num_batch

        self._all_sample_ids = []
        for length in all_sample_lengths:
            for sample_id in range(samples_per_batch):
                sample_ids = torch.tensor([num_batch * sample_id + index for index in range(num_batch)])
                self._all_sample_ids.append((length, sample_ids, torch.arange(length)))

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        num_res, sample_id, res_idx = self._all_sample_ids[idx]
        return {
            "num_res": num_res,
            "sample_id": sample_id,
            "res_idx": res_idx,
        }


class PdbDataset(Dataset):
    def __init__(self, dataset_cfg):
        self.csv = pd.read_csv(dataset_cfg.metadata_path)
        required = {"pdb_name", "processed_path", "length"}
        if not required.issubset(self.csv.columns):
            raise ValueError(f"Metadata must contain {sorted(required)}")
        if self.csv.empty:
            raise ValueError("Input metadata contains no structures")

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        features = _process_csv_row(row.processed_path)
        # Optional torsions are unused during inference and cannot be collated as None.
        features = {key: value for key, value in features.items() if value is not None}
        features["pdb_name"] = str(row.pdb_name)
        features["chain_name"] = str(row.pdb_name)
        features["diffuse_mask"] = features["res_mask"].clone()
        features["csv_idx"] = torch.tensor([index])
        return features

