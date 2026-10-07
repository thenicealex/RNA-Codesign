"""Convert RNA chains from one PDB/mmCIF file to inference features and metadata."""

import argparse
from pathlib import Path

import pandas as pd
import torch

from data import data_transforms, parsers
from np import residue_constants as rc
from utils.data_utils import write_pkl
from utils.rigid_utils import Rigid


def prepare_inputs(input_path: str, output_dir: str, chain_id: str | None = None) -> Path:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    chains = parsers.parse_structure(input_path, chain_id=chain_id)
    if not chains:
        raise ValueError("No RNA chains found for the requested input and chain")
    rows = []
    for chain in chains:
        positions = torch.as_tensor(chain.atom_positions, dtype=torch.float32)
        mask = torch.as_tensor(chain.atom_mask, dtype=torch.float32)
        c1_mask = mask[:, rc.atom_order["C1'"]]
        if c1_mask.sum() == 0:
            raise ValueError(f"Chain {chain.chain_id} has no C1' atoms")
        center = (positions[:, rc.atom_order["C1'"]] * c1_mask[:, None]).sum(0) / c1_mask.sum()
        features = {
            "aatype": torch.as_tensor(chain.aatype, dtype=torch.long),
            "all_atom_positions": (positions - center) * mask[..., None],
            "all_atom_mask": mask,
        }
        for transform in (
            data_transforms.atom28_to_frames,
            data_transforms.make_atom23_masks,
            data_transforms.atom28_to_torsion_angles,
        ):
            features = transform(features)
        frames = Rigid.from_tensor_4x4(features["rigidgroups_gt_frames"])[:, 0]
        processed = {
            "trans_1": frames.get_trans(),
            "rotmats_1": frames.get_rots().get_rot_mats(),
            "aatype": features["aatype"],
            "residue_mask": torch.ones(len(chain.aatype)),
            "trans_mask": c1_mask,
            "rot_mask": features["rigidgroups_gt_exists"][:, 0],
            "aatype_known_mask": (features["aatype"] < rc.restype_num).float(),
            "atom28_gt_positions": features["all_atom_positions"],
            "atom28_gt_mask": mask,
            "torsion_angles": features["torsion_angles_sin_cos"],
            "torsion_mask": features["torsion_angles_mask"],
        }
        for key in (
            "atom23_atom_exists", "atom28_atom_exists", "residx_atom23_to_atom28", "residx_atom28_to_atom23"
        ):
            processed[key] = features[key]
        name = f"{Path(input_path).stem}_{chain.chain_id}"
        feature_path = output / f"{name}.pkl"
        write_pkl(str(feature_path), {key: value.numpy() for key, value in processed.items()})
        rows.append({"pdb_name": name, "processed_path": str(feature_path), "length": len(chain.aatype)})
    metadata_path = output / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chain-id")
    args = parser.parse_args()
    print(prepare_inputs(args.input, args.output_dir, args.chain_id))


if __name__ == "__main__":
    main()
