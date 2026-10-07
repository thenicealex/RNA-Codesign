import collections
import dataclasses
import io
import json
import logging
import os
import pickle
import string
from typing import Any

import numpy as np
import pandas as pd
import torch
from Bio.PDB.Chain import Chain
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from np import nucleicacid, residue_constants
from utils import rigid_utils

logger = logging.getLogger(__name__)

Rigid = rigid_utils.Rigid
NucleicAcid = nucleicacid.NucleicAcid

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + " "
CHAIN_TO_INT = {chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)}
INT_TO_CHAIN = {i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)}

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

CHAIN_FEATS = ["atom_positions", "aatype", "atom_mask", "residue_index", "b_factors"]

NUM_TOKENS = residue_constants.restype_num
UNK_TOKEN_INDEX = residue_constants.unk_restype_index
MASK_TOKEN_INDEX = residue_constants.mask_token_index
NUM_MODEL_TOKENS = residue_constants.model_token_num
C1P_IDX = residue_constants.atom_order["C1'"]


def to_numpy(value):
    return value.detach().cpu().numpy()


def aatype_to_seq(aatype):
    return "".join(residue_constants.restypes_with_x[index] for index in aatype)


def seq_to_aatype(sequence):
    return [residue_constants.restypes_with_x.index(residue) for residue in sequence]


class CPU_Unpickler(pickle.Unpickler):
    """Pytorch pickle loading workaround.

    https://github.com/pytorch/pytorch/issues/16797
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def remove_com_from_tensor_7(tensor_7):
    """Removes the center of mass along the L dimension from a tensor of shape (..., L, 7)."""
    rigid = Rigid.from_tensor_7(tensor_7)
    rigid_trans = rigid.get_trans()
    COM = rigid_trans.mean(dim=[-2], keepdim=True)
    rigid_trans = rigid_trans - COM
    rigid._trans = rigid_trans
    return rigid.to_tensor_7()


def create_rigid(rots, trans):
    rots = rigid_utils.Rotation(rot_mats=rots)
    return Rigid(rots=rots, trans=trans)


def write_pkl(save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, "wb") as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        else:
            with open(read_path, "rb") as handle:
                return pickle.load(handle)
    except Exception as e:
        try:
            with open(read_path, "rb") as handle:
                return CPU_Unpickler(handle).load()
        except Exception as e2:
            if verbose:
                logger.warning(f"Failed to read {read_path}. First error: {e}\n Second error: {e2}")
            raise e from e2


def load_jsonl(path) -> dict | list:
    """Load a JSON or JSONL file.

    Returns a dict if the file contains a single JSON object, or a list of
    dicts if the file is JSONL (one JSON object per line).
    """
    with open(path) as f:
        content = f.read().strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def chain_str_to_int(chain_str: str):
    chain_int = 0
    if len(chain_str) == 1:
        return CHAIN_TO_INT[chain_str]
    for i, chain_char in enumerate(chain_str):
        chain_int += CHAIN_TO_INT[chain_char] + (i * len(ALPHANUMERIC))
    return chain_int


def parse_chain_feats2(chain_feats, scale_factor=1.0, center=True):
    chain_feats["bb_mask"] = chain_feats["atom_mask"][:, C1P_IDX]
    bb_pos = chain_feats["atom_positions"][:, C1P_IDX]
    if center:
        bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats["bb_mask"]) + 1e-5)
        centered_pos = chain_feats["atom_positions"] - bb_center[None, None, :]
        scaled_pos = centered_pos / scale_factor
    else:
        scaled_pos = chain_feats["atom_positions"] / scale_factor
    chain_feats["all_atom_positions"] = scaled_pos * chain_feats["atom_mask"][..., None]
    chain_feats["all_atom_mask"] = chain_feats["atom_mask"]
    chain_feats["bb_positions"] = chain_feats["all_atom_positions"][:, C1P_IDX]
    return chain_feats


def parse_chain_feats(chain_feats, scale_factor=1.0, center=True):
    chain_feats["bb_mask"] = chain_feats["all_atom_mask"][:, C1P_IDX]
    bb_pos = chain_feats["all_atom_positions"][:, C1P_IDX]
    if center:
        bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats["bb_mask"]) + 1e-5)
        centered_pos = chain_feats["all_atom_positions"] - bb_center[None, None, :]
        scaled_pos = centered_pos / scale_factor
    else:
        scaled_pos = chain_feats["all_atom_positions"] / scale_factor
    chain_feats["all_atom_positions"] = scaled_pos * chain_feats["all_atom_mask"][..., None]
    chain_feats["bb_positions"] = chain_feats["all_atom_positions"][:, C1P_IDX]
    return chain_feats


def concat_np_features(np_dicts: list[dict[str, np.ndarray]], add_batch_dim: bool):
    """Performs a nested concatenation of feature dicts.

    Args:
        np_dicts: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.
        add_batch_dim: whether to add a batch dimension to each feature.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in np_dicts:
        for feat_name, feat_val in chain_dict.items():
            if add_batch_dim:
                feat_val = feat_val[None]
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict


def parse_pdb_feats(
    pdb_path: str,
    scale_factor=1.0,
):
    """
    Args:
        pdb_path: path to PDB file to read.
        scale_factor: factor to scale atom positions.
        mean_center: whether to mean center atom positions.
    Returns:
        Dict with CHAIN_FEATS features extracted from PDB with specified
        preprocessing.
    """
    with open(pdb_path) as f:
        content_string = f.read()

    prot = nucleicacid.from_pdb_string(content_string)
    feat_dict = dataclasses.asdict(prot)
    feat_dict["all_atom_positions"] = feat_dict.pop("atom_positions")
    feat_dict["all_atom_mask"] = feat_dict.pop("atom_mask")
    return parse_chain_feats(feat_dict, scale_factor=scale_factor)


def process_chain(chain: Chain, chain_id: str) -> NucleicAcid:
    """Convert a PDB chain object into a AlphaFold Protein instance.

    Forked from alphafold.common.protein.from_pdb_string

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.

    Took out lines 110-112 since that would mess up CDR numbering.

    Args:
        chain: Instance of Biopython's chain class.

    Returns:
        NucleicAcid object with features.
    """
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    chain_ids = []
    for res in chain:
        res_shortname = residue_constants.restype_3to1.get(res.resname, "N")
        restype_idx = residue_constants.restype_order.get(res_shortname, residue_constants.restype_num)
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.0
            res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    return NucleicAcid(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=np.array(chain_ids),
        b_factors=np.array(b_factors),
    )


def read_clusters(cluster_path, synthetic=False):
    pdb_to_cluster = {}
    with open(cluster_path) as f:
        for i, line in enumerate(f):
            chains = line.strip().split()
            for chain in chains:
                if not synthetic:
                    pdb = chain.strip()
                else:
                    pdb = chain.strip()
                pdb_to_cluster[pdb] = i
    return pdb_to_cluster


def generate_component_file(cluster_file, output_file, verbose=False):
    if not os.path.exists(cluster_file) and verbose:
        logger.warning(f"Cluster file {cluster_file} does not exist.")
        return None

    with open(cluster_file) as f:
        content = json.load(f)

    cluster_dict = {}
    for cluster_id, members in tqdm(content.items(), desc="Processing clusters"):
        keys = []
        for member_dict in members.values():
            keys.extend(member_dict.keys())

        cluster_dict[cluster_id] = keys

    # sort clusters by size
    cluster_dict = dict(sorted(cluster_dict.items(), key=lambda item: len(item[1])))

    with open(output_file, "w") as out_f:
        for _, members in cluster_dict.items():
            out_f.write(f"{', '.join(members)}\n")

    if verbose:
        logger.info(f"The number of clusters: {len(cluster_dict)}")
        logger.info(f"Parsed cluster file saved to {output_file}")
    return cluster_dict


def rog_filter(df, quantile):
    """Filter structures by radius of gyration using polynomial regression."""
    # Compute quantile thresholds per length
    y_quant = pd.pivot_table(
        df,
        values="radius_gyration",
        index="length",
        aggfunc=lambda x: np.quantile(x, quantile),
    )

    # Fit polynomial regression
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(y_quant.index.to_numpy()[:, None])
    model = LinearRegression().fit(poly_features, y_quant.radius_gyration.to_numpy())

    # Predict cutoffs for all sequence lengths
    max_len = df.length.max()
    length_grid = np.arange(1, max_len + 1)
    pred_features = poly.transform(length_grid[:, None])
    cutoffs = model.predict(pred_features) + 0.1

    # Apply filter
    row_cutoffs = df.length.map(lambda x: cutoffs[x - 1])
    return df[df.radius_gyration < row_cutoffs]


def vector_projection(R_ab, P_n):
    """
    Project the vector R_ab onto a plane with normal vector P_n.

    Parameters
    ----------
        R_ab: Tensor, shape = (N,3)
            Vector from atom a to b.
        P_n: Tensor, shape = (N,3)
            Normal vector of a plane onto which to project R_ab.

    Returns
    -------
        R_ab_proj: Tensor, shape = (N,3)
            Projected vector (orthogonal to P_n).
    """
    a_x_b = torch.sum(R_ab * P_n, dim=-1)
    b_x_b = torch.sum(P_n * P_n, dim=-1)
    return R_ab - (a_x_b / b_x_b)[:, None] * P_n


def calculate_neighbor_angles(R_ac, R_ab):
    """Calculate angles between atoms c <- a -> b.

    Parameters
    ----------
        R_ac: Tensor, shape = (N,3)
            Vector from atom a to c.
        R_ab: Tensor, shape = (N,3)
            Vector from atom a to b.

    Returns
    -------
        angle_cab: Tensor, shape = (N,)
            Angle between atoms c <- a -> b.
    """
    # cos(alpha) = (u * v) / (|u|*|v|)
    x = torch.sum(R_ac * R_ab, dim=1)  # shape = (N,)
    # sin(alpha) = |u x v| / (|u|*|v|)
    y = torch.cross(R_ac, R_ab).norm(dim=-1)  # shape = (N,)
    # avoid that for y == (0,0,0) the gradient wrt. y becomes NaN
    y = torch.max(y, torch.tensor(1e-9))
    angle = torch.atan2(y, x)
    return angle
