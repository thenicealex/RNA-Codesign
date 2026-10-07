import os
import re

import numpy as np

from np import nucleicacid


def create_full_prot(
    atom37: np.ndarray,
    atom37_mask: np.ndarray,
    aatype=None,
    b_factors=None,
):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 28
    n = atom37.shape[0]
    residue_index = np.arange(n)
    chain_index = np.zeros(n)
    if b_factors is None:
        b_factors = np.zeros([n, 28])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return nucleicacid.NucleicAcid(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors,
    )


def write_prot_to_pdb(
    prot_pos: np.ndarray,
    file_path: str,
    aatype: np.ndarray = None,
    overwrite=False,
    no_indexing=False,
    b_factors=None,
):
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip(".pdb")
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max(
            [
                int(re.findall(r"_(\d+).pdb", x)[0])
                for x in existing_files
                if re.findall(r"_(\d+).pdb", x)
                if re.findall(r"_(\d+).pdb", x)
            ]
            + [0]
        )
    if not no_indexing:
        save_path = file_path.replace(".pdb", "") + f"_{max_existing_idx + 1}.pdb"
    else:
        save_path = file_path

    if aatype is not None:
        assert aatype.ndim == prot_pos.ndim - 2

    with open(save_path, "w") as f:
        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                prot = create_full_prot(pos37, atom37_mask, aatype=aatype[t], b_factors=b_factors)
                pdb_prot = nucleicacid.to_pdb(prot, model=t + 1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
            prot = create_full_prot(prot_pos, atom37_mask, aatype=aatype, b_factors=b_factors)
            pdb_prot = nucleicacid.to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f"Invalid positions shape {prot_pos.shape}")
        f.write("END")
    return save_path


def save_traj(
    sample: np.ndarray,
    bb_noisy_traj: np.ndarray,
    x0_traj: np.ndarray,
    diffuse_mask: np.ndarray,
    output_dir: str,
    aa_traj=None,
    x0_aa_traj=None,
    write_trajectories=True,
):
    """Writes final sample and reverse diffusion trajectory.

    Args:
        bb_noisy_traj: [noisy_T, N, 28, 3] noisy backbone states at each ODE step.
            T is number of time steps. First time step is t=eps,
            i.e. bb_noisy_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [x0_T, N, 28, 3] model x0-predictions at each ODE step.
        diffuse_mask: [N] which residues are diffused.
        output_dir: where to save samples.
        aa_traj: [noisy_T, N] nucleotide types from noisy trajectory.
        x0_aa_traj: [x0_T, N] nucleotide types from x0 predictions.
        write_trajectories: bool Whether to also write the trajectories as well
                                 as the final sample

    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues
        residues if there are any.
    """

    # Write sample.
    diffuse_mask = diffuse_mask.astype(bool)
    sample_path = os.path.join(output_dir, "sample.pdb")
    noisy_traj_path = os.path.join(output_dir, "bb_traj.pdb")
    x0_traj_path = os.path.join(output_dir, "x0_traj.pdb")

    # Use b-factors to specify which residues are diffused.
    b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 28))
    sample_b_factors = b_factors

    noisy_traj_length, num_res, _, _ = bb_noisy_traj.shape
    x0_traj_length = x0_traj.shape[0]
    assert sample.shape == (num_res, 28, 3)
    assert bb_noisy_traj.shape == (noisy_traj_length, num_res, 28, 3)
    assert x0_traj.shape == (x0_traj_length, num_res, 28, 3)

    if aa_traj is not None:
        assert aa_traj.shape == (noisy_traj_length, num_res)
        assert x0_aa_traj is not None
        assert x0_aa_traj.shape == (x0_traj_length, num_res)

    sample_path = write_prot_to_pdb(
        sample,
        sample_path,
        b_factors=sample_b_factors,
        no_indexing=True,
        aatype=aa_traj[-1] if aa_traj is not None else None,
    )
    if write_trajectories:
        noisy_traj_path = write_prot_to_pdb(
            bb_noisy_traj,
            noisy_traj_path,
            b_factors=b_factors,
            no_indexing=True,
            aatype=aa_traj,
        )
        x0_traj_path = write_prot_to_pdb(
            x0_traj,
            x0_traj_path,
            b_factors=b_factors,
            no_indexing=True,
            aatype=x0_aa_traj,
        )
    return {
        "sample_path": sample_path,
        "traj_path": noisy_traj_path,
        "x0_traj_path": x0_traj_path,
    }
