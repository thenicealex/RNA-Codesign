"""Library for parsing FASTA, PDB, and mmCIF data structures."""

import dataclasses
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
from Bio.PDB.Chain import Chain
from Bio.PDB.MMCIFParser import MMCIFParser

from data import mmcif_parsing
from np import nucleicacid, residue_constants

NucleicAcid = nucleicacid.NucleicAcid
DeletionMatrix = Sequence[Sequence[int]]


@dataclasses.dataclass
class ParsedChain:
    """Single RNA chain with coordinates and metadata."""

    pdb_id: str
    chain_id: str
    sequence: str
    aatype: np.ndarray
    atom_positions: np.ndarray
    atom_mask: np.ndarray
    residue_index: np.ndarray
    bb_mask: np.ndarray
    fixed_mask: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class ChainBreak:
    """A confirmed structural break between two adjacent residues."""

    left_index: int
    right_index: int
    left_residue_number: int
    right_residue_number: int
    chain_distance: float
    reason: str


def detect_chain_breaks(
    chain: ParsedChain,
    *,
    max_chain_distance: float = 2.2,
) -> list[ChainBreak]:
    """Find O3'-P chain breaks associated with missing residues."""
    if max_chain_distance <= 0:
        raise ValueError("max_chain_distance must be positive")

    o3_index = residue_constants.atom_order["O3'"]
    p_index = residue_constants.atom_order["P"]
    breaks = []
    for left_index in range(max(0, len(chain.sequence) - 1)):
        right_index = left_index + 1
        if chain.atom_mask[left_index, o3_index] <= 0 or chain.atom_mask[right_index, p_index] <= 0:
            continue
        o3_position = chain.atom_positions[left_index, o3_index]
        p_position = chain.atom_positions[right_index, p_index]
        if not np.all(np.isfinite(o3_position)) or not np.all(np.isfinite(p_position)):
            continue

        chain_distance = float(np.linalg.norm(o3_position - p_position))
        if chain_distance <= max_chain_distance:
            continue
        numbering_gap = int(chain.residue_index[right_index]) - int(chain.residue_index[left_index]) != 1
        shared_fields = {
            "left_index": left_index,
            "right_index": right_index,
            "left_residue_number": int(chain.residue_index[left_index]),
            "right_residue_number": int(chain.residue_index[right_index]),
            "chain_distance": chain_distance,
        }
        if numbering_gap:
            breaks.append(ChainBreak(**shared_fields, reason="missing_residue"))
    return breaks


def _drop_unmodeled_residues(chain: ParsedChain) -> ParsedChain:
    """Keep only residues with valid C1' coordinates."""
    keep = chain.bb_mask.astype(bool)
    if np.all(keep):
        return chain
    fixed_mask = chain.fixed_mask[keep] if chain.fixed_mask is not None else None
    return ParsedChain(
        pdb_id=chain.pdb_id,
        chain_id=chain.chain_id,
        sequence="".join(ch for ch, use in zip(chain.sequence, keep, strict=False) if use),
        aatype=chain.aatype[keep],
        atom_positions=chain.atom_positions[keep],
        atom_mask=chain.atom_mask[keep],
        residue_index=chain.residue_index[keep],
        bb_mask=chain.bb_mask[keep],
        fixed_mask=fixed_mask,
    )


def _chain_to_parsed(pdb_id: str, chain_id: str, na: NucleicAcid) -> ParsedChain:
    c1_idx = residue_constants.atom_order["C1'"]
    parsed = ParsedChain(
        pdb_id=pdb_id,
        chain_id=chain_id,
        sequence="".join(residue_constants.restypes_with_x[type_id] for type_id in na.aatype),
        aatype=na.aatype,
        atom_positions=na.atom_positions,
        atom_mask=na.atom_mask,
        residue_index=na.residue_index,
        bb_mask=na.atom_mask[:, c1_idx].astype(bool),
    )
    return _drop_unmodeled_residues(parsed)


def parse_fasta(fasta_string: str) -> tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
        fasta_string: The string contents of a FASTA file.

    Returns:
        A tuple of two lists:
        * A list of sequences.
        * A list of sequence descriptions taken from the comment lines. In the
            same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif line.startswith("#"):
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions


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
    all_atom_positions = []
    aatype = []
    all_atom_mask = []
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
        all_atom_positions.append(pos)
        all_atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    return NucleicAcid(
        atom_positions=np.array(all_atom_positions),
        atom_mask=np.array(all_atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=np.array(chain_ids),
        b_factors=np.array(b_factors),
    )


def process_pdb(file_path):
    with open(file_path) as f:
        content_string = f.read()

    prot = nucleicacid.from_pdb_string(content_string)
    return asdict(prot)


def parse_pdb(file_path: str, chain_id: str | None = None) -> list[ParsedChain]:
    """Parse a PDB file and return one ``ParsedChain`` per RNA chain."""
    from Bio.PDB.PDBParser import PDBParser

    pdb_id = Path(file_path).stem
    structure = PDBParser(QUIET=True).get_structure(pdb_id, file_path)
    results = []
    for model in list(structure.get_models())[:1]:
        for chain in model.get_chains():
            if chain_id is not None and chain.id != chain_id:
                continue
            parsed = _chain_to_parsed(pdb_id, chain.id, process_chain(chain, chain.id))
            if len(parsed.aatype) >= 2:
                results.append(parsed)
    return results


def parse_mmcif(file_path: str, chain_id: str | None = None) -> list[ParsedChain]:
    """Parse an mmCIF file and return one ``ParsedChain`` per RNA chain."""
    pdb_id = Path(file_path).stem
    result = mmcif_parsing.parse(file_id=pdb_id, mmcif_string=Path(file_path).read_text())
    if result.mmcif_object is None:
        return _parse_mmcif_atom_site_fallback(file_path, pdb_id, chain_id, result.errors)

    obj = result.mmcif_object
    chain_ids = list(obj.chain_to_seqres)
    if chain_id is not None:
        chain_ids = [chain for chain in chain_ids if chain == chain_id]
    results = []
    for current_chain_id in chain_ids:
        positions, mask = mmcif_parsing.get_atom_coords(obj, current_chain_id)
        sequence = obj.chain_to_seqres[current_chain_id]
        c1_idx = residue_constants.atom_order["C1'"]
        parsed = ParsedChain(
            pdb_id=pdb_id,
            chain_id=current_chain_id,
            sequence=sequence,
            aatype=np.array(
                [residue_constants.restype_order.get(residue, residue_constants.restype_num) for residue in sequence],
                dtype=np.int32,
            ),
            atom_positions=positions.astype(np.float32),
            atom_mask=mask.astype(np.float32),
            residue_index=np.arange(len(sequence), dtype=np.int32),
            bb_mask=mask[:, c1_idx].astype(bool),
        )
        parsed = _drop_unmodeled_residues(parsed)
        if len(parsed.aatype) >= 2:
            results.append(parsed)
    return results


def _parse_mmcif_atom_site_fallback(
    file_path: str,
    pdb_id: str,
    chain_id: str | None,
    parse_errors,
) -> list[ParsedChain]:
    """Read coordinate-only mmCIF exports that lack polymer sequence metadata."""
    try:
        structure = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=False).get_structure(
            pdb_id,
            file_path,
        )
    except Exception as error:
        raise ValueError(f"Failed to parse {file_path}: {list(parse_errors.values())}") from error

    results = []
    for model in list(structure.get_models())[:1]:
        for chain in model.get_chains():
            if chain_id is not None and chain.id != chain_id:
                continue
            parsed = _chain_to_parsed(pdb_id, chain.id, process_chain(chain, chain.id))
            if len(parsed.aatype) >= 2:
                results.append(parsed)
    if results:
        return results
    raise ValueError(f"Failed to parse {file_path}: {list(parse_errors.values())}")


def parse_structure(file_path: str, chain_id: str | None = None) -> list[ParsedChain]:
    """Auto-detect PDB versus mmCIF and parse RNA chains."""
    if Path(file_path).suffix.lower() in (".cif", ".mmcif"):
        return parse_mmcif(file_path, chain_id)
    return parse_pdb(file_path, chain_id)
