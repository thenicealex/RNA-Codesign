# Modified for RNA-CodeSign RNA processing and inference.
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NucleicAcid data type."""

import dataclasses
import io
import string
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Chain import Chain

from . import residue_constants

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.


@dataclasses.dataclass(frozen=True)
class NucleicAcid:
    """NucleicAcid structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # 0-indexed number corresponding to the chain in the nucleicacid that this residue
    # belongs to.
    chain_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    chain_ids: Sequence[str] | None = None

    # Optional remark about the nucleic acid. Included as a comment in output PDB
    # files
    remark: str | None = None

    # Templates used to generate this nucleic acid (prediction-only)
    parents: Sequence[str] | None = None

    # Chain corresponding to each parent
    parents_chain_index: Sequence[int] | None = None


def from_pdb_string(pdb_str: str, chain_id: str | None = None) -> NucleicAcid:
    """Takes a PDB string and constructs a NucleicAcid object.

    WARNING: All non-standard residue types will be converted into N. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If chain_id is specified (e.g. A), then only that chain
        is parsed. Otherwise all chains are parsed.

    Returns:
      A new `NucleicAcid` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(f"Only single model PDBs are supported. Found {len(models)} models.")
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != " ":
                raise ValueError(
                    f"PDB contains an insertion code at chain {chain.id} and residue "
                    f"index {res.id[1]}. These are not supported."
                )
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
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return NucleicAcid(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        chain_ids=tuple(unique_chain_ids),
        b_factors=np.array(b_factors),
    )


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = "TER"
    return f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} {chain_name:>1}{residue_index:>4}"


def get_pdb_headers(prot: NucleicAcid, chain_id: int = 0) -> Sequence[str]:
    pdb_headers = []

    remark = prot.remark
    if remark is not None:
        pdb_headers.append(f"REMARK {remark}")

    parents = prot.parents
    parents_chain_index = prot.parents_chain_index
    if parents_chain_index is not None:
        parents = [p for i, p in zip(parents_chain_index, parents, strict=False) if i == chain_id]

    if parents is None or len(parents) == 0:
        parents = ["N/A"]

    pdb_headers.append(f"PARENT {' '.join(parents)}")

    return pdb_headers


def add_pdb_headers(prot: NucleicAcid, pdb_str: str) -> str:
    """Add pdb headers to an existing PDB string. Useful during multi-chain
    recycling
    """
    out_pdb_lines = []
    lines = pdb_str.split("\n")

    remark = prot.remark
    if remark is not None:
        out_pdb_lines.append(f"REMARK {remark}")

    parents_per_chain = None
    if prot.parents is not None and len(prot.parents) > 0:
        parents_per_chain = []
        if prot.parents_chain_index is not None:
            cur_chain = prot.parents_chain_index[0]
            parent_dict = {}
            for p, i in zip(prot.parents, prot.parents_chain_index, strict=False):
                parent_dict.setdefault(str(i), [])
                parent_dict[str(i)].append(p)

            max_idx = max([int(chain_idx) for chain_idx in parent_dict])
            for i in range(max_idx + 1):
                chain_parents = parent_dict.get(str(i), ["N/A"])
                parents_per_chain.append(chain_parents)
        else:
            parents_per_chain.append(prot.parents)
    else:
        parents_per_chain = [["N/A"]]

    make_parent_line = lambda p: f"PARENT {' '.join(p)}"

    out_pdb_lines.append(make_parent_line(parents_per_chain[0]))

    chain_counter = 0
    for i, line in enumerate(lines):
        if "PARENT" not in line and "REMARK" not in line:
            out_pdb_lines.append(line)
        if "TER" in line and "END" not in lines[i + 1]:
            chain_counter += 1
            if not chain_counter >= len(parents_per_chain):
                chain_parents = parents_per_chain[chain_counter]
            else:
                chain_parents = ["N/A"]

            out_pdb_lines.append(make_parent_line(chain_parents))

    return "\n".join(out_pdb_lines)


def to_pdb(prot: NucleicAcid, model=1, add_end=True) -> str:
    """Converts a `NucleicAcid` instance to a PDB string.

    Args:
      prot: The nucleicacid to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = [*residue_constants.restypes, "N"]
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], "N")
    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(int)
    b_factors = prot.b_factors

    aatype = np.where(aatype > residue_constants.restype_num, residue_constants.unk_restype_index, aatype)

    pdb_lines.append(f"MODEL     {model}")
    atom_index = 1
    chain_id = "A"
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        chain_number = int(prot.chain_index[i]) if prot.chain_index is not None else 0
        chain_id = prot.chain_ids[chain_number] if prot.chain_ids is not None else string.ascii_uppercase[chain_number]
        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i], strict=False
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # NucleicAcid supports only C, N, O, S, this works.
            charge = ""
            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_id:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

        if i == len(aatype) - 1 or (prot.chain_index is not None and prot.chain_index[i + 1] != prot.chain_index[i]):
            pdb_lines.append(_chain_end(atom_index, res_name_3, chain_id, residue_index[i]))
            atom_index += 1

    pdb_lines.append("ENDMDL")
    if add_end:
        pdb_lines.append("END")

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"  # Add terminating newline.


def ideal_atom_mask(prot: NucleicAcid) -> np.ndarray:
    """Computes an ideal atom mask.

    `NucleicAcid.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `NucleicAcid` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: np.ndarray | None = None,
    chain_index: np.ndarray | None = None,
) -> NucleicAcid:
    """Assembles a nucleicacid from a prediction.

    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the nucleicacid.
      remove_leading_feature_dimension: Whether to remove the leading dimension
        of the `features` values.

    Returns:
      A nucleicacid instance.
    """
    if b_factors is None:
        b_factors = np.zeros_like(result["final_atom_mask"])

    return NucleicAcid(
        aatype=features["aatype"],
        atom_positions=features["final_atom_positions"],
        atom_mask=features["final_atom_mask"],
        residue_index=features["residue_index"] + 1,
        chain_index=chain_index,
        b_factors=b_factors,
    )


def process_chain(chain: Chain, chain_id: str) -> NucleicAcid:
    """Convert a PDB chain object into a AlphaFold NucleicAcid instance.

    Forked from alphafold.common.nucleicacid.from_pdb_string

    WARNING: All non-standard residue types will be converted into N. All
        non-standard atoms will be ignored.

    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.

    Took out lines 110-112 since that would mess up CDR numbering.

    Args:
        chain: Instance of Biopython's chain class.

    Returns:
        NucleicAcid object with nucleicacid features.
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


def to_modelcif(prot: NucleicAcid) -> str:
    """
    Converts a `NucleicAcid` instance to a ModelCIF string. Chains with identical modelled coordinates
    will be treated as the same polymer entity. But note that if chains differ in modelled regions,
    no attempt is made at identifying them as a single polymer entity.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      ModelCIF string.
    """

    import ihm
    import modelcif
    import modelcif.dumper
    import modelcif.model
    import modelcif.qa_metric

    restypes = [*residue_constants.restypes, "X"]
    atom_types = residue_constants.atom_types

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors
    chain_index = prot.chain_index

    n = aatype.shape[0]
    if chain_index is None:
        chain_index = [0 for i in range(n)]

    system = modelcif.System(title="OpenFold prediction")

    # Finding chains and creating entities
    seq_ids = np.zeros(n, dtype=int)
    seqs = {}
    seq = []
    last_chain_idx = None
    for i in range(n):
        if last_chain_idx is not None and last_chain_idx != chain_index[i]:
            seqs[last_chain_idx] = seq
            seq = []
        seq.append(restypes[aatype[i]])
        seq_ids[i] = len(seq)
        last_chain_idx = chain_index[i]
    # finally add the last chain
    seqs[last_chain_idx] = seq

    # now reduce sequences to unique ones (note this won't work if different asyms have different unmodelled regions)
    unique_seqs = {}
    for chain_idx, seq_list in seqs.items():
        seq = "".join(seq_list)
        if seq in unique_seqs:
            unique_seqs[seq].append(chain_idx)
        else:
            unique_seqs[seq] = [chain_idx]

    # adding 1 entity per unique sequence
    entities_map = {}
    for key, value in unique_seqs.items():
        model_e = modelcif.Entity(key, alphabet=ihm.RNAAlphabet, description="RNA subunit")
        for chain_idx in value:
            entities_map[chain_idx] = model_e

    chain_tags = string.ascii_uppercase
    asym_unit_map = {}
    for chain_idx in set(chain_index):
        # Define the model assembly
        chain_id = chain_tags[chain_idx]
        asym = modelcif.AsymUnit(entities_map[chain_idx], details=f"Model subunit {chain_id}", id=chain_id)
        asym_unit_map[chain_idx] = asym
    modeled_assembly = modelcif.Assembly(asym_unit_map.values(), name="Modeled assembly")

    class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
        name = "pLDDT"
        software = None
        description = "Predicted lddt"

    class _GlobalPLDDT(modelcif.qa_metric.Global, modelcif.qa_metric.PLDDT):
        name = "pLDDT"
        software = None
        description = "Global pLDDT, mean of per-residue pLDDTs"

    class _MyModel(modelcif.model.AbInitioModel):
        def get_atoms(self):
            # Add all atom sites.
            for i in range(n):
                for atom_name, pos, mask, b_factor in zip(
                    atom_types, atom_positions[i], atom_mask[i], b_factors[i], strict=False
                ):
                    if mask < 0.5:
                        continue
                    element = atom_name[0]  # Protein supports only C, N, O, S, this works.
                    yield modelcif.model.Atom(
                        asym_unit=asym_unit_map[chain_index[i]],
                        type_symbol=element,
                        seq_id=int(seq_ids[i]),
                        atom_id=atom_name,
                        x=pos[0],
                        y=pos[1],
                        z=pos[2],
                        het=False,
                        biso=b_factor,
                        occupancy=1.00,
                    )

        def add_scores(self):
            # local scores
            plddt_per_residue = {}
            for i in range(n):
                for mask, b_factor in zip(atom_mask[i], b_factors[i], strict=False):
                    if mask < 0.5:
                        continue
                    # add 1 per residue, not 1 per atom
                    if chain_index[i] not in plddt_per_residue:
                        # first time a chain index is seen: add the key and start the residue dict
                        plddt_per_residue[chain_index[i]] = {int(seq_ids[i]): b_factor}
                    if int(seq_ids[i]) not in plddt_per_residue[chain_index[i]]:
                        plddt_per_residue[chain_index[i]][int(seq_ids[i])] = b_factor
            plddts = []
            for chain_idx in plddt_per_residue:
                for residue_idx in plddt_per_residue[chain_idx]:
                    plddt = plddt_per_residue[chain_idx][residue_idx]
                    plddts.append(plddt)
                    self.qa_metrics.append(_LocalPLDDT(asym_unit_map[chain_idx].residue(residue_idx), plddt))
            # global score
            self.qa_metrics.append(_GlobalPLDDT(np.mean(plddts)))

    # Add the model and modeling protocol to the file and write them out:
    model = _MyModel(assembly=modeled_assembly, name="Best scoring model")
    model.add_scores()

    model_group = modelcif.model.ModelGroup([model], name="All models")
    system.model_groups.append(model_group)

    fh = io.StringIO()
    modelcif.dumper.write(fh, [system])
    return fh.getvalue()
