import random
from types import SimpleNamespace

import pytest
import torch

from data.distance_span_distribution import save_dataset_distribution
from data.scaffold_datasets import (
    AutoLinkerSegment,
    ScaffoldDataset,
    ScaffoldPlan,
    TerminalSegment,
    _add_terminal_segments,
    _build_auto_suffix_masses,
    _center_atom_positions,
    _compute_length_configs,
    _load_pdb_chain_features,
    _parse_contig,
    _sample_auto_lengths,
    _set_auto_linker_distances,
)
from np import residue_constants


def _four_residue_chain_features():
    identity = torch.eye(3).repeat(4, 1, 1)
    translations = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    atom_positions = torch.zeros(4, 28, 3)
    atom_positions[:, residue_constants.atom_order["C1'"]] = translations
    return {
        "A": {
            "res_nums": [1, 2, 3, 4],
            "aatype": torch.tensor([1, 2, 3, 0]),
            "trans": translations,
            "rotmats": identity,
            "atom_positions": atom_positions,
            "atom_mask": torch.ones(4, 28, dtype=torch.bool),
        }
    }


def test_scaffold_plan_is_reused_and_sample_instantiation_is_isolated(monkeypatch, tmp_path):
    pdb_path = tmp_path / "motif.pdb"
    pdb_path.touch()
    load_calls = []

    def load_features(path):
        load_calls.append(path)
        return _four_residue_chain_features()

    monkeypatch.setattr("data.scaffold_datasets._load_pdb_chain_features", load_features)
    cfg = SimpleNamespace(
        seed=7,
        samples=SimpleNamespace(samples_per_length=2),
        scaffolding=SimpleNamespace(
            input_pdb=str(pdb_path),
            contigs="A1-2/2-3/A3-4",
            distance_span_distribution=None,
            length=None,
            generate_5prime=False,
            generate_3prime=False,
        ),
    )

    dataset = ScaffoldDataset(cfg)
    first = dataset[0]
    repeated = dataset[0]

    assert isinstance(dataset.plan, ScaffoldPlan)
    assert load_calls == [str(pdb_path)]
    assert first.keys() == repeated.keys()
    for key, value in first.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, repeated[key], equal_nan=True)
        else:
            assert value == repeated[key]

    first["trans_1"][0, 0] = 99
    first["linker_lengths"][0] = 99
    assert repeated["trans_1"][0, 0] != 99
    assert repeated["linker_lengths"][0] != 99


def test_center_atom_positions_uses_masked_c1_prime_center():
    c1_index = residue_constants.atom_order["C1'"]
    atom_positions = torch.zeros(2, 28, 3, dtype=torch.float64)
    atom_mask = torch.zeros(2, 28, dtype=torch.float64)
    atom_positions[0, c1_index] = torch.tensor([100.0, 0.0, 0.0])
    atom_positions[1, c1_index] = torch.tensor([120.0, 0.0, 0.0])
    atom_positions[0, 1] = torch.tensor([101.0, 0.0, 0.0])
    atom_mask[0, c1_index] = 1.0
    atom_mask[1, c1_index] = 1.0
    atom_mask[0, 1] = 1.0

    centered = _center_atom_positions(atom_positions, atom_mask)

    assert centered[:, c1_index].mean(dim=0).tolist() == pytest.approx(
        [0.0, 0.0, 0.0],
        abs=1e-3,
    )
    assert centered[0, 1].tolist() == pytest.approx(
        [-9.0, 0.0, 0.0],
        abs=1e-3,
    )
    assert centered[1, 1].tolist() == [0.0, 0.0, 0.0]


def test_auto_linker_distance_comes_from_reordered_motif_endpoints():
    segments = _parse_contig("A10-11/auto:3-12/A1-2")
    left, linker, right = segments
    c1_index = residue_constants.atom_order["C1'"]
    left.atom_positions = torch.zeros(2, 28, 3)
    right.atom_positions = torch.zeros(2, 28, 3)
    left.atom_positions[-1, c1_index] = torch.tensor([1.0, 0.0, 0.0])
    right.atom_positions[0, c1_index] = torch.tensor([4.0, 4.0, 0.0])
    left.atom_mask = torch.ones(2, 28, dtype=torch.bool)
    right.atom_mask = torch.ones(2, 28, dtype=torch.bool)

    _set_auto_linker_distances(segments)

    assert isinstance(linker, AutoLinkerSegment)
    assert linker.min_len == 3
    assert linker.max_len == 12
    assert linker.distance == pytest.approx(5.0)


def test_auto_linker_requires_two_fixed_flanks():
    segments = _parse_contig("auto/A1-2")
    with pytest.raises(ValueError, match="directly flanked"):
        _set_auto_linker_distances(segments)


def test_scaffold_contig_rejects_chain_breaks():
    with pytest.raises(ValueError, match="single-chain contig"):
        _parse_contig("A1/0/A2")


def test_scaffold_input_rejects_multichain_structures(monkeypatch):
    chains = [
        SimpleNamespace(id="A", get_residues=lambda: [SimpleNamespace()]),
        SimpleNamespace(id="B", get_residues=lambda: [SimpleNamespace()]),
    ]
    structure = SimpleNamespace(get_chains=lambda: iter(chains))
    parser = SimpleNamespace(get_structure=lambda *args: structure)
    monkeypatch.setattr(
        "data.scaffold_datasets.PDB.PDBParser",
        lambda **kwargs: parser,
    )

    with pytest.raises(ValueError, match="Multi-chain structures are not supported"):
        _load_pdb_chain_features("multi.pdb")


def test_terminal_generation_flags_are_independent():
    segments = _parse_contig("A1-2")
    cfg = SimpleNamespace(
        generate_5prime=True,
        terminal_5prime_length="2-4",
        generate_3prime=False,
        terminal_3prime_length="0-10",
    )

    result = _add_terminal_segments(segments, cfg)

    assert isinstance(result[0], TerminalSegment)
    assert result[0].end == "5prime"
    assert (result[0].min_len, result[0].max_len) == (2, 4)
    assert len(result) == 2


def test_length_configs_exclude_impossible_auto_linker_totals():
    segments = _parse_contig("1-3/A1/auto:2/A2")

    configs = _compute_length_configs(
        segments,
        length_str="5-5",
        generated_length_sums=[3],
    )

    assert configs == [{"total": 2}]


def test_auto_lengths_are_sampled_from_the_feasible_conditional_distribution():
    distributions = [
        [(1, 0.999), (4, 0.001)],
        [(1, 0.999), (2, 0.001)],
    ]
    suffix_masses = _build_auto_suffix_masses(distributions)

    for seed in range(10):
        sampled = _sample_auto_lengths(
            distributions,
            suffix_masses,
            random.Random(seed),
            min_total=6,
            max_total=6,
        )
        assert sampled == [4, 2]


def test_scaffold_dataset_conditions_auto_linker_on_total_length(monkeypatch, tmp_path):
    distribution_path = tmp_path / "distance_span_distribution.pt"
    dataset_prob = torch.zeros(2, 8)
    dataset_prob[0, 4] = 0.999
    dataset_prob[0, 6] = 0.001
    save_dataset_distribution(
        distribution_path,
        dataset_prob,
        support_count=torch.tensor([1.0, 0.0]),
        distance_bin_width=10.0,
        distance_max=20.0,
    )
    pdb_path = tmp_path / "motif.pdb"
    pdb_path.touch()
    monkeypatch.setattr(
        "data.scaffold_datasets._load_pdb_chain_features",
        lambda _: _four_residue_chain_features(),
    )
    cfg = SimpleNamespace(
        seed=7,
        samples=SimpleNamespace(samples_per_length=1),
        scaffolding=SimpleNamespace(
            input_pdb=str(pdb_path),
            contigs="2/A3-4/auto/A1-2",
            distance_span_distribution=str(distribution_path),
            distance_span_temperature=1.0,
            length="12-12",
            generate_5prime=True,
            terminal_5prime_length="1-1",
            generate_3prime=False,
        ),
    )

    sample = ScaffoldDataset(cfg)[0]

    assert sample["num_res"].item() == 12
    assert sample["linker_lengths"].tolist() == [1, 2, 5]
    assert sample["linker_kinds"].tolist() == [1, 3, 0]
    assert sample["diffuse_mask"].tolist() == [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0]
    assert sample["aatypes_1"].tolist() == [0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 1, 2]


def test_scaffold_dataset_samples_auto_range_uniformly_without_distribution(monkeypatch, tmp_path):
    pdb_path = tmp_path / "motif.pdb"
    pdb_path.touch()
    monkeypatch.setattr(
        "data.scaffold_datasets._load_pdb_chain_features",
        lambda _: _four_residue_chain_features(),
    )
    cfg = SimpleNamespace(
        seed=7,
        samples=SimpleNamespace(samples_per_length=1),
        scaffolding=SimpleNamespace(
            input_pdb=str(pdb_path),
            contigs="A3-4/auto:2-2/A1-2",
            distance_span_distribution=None,
            length=None,
            generate_5prime=False,
            generate_3prime=False,
        ),
    )

    sample = ScaffoldDataset(cfg)[0]

    assert sample["num_res"].item() == 6
    assert sample["linker_lengths"].tolist() == [2]
    assert torch.isnan(sample["linker_distances"][0])
    assert sample["diffuse_mask"].tolist() == [0, 0, 1, 1, 0, 0]


def test_scaffold_dataset_requires_auto_range_without_distribution(monkeypatch, tmp_path):
    pdb_path = tmp_path / "motif.pdb"
    pdb_path.touch()
    monkeypatch.setattr(
        "data.scaffold_datasets._load_pdb_chain_features",
        lambda _: _four_residue_chain_features(),
    )
    cfg = SimpleNamespace(
        seed=7,
        samples=SimpleNamespace(samples_per_length=1),
        scaffolding=SimpleNamespace(
            input_pdb=str(pdb_path),
            contigs="A3-4/auto/A1-2",
            distance_span_distribution=None,
            length=None,
            generate_5prime=False,
            generate_3prime=False,
        ),
    )

    with pytest.raises(ValueError, match="must specify a length range"):
        ScaffoldDataset(cfg)


def test_fixed_unknown_motif_residues_regenerate_sequence_only(monkeypatch, tmp_path):
    pdb_path = tmp_path / "motif.pdb"
    pdb_path.touch()
    chain_features = _four_residue_chain_features()
    chain_features["A"]["aatype"] = torch.tensor([1, 4, 3, 0])
    chain_features["A"]["atom_mask"] = torch.ones(4, 28, dtype=torch.bool)
    monkeypatch.setattr(
        "data.scaffold_datasets._load_pdb_chain_features",
        lambda _: chain_features,
    )
    cfg = SimpleNamespace(
        seed=7,
        samples=SimpleNamespace(samples_per_length=1),
        scaffolding=SimpleNamespace(
            input_pdb=str(pdb_path),
            contigs="A1-2",
            distance_span_distribution=None,
            length=None,
            generate_5prime=False,
            generate_3prime=False,
        ),
    )

    sample = ScaffoldDataset(cfg)[0]
    c1_index = residue_constants.atom_order["C1'"]
    n9_index = residue_constants.atom_order["N9"]

    assert sample["diffuse_mask"].tolist() == [0.0, 0.0]
    assert sample["aatype_diffuse_mask"].tolist() == [0.0, 1.0]
    assert sample["aatypes_1"].tolist() == [1, 4]
    assert bool(sample["atom28_fixed_mask"][1, c1_index])
    assert not bool(sample["atom28_fixed_mask"][1, n9_index])
