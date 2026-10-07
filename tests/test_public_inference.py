from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from data import data_transforms
from export_checkpoint import export_checkpoint
from models.flow_module import FlowModule
from np import residue_constants as rc
from prepare_inputs import prepare_inputs
from utils import feats
from utils.checkpoint_bundle import ARTIFACT_KEY, EMA_STATE_KEY, add_checkpoint_artifact, load_checkpoint_bundle
from utils.inference_task import build_inference_dataset
from utils.pdb_io import write_prot_to_pdb


@pytest.fixture
def input_pdb(tmp_path):
    aatypes = torch.tensor([[0, 1, 2, 3]])
    translations = torch.tensor([[[0.0, 10.0, 1.0], [5.0, 10.0, 1.0], [10.0, 10.0, 1.0], [15.0, 10.0, 1.0]]])
    rotations = torch.eye(3)[None, None].repeat(1, 4, 1, 1)
    masks = data_transforms.make_atom23_masks({"aatype": aatypes})
    torsions = torch.zeros(1, 4, 9, 2)
    torsions[..., 1] = 1
    positions, _ = feats.to_atom28(translations, rotations, torsions, aatypes, masks)
    path = tmp_path / "rna.pdb"
    write_prot_to_pdb(positions[0].numpy(), str(path), aatypes[0].numpy(), no_indexing=True)
    return path


@pytest.mark.parametrize("task", ["unconditional", "forward_folding", "inverse_folding", "scaffolding"])
def test_four_tasks_sample_and_write_outputs(task, input_pdb, tmp_path):
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="inference", overrides=[f"experiment=inference_{task}"])
    # A small random model verifies the complete data/model/output path on CPU.
    cfg.model.node_embed_size = 32
    cfg.model.edge_embed_size = 16
    cfg.model.ipa.c_hidden = 8
    cfg.model.ipa.no_heads = 2
    cfg.model.ipa.no_qk_points = 2
    cfg.model.ipa.no_v_points = 2
    cfg.model.ipa.seq_tfmr_num_layers = 1
    cfg.model.ipa.num_blocks = 1
    cfg.interpolant.sampling.num_timesteps = 3
    cfg.inference.samples.length_subset = [4]
    cfg.inference.samples_dir = str(tmp_path / task)
    cfg.paths.output_dir = str(tmp_path)
    if task in {"forward_folding", "inverse_folding"}:
        cfg.data.dataset.metadata_path = str(prepare_inputs(str(input_pdb), str(tmp_path / "features")))
    elif task == "scaffolding":
        cfg.inference.scaffolding.input_pdb = str(input_pdb)
        cfg.inference.scaffolding.contigs = "A1-2/2"
    dataset = build_inference_dataset(task, cfg.inference, cfg.inference.samples, cfg.data.dataset)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    module = FlowModule(cfg).eval()
    with torch.no_grad():
        samples = module.predict_step(batch, 0)
    assert len(samples) == 1
    assert Path(samples[0].sample_path).is_file()
    sequence = (Path(samples[0].sample_dir) / "codesign_seqs/codesign.fa").read_text().splitlines()[1]
    assert len(sequence) == 4
    assert set(sequence).issubset(set(rc.restypes))
    if task == "forward_folding":
        assert sequence == "".join(rc.restypes)
    elif task == "scaffolding":
        assert sequence[:2] == "".join(rc.restypes[1:3])
    elif task == "inverse_folding":
        assert "rna_A" in samples[0].sample_dir


def test_checkpoint_export_keeps_weights_and_removes_research_metadata(tmp_path):
    raw = torch.tensor([1.0])
    ema = torch.tensor([2.0])
    checkpoint = {
        "state_dict": {"model.weight": raw},
        "optimizer_states": [{"private": "internal-data-path"}],
        EMA_STATE_KEY: {"initialized": True, "params": {"weight": ema}},
    }
    config = OmegaConf.create({"model": {"width": 1}, "data": {"path": "internal-data-path"}})
    add_checkpoint_artifact(checkpoint, config, lineage={"path": "internal-data-path"})
    checkpoint[ARTIFACT_KEY]["run_id"] = "private-run"
    input_path = tmp_path / "research.ckpt"
    output_path = tmp_path / "public.ckpt"
    torch.save(checkpoint, input_path)
    export_checkpoint(str(input_path), str(output_path))
    bundle = load_checkpoint_bundle(str(output_path))
    assert set(bundle.training_config) == {"model"}
    assert bundle.metadata["lineage"] == {}
    assert "run_id" not in bundle.metadata
    assert "optimizer_states" not in bundle.checkpoint
    assert torch.equal(bundle.get_state_dict("raw")["model.weight"], raw)
    assert torch.equal(bundle.get_state_dict("ema")["model.weight"], ema)
    with pytest.raises(FileExistsError):
        export_checkpoint(str(input_path), str(output_path))
