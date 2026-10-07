# RNA-CodeSign inference

SE(3) flow matching for RNA sequence and structure generation. This standalone
inference snapshot supports four tasks:

| Task | Conditioning | Output |
| --- | --- | --- |
| Unconditional generation | RNA length | Generated structure and sequence |
| Inverse folding | Target structure | Designed sequence and sampled structure |
| Forward folding | Sequence from a preprocessed RNA chain | Predicted structure |
| Motif scaffolding | Input PDB and fixed motif regions | Generated scaffold and sequence |

The source code is released under [Apache-2.0](LICENSE). Model weights are not
included and will be published separately.
Trained weights are required to produce useful results.
Training, circular-permutation dataset generation, relaxation, BRiQ scoring,
and external evaluation backends are outside this snapshot.

This public snapshot uses ideal atom templates derived from Apache-2.0 RhoFold
reference data, replacing the research version's NuFold constants. The token
and atom indices, model parameter shapes, sampling schedules, and output formats
are preserved, but reconstructed coordinates can differ from the research
version. Trained-weight CUDA inference and model quality have not yet been
validated for this template replacement.

## Environment

The reference server environment uses Python 3.10.18, PyTorch 2.5.0 with CUDA
12.4, and torch-scatter 2.1.2. The CLI requires an NVIDIA GPU. Run commands from
the repository root with `PYTHONPATH=src`.

On the existing server, activate the environment before running the examples:

```bash
conda activate rna-codesign
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

GPU inference must run on a GPU compute node, rather than a cluster login node.
To recreate the reference runtime in a new environment:

```bash
conda create -n rna-codesign-inference python=3.10 -y
conda activate rna-codesign-inference
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --no-index torch-scatter==2.1.2 \
  -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
python -m pip install -r requirements-server.txt
```

The CUDA wheel commands follow the official
[PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/)
and the matching [torch-scatter wheel index](https://data.pyg.org/whl/torch-2.5.0+cu124.html).
`requirements.txt` also lists the runtime dependencies without fixing every
version; the pinned server versions are the reference for this release.

## Model weights

Place a compatible schema-v2 checkpoint at `weights/model.ckpt`. Architecture
settings are read from the checkpoint; a plain state dictionary or an older
checkpoint without embedded `rna_codesign_artifact` metadata is unsupported.
The default weight variant is EMA. For a checkpoint containing only raw
weights, add `inference.weight_variant=raw` to the commands below.

For maintainers, export a research checkpoint before distributing weights:

```bash
PYTHONPATH=src python src/export_checkpoint.py \
  --input /path/to/research.ckpt --output weights/model.ckpt
```

The exporter preserves raw and available EMA weights, retains the resolved
model architecture, and recalculates metadata hashes. Optimizer state, training
data configuration, experiment identifiers, and checkpoint lineage are omitted.
It refuses to overwrite existing files. Only load trusted checkpoints and
processed feature files, which use PyTorch/pickle serialization.

## Unconditional generation

```bash
PYTHONPATH=src python src/inference_se3_flows.py \
  experiment=inference_unconditional inference.ckpt_path=weights/model.ckpt \
  'inference.samples.length_subset=[40,60]' \
  inference.samples.samples_per_length=1 inference.samples.num_batch=1
```

Increase `samples_per_length` for more samples. It must be divisible by
`num_batch`, which is the number of samples generated together for each length
request. The default is one sample on one GPU.

## Forward and inverse folding

Prepare the desired RNA chain from a PDB or mmCIF file:

```bash
PYTHONPATH=src python src/prepare_inputs.py \
  --input /path/to/target.pdb --chain-id A --output-dir inputs/target
```

Omit `--chain-id` to process all RNA chains separately. This writes one feature
pickle per chain and `inputs/target/metadata.csv`; metadata contains `pdb_name`,
`processed_path`, and `length`. Paths in generated metadata are absolute.

```bash
# Design a sequence for the input structure.
PYTHONPATH=src python src/inference_se3_flows.py \
  experiment=inference_inverse_folding inference.ckpt_path=weights/model.ckpt \
  data.dataset.metadata_path=inputs/target/metadata.csv

# Predict a structure conditioned on the input chain's sequence.
PYTHONPATH=src python src/inference_se3_flows.py \
  experiment=inference_forward_folding inference.ckpt_path=weights/model.ckpt \
  data.dataset.metadata_path=inputs/target/metadata.csv
```

Forward folding preserves the current processed-chain input interface. It does
not directly accept FASTA. The model is conditioned on the sequence; input
coordinates are used for preprocessing and a reference PDB, rather than as
fixed structural conditioning. Unknown nucleotide identities may be generated.
Forward/inverse folding produce one sample per metadata row; the unconditional
length and sample-count settings do not replicate these rows. Use a different
`inference.ckpt_id` or `paths.output_dir` for separate runs.

## Motif scaffolding

```bash
PYTHONPATH=src python src/inference_se3_flows.py \
  experiment=inference_scaffolding inference.ckpt_path=weights/model.ckpt \
  inference.scaffolding.input_pdb=/path/to/motif.pdb \
  'inference.scaffolding.contigs=5-15/A10-25/30-40' \
  inference.samples.samples_per_length=1
```

Each slash-separated contig segment is either a generated length range
(`5-15`) or a fixed motif (`A10-25`, PDB chain A, residues 10 through 25).
Only single-chain scaffolding inputs are supported. Known motif identities and
available motif atoms remain fixed. Unknown motif identities are generated.
`auto:3-20` can sample a generated segment uniformly; bare `auto` requires an
explicit `inference.scaffolding.distance_span_distribution` file, which is not
bundled here.

## Outputs

Samples are written under
`outputs/<task>/<checkpoint-id>/samples/length_<N>/<sample-name>/`.
The output contains a final sample PDB and `codesign_seqs/codesign.fa`.
Inverse folding also writes the native sequence; forward folding writes a
reference PDB; scaffolding writes motif and linker metadata. The samples root
contains `config.yaml` and `manifest.yaml` with effective settings and weight
provenance.

Set `inference.write_sample_trajectories=true` to write `bb_traj.pdb` and
`x0_traj.pdb`. Checkpoint ID defaults to the checkpoint filename stem; override
`inference.ckpt_id` to give a run a distinct output location.

## Tests

```bash
python -m pip install pytest ruff
PYTHONPATH=src python -m pytest tests/ -q
python -m ruff check src tests
```

The tests cover sampling helpers, RNA transforms, motif handling, checkpoint
export, and complete input/model/output paths for all four tasks using a small
random model on CPU. These smoke tests do not measure trained-model quality.

See [PUBLISHING.md](PUBLISHING.md) for the remaining release decisions and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution.
