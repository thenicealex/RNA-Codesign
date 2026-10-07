# Third-party source attribution

The project is distributed under Apache-2.0. Existing copyright and license
notices are retained, and adapted MIT components retain their original terms.

| Files | Source and terms |
| --- | --- |
| `src/np/rna_template_data.json` | [RhoFold RNA constants](https://github.com/ml4bio/RhoFold/blob/24ef5b9d19349bc6a3ecd8075742334e471cbd7d/rhofold/utils/constants.py), Apache-2.0; full text in `licenses/RhoFold-Apache-2.0.txt` |
| `src/np/residue_constants.py` | RNA-CodeSign implementation that converts the licensed RhoFold reference data to checkpoint-compatible atom and torsion layouts |
| `src/models/ipa_pytorch.py` | Adaptation of [OpenFold invariant point attention](https://github.com/aqlaboratory/openfold), Apache-2.0 |
| `src/data/data_transforms.py`, `src/data/mmcif_parsing.py`, `src/np/nucleicacid.py`, `src/utils/rigid_utils.py`, `src/utils/tensor_utils.py` | Apache-2.0 helpers with retained DeepMind and AlQuraishi Laboratory notices, adapted for RNA processing |
| `src/utils/feats.py` | Adaptation of the individually [Apache-2.0-licensed NuFold/OpenFold helper](https://github.com/kiharalab/NuFold/blob/master/nufold/model/openfold/feats.py); retains Y.K / Kihara Lab, AlQuraishi Laboratory, and DeepMind notices |
| `src/utils/so3_utils.py` | Adaptations identified in the source from [se3_diffusion](https://github.com/jasonkyuyim/se3_diffusion) and [geomstats](https://github.com/geomstats/geomstats), MIT; full texts in `licenses/` |

The NuFold constant module and RNA stereo-chemical resource from the research
snapshot are not distributed. Unused helpers citing twisted_diffusion_sampler
were removed. The time embedding is a new implementation of the mathematical
sinusoidal formula, verified to preserve the existing checkpoint's numerical
features; the unlicensed cited DDPM snippet is not included.

RhoFold reference coordinates differ from the research version's original
ideal atom tables. This changes reconstructed coordinates without changing
model weight keys, atom indices, or sampling schedules. See the README for the
current validation limits.

Runtime packages are installed separately and retain their respective licenses.
