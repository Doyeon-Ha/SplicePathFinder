# Third-party notices

Bundled model implementations retain their upstream licenses. Repository-level
licensing does not replace those terms or the licenses of separately installed
dependencies and downloaded weights.

| Bundled source | Upstream | License |
| --- | --- | --- |
| `model/hyenadna/src/` | [LongSafari Hugging Face port](https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen-hf) and [HazyResearch/hyena-dna](https://github.com/HazyResearch/hyena-dna) | Hugging Face model-card declaration: BSD 3-Clause; original repository: Apache License 2.0, included in `LICENSES/Apache-2.0.txt` |
| `model/neobert/` | [chandar-lab/NeoBERT](https://huggingface.co/chandar-lab/NeoBERT), commit `5424c8efeea6491b151d62dee55a752165407430` | MIT; see `model/neobert/LICENSE` |
| `model/caduceus/` | [kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16](https://huggingface.co/kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16), commit `d89eeb853136ea64da7feb3d0c8e909771b17ae6` | Apache License 2.0 |
| `model/evo2/` | [Taykhoom/Evo2-7B-262K](https://huggingface.co/Taykhoom/Evo2-7B-262K), commit `998bb1c6f923af0b46eb80e3a909aad4202ff837` | Apache License 2.0 |

HyenaDNA-derived model code contains adaptations of FlashAttention multilayer
perceptron and block implementations. [FlashAttention](https://github.com/Dao-AILab/flash-attention)
uses the BSD 3-Clause License; its notice is included in `LICENSES/FlashAttention.txt`.

HyenaDNA and the local bidirectional integration contain project adaptations;
Caduceus contains compatibility adaptations; Evo 2 files retain the vendored
implementation. NeoBERT's gated-layer dependency has a PyTorch fallback. These
optional backbone modules accompany the shared model factory; the submitted
SplicePath architecture uses HyenaDNA.

The dynamic chunker implements the H-Net mechanism described by
[goombalab/hnet](https://github.com/goombalab/hnet), using the separately installed
Mamba-2 implementation from [state-spaces/mamba](https://github.com/state-spaces/mamba).
Official SpliceAI, Pangolin, SpliceTransformer and SpliceSelectNet weights are
downloaded separately and are not included here.
