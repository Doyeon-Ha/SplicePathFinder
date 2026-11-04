# Third-party notices

Bundled implementations retain their upstream terms. Dataset, dependency and
external model licenses apply independently of the project's original code.

| Bundled source | Upstream | License |
| --- | --- | --- |
| `model/hyenadna/src/` | [LongSafari Hugging Face port](https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen-hf) and [HazyResearch/hyena-dna](https://github.com/HazyResearch/hyena-dna) | Hugging Face model-card declaration: BSD 3-Clause; original repository: Apache License 2.0, included in `LICENSES/Apache-2.0.txt` |
| FlashAttention-derived blocks within HyenaDNA | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | BSD 3-Clause, included in `LICENSES/FlashAttention.txt` |

The HyenaDNA implementation and bidirectional integration include project
adaptations. The dynamic chunker implements the H-Net mechanism described by
[goombalab/hnet](https://github.com/goombalab/hnet), using the separately installed
Mamba-2 implementation from [state-spaces/mamba](https://github.com/state-spaces/mamba).

Official SpliceAI, Pangolin, SpliceTransformer and SpliceSelectNet implementations
and weights are obtained from their publishers in separate environments. The
repository's scoring adapters do not bundle those weights. SpliceVarDB data is
obtained directly from its authors under the applicable data terms.
