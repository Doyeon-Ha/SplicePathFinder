"""Score variants with the OFFICIAL Illumina SpliceAI (get_delta_scores).

Uses the official ``spliceai`` package (TensorFlow/Keras, Jaganathan 2019
weights) via ``Annotator`` + ``get_delta_scores`` — the tool's own native
inference — with a GENCODE v49 annotation (built by
``process_input/build_spliceai_annotation_v49.py``) and hg38 reference, so the
whole benchmark stays on v49/hg38.

Scoring policy (matches project convention, ONBOARDING §5-8):
  - distance/window: ``-D 2500`` (matched to head/Pangolin; native default 50)
  - mask: ``-M 0`` (raw Δprobability; matches Pangolin -m False)
  - headline per variant = max of the 4 directional delta scores
    (DS_AG, DS_AL, DS_DG, DS_DL) over all overlapping genes.

Coverage caveats (native SpliceAI behaviour, logged not hidden):
  - delins / MNV (len(ref)>1 AND len(alt)>1) are NOT scored by get_delta_scores
    (returns '.'); recorded as no_call.
  - variants with no overlapping v49 gene, or REF not matching the reference,
    return no entries → no_call.
  no_call variants get score 0.0 and a scored=False flag (report both pooled
  and scored-only metrics downstream; no silent truncation).

Runs on GPU (TensorFlow). Shardable via --shard-idx/--n-shards for Slurm arrays.

``--torch-checkpoint`` runs PyTorch SpliceAI-10k networks through the same official
``Annotator`` + ``get_delta_scores``: the released ``spliceai/utils.py``
(``--spliceai-utils``) is loaded with ``keras`` and ``pkg_resources`` stubbed and its
five models replaced by adapters, so every step but the networks is the released
code. No TensorFlow is needed then. One path fills all five slots; five paths fill
one slot each, so the released five-model mean becomes their mean. A path is a
project ``SpliceAI_10k`` checkpoint, a ``train/train_spliceai_illumina.py``
checkpoint (``SpliceAIIllumina``), or a released Keras ``spliceai{1..5}.h5`` loaded
into ``SpliceAIIllumina``.
Output: ``<output-dir>/scores[_shard<k>].npz`` with idx/label/delta_* /scored
plus gene_region/variant_kind passthrough for stratified metrics, and ``raw``:
every ``get_delta_scores`` entry joined by ``,``. The raw entries keep each
gene's four DP offsets (genomic-forward, relative to POS), which a consequence
rule needs and the maxima drop.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

SPLICEAI_UTILS = ("./third_party/spliceai_official/"
                  "spliceai-1.3.1/utils.py")


class _Record:
    """Minimal pysam-VariantRecord stand-in for get_delta_scores."""

    __slots__ = ("chrom", "pos", "ref", "alts")

    def __init__(self, chrom: str, pos: int, ref: str, alt: str):
        self.chrom = chrom
        self.pos = int(pos)        # 1-based VCF position
        self.ref = ref.upper()
        self.alts = [alt.upper()]


def _decode(x: object) -> str:
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def _load_variants(h5_path: str) -> list[dict]:
    """Read variants (+ optional gene_region / variant_kind) from an H5."""
    with h5py.File(h5_path, "r") as f:
        chrom = f["chrom"][:]
        pos = f["pos"][:].astype(np.int64)
        ref = f["ref"][:]
        alt = f["alt"][:]
        label = f["label"][:].astype(np.int64)
        gene_region = f["gene_region"][:] if "gene_region" in f else None
        variant_kind = f["variant_kind"][:] if "variant_kind" in f else None
    out = []
    for i in range(len(pos)):
        rec = {
            "idx": i,
            "chrom": _decode(chrom[i]),
            "pos": int(pos[i]),
            "ref": _decode(ref[i]).upper(),
            "alt": _decode(alt[i]).upper(),
            "label": int(label[i]),
            "gene_region": _decode(gene_region[i]) if gene_region is not None else "",
            "variant_kind": _decode(variant_kind[i]) if variant_kind is not None else "",
        }
        out.append(rec)
    return out


def _parse_delta(entries: list[str]) -> tuple[float, float, float, float, bool]:
    """Parse get_delta_scores output → (acc_max, don_max, delta_max, scored).

    Each entry: ``ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL``.
    Takes the max over all overlapping-gene entries. '.' fields (delins/MNV) →
    unscored. Returns 0.0s and scored=False when no numeric entry exists.
    """
    best_ag = best_al = best_dg = best_dl = None
    for e in entries:
        parts = e.split("|")
        if len(parts) < 6:
            continue
        try:
            ag, al, dg, dl = (float(parts[k]) for k in range(2, 6))
        except ValueError:
            continue  # '.' delins/MNV entry
        best_ag = ag if best_ag is None else max(best_ag, ag)
        best_al = al if best_al is None else max(best_al, al)
        best_dg = dg if best_dg is None else max(best_dg, dg)
        best_dl = dl if best_dl is None else max(best_dl, dl)
    if best_ag is None:
        return 0.0, 0.0, 0.0, False
    acc_max = max(best_ag, best_al)
    don_max = max(best_dg, best_dl)
    return acc_max, don_max, max(acc_max, don_max), True  # noqa: RET504


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant-h5", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--annotation",
        default="./spliceai_train/reference/spliceai_annotation_v49.txt",
    )
    p.add_argument(
        "--ref-fasta", default="./genome/GRCh38.fa"
    )
    p.add_argument("--dist", type=int, default=2500)
    p.add_argument("--mask", type=int, default=0, choices=[0, 1])
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--torch-checkpoint", nargs="+", default=None,
                   help="one or five PyTorch SpliceAI-10k checkpoints (or released "
                        "Keras .h5 files) to run instead of the released ensemble")
    p.add_argument("--spliceai-utils", default=SPLICEAI_UTILS,
                   help="released spliceai/utils.py, used with --torch-checkpoint")
    return p.parse_args()


def _torch_spliceai_api(utils_path: str, checkpoints: list[str]) -> tuple:
    """Return the released ``Annotator`` and ``get_delta_scores`` on PyTorch networks.

    ``utils_path`` is loaded with ``keras.models.load_model`` returning one adapter
    per call: the same adapter for all five ensemble slots when one path is given,
    the ``i``-th network for the ``i``-th slot when five are. An adapter takes the
    released one-hot input ``(1, L, 4)`` and returns softmax probabilities for the
    central ``L - 10000`` positions in the released channel order (neither,
    acceptor, donor). ``SpliceAIIllumina`` emits that order; the project
    ``SpliceAI_10k`` emits (acceptor, donor, neither) and is reordered. Repeated
    calls with the same input are computed once.

    Args:
        utils_path (str): released ``spliceai/utils.py``.
        checkpoints (list[str]): One or five networks (see the module docstring).

    Returns:
        tuple: ``(Annotator, get_delta_scores)``.

    Raises:
        ValueError: If neither one nor five paths are given.
    """
    import importlib.util
    import itertools
    import sys
    import types

    import torch

    from evaluate.evaluate_splicevardb_sav import _load_spliceai_model
    from model.spliceai import SpliceAIIllumina

    if len(checkpoints) not in (1, 5):
        raise ValueError(f"give one or five networks, got {len(checkpoints)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(path: str) -> tuple:
        if path.endswith(".h5"):
            model = SpliceAIIllumina()
            model.load_keras_h5(path)
            return model.to(device).eval(), None
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and ckpt.get("model_class") == "SpliceAIIllumina":
            model = SpliceAIIllumina(**ckpt["model_kwargs"])
            model.load_state_dict(ckpt["state_dict"])
            return model.to(device).eval(), None
        model, _, _ = _load_spliceai_model(checkpoint_path=path, context_length=10000,
                                           target_length=5000, device=device)
        return model, [2, 0, 1]

    class _Adapter:
        def __init__(self, model: torch.nn.Module, order: list[int] | None) -> None:
            self._model, self._order = model, order
            self._key, self._out = None, None

        def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
            key = x.tobytes()
            if key != self._key:
                t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
                t = t.permute(0, 2, 1).to(device)
                if self._order is not None:  # SpliceAI_10k slices a fixed length
                    self._model.TARGET_LEN = t.shape[-1] - self._model.S
                with torch.no_grad():
                    prob = torch.softmax(self._model(t).float(), dim=-1)
                if self._order is not None:
                    prob = prob[..., self._order]
                self._key, self._out = key, prob.cpu().numpy()
            return self._out

    adapters = [_Adapter(*load(path)) for path in checkpoints]
    calls = itertools.count()

    def load_model(*_a: object, **_k: object) -> _Adapter:
        return adapters[next(calls) % len(adapters)]

    keras_models = types.ModuleType("keras.models")
    keras_models.load_model = load_model
    keras = types.ModuleType("keras")
    keras.models = keras_models
    pkg_resources = types.ModuleType("pkg_resources")
    pkg_resources.resource_filename = lambda _pkg, path: path
    sys.modules.update({"keras": keras, "keras.models": keras_models,
                        "pkg_resources": pkg_resources})
    spec = importlib.util.spec_from_file_location("spliceai_released_utils", utils_path)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)

    def one_hot_encode(seq: str) -> np.ndarray:
        # The released body; numpy 2 removed np.fromstring's binary mode, and
        # frombuffer over the latin-1 bytes gives the same codes (checked equal
        # under numpy 1.26.4, the released environment).
        table = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0],
                            [0, 0, 1, 0], [0, 0, 0, 1]])
        seq = seq.upper().replace("A", "\x01").replace("C", "\x02")
        seq = seq.replace("G", "\x03").replace("T", "\x04").replace("N", "\x00")
        return table[np.frombuffer(seq.encode("latin-1"), np.int8) % 5]

    utils.one_hot_encode = one_hot_encode
    return utils.Annotator, utils.get_delta_scores


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    if args.torch_checkpoint:
        Annotator, get_delta_scores = _torch_spliceai_api(args.spliceai_utils,
                                                          args.torch_checkpoint)
        logger.info("PyTorch network %s in the released get_delta_scores",
                    args.torch_checkpoint)
    else:
        import tensorflow as tf
        from spliceai.utils import Annotator, get_delta_scores

        tf.keras.utils.disable_interactive_logging()  # no predict() progress bars
        gpus = tf.config.list_physical_devices("GPU")
        logger.info("TensorFlow %s; GPUs visible: %d", tf.__version__, len(gpus))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = _load_variants(args.variant_h5)
    n_total = len(variants)
    shard = variants[args.shard_idx :: args.n_shards] if args.n_shards > 1 else variants
    logger.info(
        "Loaded %d variants; shard %d/%d -> %d variants",
        n_total, args.shard_idx, args.n_shards, len(shard),
    )

    ann = Annotator(args.ref_fasta, args.annotation)
    # Force Keras .predict(verbose=0): keeps the compiled inference graph (fast
    # for the constant-length SNV inputs that dominate) while silencing the
    # per-call progress logging get_delta_scores would otherwise emit.
    import functools
    for _m in ann.models:
        _m.predict = functools.partial(_m.predict, verbose=0)
    logger.info("Annotator ready (%d genes, 5 models)", len(ann.genes))

    idxs, labels, dmax, dacc, ddon, scored, raws = [], [], [], [], [], [], []
    regions, kinds = [], []
    n_nocall = 0
    for n, v in enumerate(shard):
        rec = _Record(v["chrom"], v["pos"], v["ref"], v["alt"])
        try:
            entries = get_delta_scores(rec, ann, args.dist, args.mask)
        except Exception as exc:  # noqa: BLE001 - one bad variant must not kill shard
            logger.warning("variant idx=%d (%s:%d) failed: %s",
                           v["idx"], v["chrom"], v["pos"], exc)
            entries = []
        acc_max, don_max, delta_max, was_scored = _parse_delta(entries)
        if not was_scored:
            n_nocall += 1
        idxs.append(v["idx"]); labels.append(v["label"])
        dmax.append(delta_max); dacc.append(acc_max); ddon.append(don_max)
        scored.append(was_scored)
        raws.append(",".join(entries))
        regions.append(v["gene_region"]); kinds.append(v["variant_kind"])
        if (n + 1) % 500 == 0:
            logger.info("  %d/%d (no_call so far: %d)", n + 1, len(shard), n_nocall)

    suffix = f"_shard{args.shard_idx}" if args.n_shards > 1 else ""
    out_npz = out_dir / f"scores{suffix}.npz"
    np.savez(
        out_npz,
        idx=np.asarray(idxs, dtype=np.int64),
        label=np.asarray(labels, dtype=np.int64),
        delta_max=np.asarray(dmax, dtype=np.float32),
        delta_acc_max=np.asarray(dacc, dtype=np.float32),
        delta_don_max=np.asarray(ddon, dtype=np.float32),
        scored=np.asarray(scored, dtype=bool),
        gene_region=np.asarray(regions, dtype=object),
        variant_kind=np.asarray(kinds, dtype=object),
        raw=np.asarray(raws, dtype=object),
    )
    summary = {
        "variant_h5": args.variant_h5,
        "annotation": args.annotation,
        "ref_fasta": args.ref_fasta,
        "dist": args.dist,
        "mask": args.mask,
        "shard_idx": args.shard_idx,
        "n_shards": args.n_shards,
        "n_total": n_total,
        "n_shard": len(shard),
        "n_no_call": n_nocall,
        "torch_checkpoint": args.torch_checkpoint,
    }
    with open(out_dir / f"summary{suffix}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote %s (%d variants, %d no_call)", out_npz, len(shard), n_nocall)


if __name__ == "__main__":
    main()
