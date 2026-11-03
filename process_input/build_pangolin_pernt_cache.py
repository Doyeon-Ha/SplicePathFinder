"""Build a Pangolin per-nt probability cache for the learned SAV head (spec 019).

Swaps the learned_mamba2 head's frozen base from med_ac_ep6 / SpliceAI-10k to
Pangolin (tkzeng/Pangolin), keeping the head structure unchanged. For every
SpliceVarDB full-H5 variant, extract a variant-centered chunk, run all 12
Pangolin models, and cache **all 12 per-nt channels** for ref and alt —
row-aligned to the med_ac / SpliceAI caches so the same gene-grouped split
(seed42) trains/evaluates an identical head.

Key decisions (spec 019 research):
- **12-channel provenance (R-001, DD-001).** Pangolin loads 12 models as
  ``for i in [0,2,4,6]: for j in [1,2,3]`` → ``models[3k:3k+3]`` is tissue-group
  ``k``. Each model's ``forward`` emits 12 channels ``[out1(softmax 2ch),
  out2(sigmoid 1ch)]×4`` → channel ``3k``=not-splice, ``3k+1``=splice-prob,
  ``3k+2``=usage for group ``k``. **Cache channel c = mean over
  ``models[3(c//3):3(c//3)+3]`` of each model's forward output channel c.** This
  reproduces native scoring, which reads channel ``[1,4,7,10][j]=3j+1`` from
  group ``j`` (``pangolin.compute_score``). Main splice-prob = [1,4,7,10],
  usage = [2,5,8,11].
- **Context length computed at runtime (R-002, DD-002).** ``CL = 2·Σ(AR·(W−1))``
  = 10,000. Input = ``TARGET_LENGTH + CL`` = 15,001; the model crops CL//2 from
  each side, so the variant at chunk index ``CLP = VOFF + CL//2 = 7500`` maps to
  output index ``VOFF = 2500`` (center of the 5,001 window).
- **Strand / one-hot = Pangolin convention, no double-RC (R-003, DD-005).**
  ``_build_indel_chunk`` already applies strand reverse-complement and centers
  the variant (variant at chunk index ``CLP``; RC'd for ``-`` genes). We decode
  its output to a string (model orientation) and feed Pangolin's
  ``one_hot_encode(seq, "+")`` — **no further strand handling** — and do **not**
  flip the output back, matching the med_ac / SpliceAI cache convention (head
  reads raw ref/alt in a consistent frame; native's ``[::-1]`` flip-back and
  indel length-alignment are scoring-only).
- Row order == full H5 ``(chrom,pos,ref,alt)`` (asserted; FR-002).
"""

from __future__ import annotations

import argparse
import glob
import logging

import h5py
import numpy as np
import torch

from scripts.evaluate_splicevardb_sav_indel_v3 import _build_indel_chunk
from scripts.evaluate_splicevardb_sav_multichunk import _decode_seq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_LENGTH: int = 5001  # per-nt output length (odd → symmetric window)
VOFF: int = TARGET_LENGTH // 2  # 2500 = variant-centered output index
NUM_CHANNELS: int = 12  # 4 tissue-groups × [not-splice, splice-prob, usage]

# Byte-exact copy of ``pangolin.pangolin.IN_MAP`` / ``one_hot_encode`` (see
# ``_one_hot_encode``). Reproduced locally because ``pangolin.pangolin`` imports
# PyVCF/gffutils at module top (absent in this venv) and we only need the encoder.
_IN_MAP: np.ndarray = np.asarray(
    [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)


def _one_hot_encode(seq: str, strand: str) -> np.ndarray:
    """One-hot encode a sequence exactly as Pangolin does (A/C/G/T/N → IN_MAP).

    Byte-for-byte identical to ``pangolin.pangolin.one_hot_encode``. We always
    call it with ``strand="+"`` here because ``_build_indel_chunk`` has already
    applied the strand reverse-complement (no double-RC).

    Args:
        seq (str): nucleotide sequence.
        strand (str): ``"+"`` (encode as-is) or ``"-"`` (reverse-complement).

    Returns:
        np.ndarray: shape ``(len(seq), 4)`` one-hot (N → all-zero row).
    """
    seq = seq.upper().replace("A", "1").replace("C", "2")
    seq = seq.replace("G", "3").replace("T", "4").replace("N", "0")
    if strand == "+":
        arr = np.asarray(list(map(int, list(seq))))
    else:  # "-": reverse-complement
        arr = np.asarray(list(map(int, list(seq[::-1]))))
        arr = (5 - arr) % 5
    return _IN_MAP[arr.astype("int8")]


def _decode(arr: np.ndarray) -> list[str]:
    """Decode an object/bytes H5 column to a list of str.

    Args:
        arr (np.ndarray): object or bytes array.

    Returns:
        list[str]: decoded strings.
    """
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def _shard_range(n: int, shard_index: int, num_shards: int) -> tuple[int, int]:
    """Contiguous [start, end) row range for a shard.

    Args:
        n (int): total rows.
        shard_index (int): 0-based shard index.
        num_shards (int): shard count.

    Returns:
        tuple[int, int]: (start, end) row indices.
    """
    per = -(-n // num_shards)
    start = shard_index * per
    return start, min(n, start + per)


def _load_pangolin_models(
    models_dir: str, device: torch.device
) -> tuple[list[torch.nn.Module], int]:
    """Load the 12 Pangolin models and compute the context length.

    Mirrors ``pangolin.pangolin.main``: ``for i in [0,2,4,6]: for j in [1,2,3]``
    so ``models[3k:3k+3]`` is tissue-group ``k`` (R-001).

    Args:
        models_dir (str): dir holding ``final.{j}.{i}.3.v2`` weight files.
        device (torch.device): device to place models on.

    Returns:
        tuple[list[torch.nn.Module], int]: (12 eval-mode models, context length
            ``CL = 2·Σ(AR·(W−1))``).
    """
    from pangolin.model import AR, L, Pangolin, W

    models: list[torch.nn.Module] = []
    for i in (0, 2, 4, 6):
        for j in (1, 2, 3):
            m = Pangolin(L, W, AR)
            weights = torch.load(
                f"{models_dir}/final.{j}.{i}.3.v2", map_location=device
            )
            m.load_state_dict(weights)
            m.to(device).eval()
            models.append(m)
    cl = int(2 * np.sum(AR * (W - 1)))
    logger.info("loaded %d Pangolin models, CL=%d", len(models), cl)
    return models, cl


def _pangolin_forward_probs(
    models: list[torch.nn.Module],
    seqs: list[str],
    device: torch.device,
    target_length: int,
) -> np.ndarray:
    """Run the 12 Pangolin models and assemble the group-ensemble 12-channel out.

    ``seqs`` are model-orientation strings (already strand-RC'd and centered by
    ``_build_indel_chunk``), so ``one_hot_encode(seq, "+")`` applies no further
    RC. Cache channel ``c`` = mean over group ``c//3``'s three models of each
    model's forward output channel ``c`` (R-001).

    Args:
        models (list[torch.nn.Module]): 12 Pangolin models (group k = [3k:3k+3]).
        seqs (list[str]): input sequences of length ``target_length + CL``.
        device (torch.device): compute device.
        target_length (int): per-nt output length (model crops CL).

    Returns:
        np.ndarray: shape ``(len(seqs), target_length, 12)`` float32.
    """
    x = np.stack([_one_hot_encode(s, "+") for s in seqs])  # (B, L, 4)
    xt = torch.from_numpy(x).float().transpose(1, 2).to(device)  # (B, 4, L)
    b = xt.shape[0]
    out = np.zeros((b, target_length, NUM_CHANNELS), dtype=np.float32)
    for k in range(4):
        chans = [3 * k, 3 * k + 1, 3 * k + 2]
        acc = np.zeros((b, target_length, 3), dtype=np.float32)
        for m in models[3 * k : 3 * k + 3]:
            o = m(xt)[:, chans, :].cpu().numpy()  # (B, 3, target_length)
            acc += np.transpose(o, (0, 2, 1))  # (B, target_length, 3)
        out[:, :, 3 * k : 3 * k + 3] = acc / 3.0
    return out


def _assert_channel_provenance(probs: np.ndarray, tol: float = 1e-2) -> None:
    """Verify the 12-channel grouping via structural invariants (FR-002).

    Each group ``k``'s ``[3k, 3k+1]`` are a 2-way softmax (not-splice, splice),
    so they must sum to ~1; ``3k+2`` is an independent sigmoid usage. If the
    group→channel mapping (R-001) were wrong, the softmax pairs would not sum to
    1. All channels must lie in [0, 1].

    Args:
        probs (np.ndarray): a per-nt cache slice ``(*, target_length, 12)``.
        tol (float): tolerance for the softmax-pair sum-to-1 check.

    Raises:
        AssertionError: if any invariant is violated.
    """
    assert probs.shape[-1] == NUM_CHANNELS, f"expected 12 channels, got {probs.shape}"
    assert float(probs.min()) >= -tol and float(probs.max()) <= 1.0 + tol, (
        f"channel values out of [0,1]: min={probs.min()} max={probs.max()}"
    )
    for k in range(4):
        pair_sum = probs[..., 3 * k] + probs[..., 3 * k + 1]
        dev = float(np.abs(pair_sum - 1.0).max())
        assert dev <= tol, (
            f"group {k} softmax pair (ch {3 * k},{3 * k + 1}) sum dev {dev} > {tol}"
            " — 12-channel provenance (R-001) is wrong"
        )
    splice = probs[..., [1, 4, 7, 10]]
    logger.info(
        "provenance OK: splice-prob[1,4,7,10] max=%.3f (expect ≈1 near splice sites)",
        float(splice.max()),
    )


def _run_shard(args: argparse.Namespace) -> None:
    """Build one shard (or the whole file) of the Pangolin per-nt cache.

    Args:
        args (argparse.Namespace): parsed CLI args.
    """
    from pyfaidx import Fasta

    with h5py.File(args.variant_h5, "r") as f:
        n_total = int(f["pos"].shape[0])
        pos = f["pos"][:].astype(np.int64)
        chrom = _decode(f["chrom"][:])
        ref = _decode(f["ref"][:])
        alt = _decode(f["alt"][:])
        strand = _decode(f["strand"][:])
        label = f["label"][:].astype(np.float32)
        gene = _decode(f["gene_symbol"][:])
        kind = _decode(f["variant_kind"][:])

    start, end = _shard_range(n_total, args.shard_index, args.num_shards)
    n = end - start
    logger.info(
        "shard %d/%d rows [%d,%d) n=%d",
        args.shard_index, args.num_shards, start, end, n,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, cl = _load_pangolin_models(args.pangolin_models_dir, device)
    total_length = TARGET_LENGTH + cl  # 15,001
    clp = VOFF + cl // 2  # 7500 → output index VOFF=2500
    fasta = Fasta(args.genome_fasta)

    out_path = (
        args.out_h5
        if args.num_shards == 1
        else f"{args.out_h5}.shard{args.shard_index}_of{args.num_shards}.h5"
    )
    n_invalid = 0
    checked = False
    with h5py.File(out_path, "w") as out, torch.no_grad():
        dref = out.create_dataset(
            "ref_probs", (n, TARGET_LENGTH, NUM_CHANNELS), dtype="float16",
            chunks=(1, TARGET_LENGTH, NUM_CHANNELS), compression="gzip",
            compression_opts=1,
        )
        dalt = out.create_dataset(
            "alt_probs", (n, TARGET_LENGTH, NUM_CHANNELS), dtype="float16",
            chunks=(1, TARGET_LENGTH, NUM_CHANNELS), compression="gzip",
            compression_opts=1,
        )
        keys = np.empty(n, dtype=object)
        row_index = np.arange(start, end, dtype=np.int64)
        bs = args.batch_size
        buf: list[tuple[int, str, str]] = []  # (local_j, ref_str, alt_str)

        def _flush(batch: list[tuple[int, str, str]]) -> None:
            nonlocal checked
            if not batch:
                return
            seqs = [s for _, r, a in batch for s in (r, a)]
            probs = _pangolin_forward_probs(
                models=models, seqs=seqs, device=device,
                target_length=TARGET_LENGTH,
            )  # (2*len, 5001, 12)
            if not checked:  # FR-002 provenance assert (once per shard)
                _assert_channel_provenance(probs)
                checked = True
            for k, (j, _, _) in enumerate(batch):
                dref[j] = probs[2 * k].astype(np.float16)
                dalt[j] = probs[2 * k + 1].astype(np.float16)

        for j, i in enumerate(range(start, end)):
            keys[j] = f"{chrom[i]}:{int(pos[i])}:{ref[i]}:{alt[i]}"
            rec = _build_indel_chunk(
                fasta, chrom[i], int(pos[i]), ref[i], alt[i], strand[i],
                total_length, clp,
            )
            if rec is None:  # ref-mismatch / contig edge — zero-fill (rare)
                n_invalid += 1
                continue
            buf.append((j, _decode_seq(rec["x_ref"]), _decode_seq(rec["x_alt"])))
            if len(buf) >= bs:
                _flush(buf)
                buf = []
            if (j + 1) % 1000 == 0:
                logger.info("  %d/%d (invalid %d)", j + 1, n, n_invalid)
        _flush(buf)

        out.create_dataset("variant_offset", data=np.full(n, VOFF, dtype=np.int64))
        out.create_dataset("labels", data=label[start:end])
        out.create_dataset("gene_symbol", data=np.array(gene[start:end], dtype="S"))
        out.create_dataset("variant_kind", data=np.array(kind[start:end], dtype="S"))
        out.create_dataset("source_row_key", data=keys.astype("S"))
        out.create_dataset("row_index", data=row_index)
    fasta.close()
    logger.info("shard done -> %s (invalid %d/%d)", out_path, n_invalid, n)


def _merge(args: argparse.Namespace) -> None:
    """Merge shard files into the final cache (row-index ordered, verified).

    Args:
        args (argparse.Namespace): parsed CLI args.
    """
    shards = sorted(glob.glob(args.shard_glob))
    if not shards:
        raise FileNotFoundError(f"no shards match {args.shard_glob}")
    with h5py.File(args.variant_h5, "r") as f:
        n_total = int(f["pos"].shape[0])
        chrom = _decode(f["chrom"][:])
        pos = f["pos"][:].astype(np.int64)
        ref = _decode(f["ref"][:])
        alt = _decode(f["alt"][:])
        gene = _decode(f["gene_symbol"][:])
        kind = _decode(f["variant_kind"][:])
    with h5py.File(args.out_h5, "w") as out:
        dref = out.create_dataset(
            "ref_probs", (n_total, TARGET_LENGTH, NUM_CHANNELS), dtype="float16",
            chunks=(1, TARGET_LENGTH, NUM_CHANNELS), compression="gzip",
            compression_opts=1,
        )
        dalt = out.create_dataset(
            "alt_probs", (n_total, TARGET_LENGTH, NUM_CHANNELS), dtype="float16",
            chunks=(1, TARGET_LENGTH, NUM_CHANNELS), compression="gzip",
            compression_opts=1,
        )
        voff = np.full(n_total, VOFF, dtype=np.int64)
        labels = np.zeros(n_total, dtype=np.float32)
        keys = np.empty(n_total, dtype=object)
        for sp in shards:
            with h5py.File(sp, "r") as s:
                ri = s["row_index"][:]
                dref[ri[0] : ri[-1] + 1] = s["ref_probs"][:]
                dalt[ri[0] : ri[-1] + 1] = s["alt_probs"][:]
                labels[ri[0] : ri[-1] + 1] = s["labels"][:]
                for k, gi in enumerate(ri):
                    keys[gi] = s["source_row_key"][k].decode()
        for i in range(n_total):  # FR-002 row-order assert
            exp = f"{chrom[i]}:{int(pos[i])}:{ref[i]}:{alt[i]}"
            assert keys[i] == exp, f"row {i}: {keys[i]} != {exp}"
        out.create_dataset("variant_offset", data=voff)
        out.create_dataset("labels", data=labels)
        out.create_dataset("gene_symbol", data=np.array(gene, dtype="S"))
        out.create_dataset("variant_kind", data=np.array(kind, dtype="S"))
        out.create_dataset("source_row_key", data=np.array(keys, dtype="S"))
    logger.info("merged %d shards -> %s (row-order verified)", len(shards), args.out_h5)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: parsed args.
    """
    p = argparse.ArgumentParser(
        description="Build Pangolin per-nt 12-channel cache for the learned SAV head."
    )
    p.add_argument("--variant-h5", required=True, dest="variant_h5")
    p.add_argument(
        "--pangolin-models-dir", required=True, dest="pangolin_models_dir",
        help="dir with final.{j}.{i}.3.v2 weights",
    )
    p.add_argument(
        "--genome-fasta",
        default="./genome/GRCh38.fa",
        dest="genome_fasta",
    )
    p.add_argument("--out-h5", required=True, dest="out_h5")
    p.add_argument("--shard-index", type=int, default=0, dest="shard_index")
    p.add_argument("--num-shards", type=int, default=1, dest="num_shards")
    p.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--shard-glob", default="", dest="shard_glob")
    return p.parse_args()


def main() -> None:
    """Entry point: build a shard or merge shards."""
    args = parse_args()
    if args.merge:
        _merge(args)
    else:
        _run_shard(args)


if __name__ == "__main__":
    main()
