"""Readers for separately generated native-tool SpliceVarDB scores."""
import csv
import glob
import re
from pathlib import Path
import numpy as np

ANCHORED_H5 = (
    "./data/eval/"
    "splicevardb_full_sa_normal_gencode_anchored.h5"
)
MEDEIAC_DIR = "./output_ep14/svdb_full_ep14"
WORKLIST = "./worklists/svdb_full_ep14_worklist.tsv"
ST_NPZ = (
    "./results_registry/splicetransformer/"
    "st_svdb_cv5_scores.npz"
)
SSNET_NPZ = "./data/ssnet/svdb_full/ensemble_scores.npz"
# /data inputs are mirrored to /NAS by the launcher so any CPU node can read them.
PANGOLIN_CSV = (
    "./data/eval/svdb_native_table1/"
    "pangolin_svdb_out.csv"
)
SPLICEAI_DIR = (
    "./data/eval/svdb_native_table1/spliceai_official_full"
)


_SCORE_RE = re.compile(r"-?\d+:(-?\d+\.\d+)")

def _dec(a: np.ndarray) -> np.ndarray:
    """Decode a byte/object string array to a unicode ``np.ndarray``."""
    return np.array(
        [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in a]
    )

def _keys(chrom, pos, ref, alt) -> np.ndarray:
    """Build ``chrom:pos:ref:alt`` join keys."""
    return np.array(
        [f"{chrom[i]}:{int(pos[i])}:{ref[i]}:{alt[i]}" for i in range(len(pos))]
    )

def _score_pangolin(n: int, key2i: dict) -> np.ndarray:
    """Per-variant Pangolin native = max |Delta| over genes; NaN = no-call."""
    s = np.full(n, np.nan)
    with open(PANGOLIN_CSV) as f:
        next(f)
        for line in f:
            fields = line.rstrip("\n").split(",")
            if len(fields) < 5:
                continue
            k = f"{fields[0]}:{int(fields[1])}:{fields[2]}:{fields[3]}"
            i = key2i.get(k)
            if i is None:
                continue
            tail = ",".join(fields[5:]).strip()
            vals = [abs(float(m)) for m in _SCORE_RE.findall(tail)]
            if vals:
                s[i] = max(vals)
    return s

def _score_spliceai(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Official SpliceAI native delta_max indexed by anchored order; scored mask.

    delins/MNV no-calls are returned as NaN in ``scored``-false slots; the caller
    pools them at 0 to match Table 1.
    """
    delta = np.full(n, np.nan)
    scored = np.zeros(n, bool)
    seen = np.zeros(n, bool)
    files = sorted(glob.glob(str(Path(SPLICEAI_DIR) / "scores*.npz")))
    assert files, f"no scores*.npz in {SPLICEAI_DIR}"
    for fp in files:
        z = np.load(fp)
        idx = z["idx"]
        delta[idx] = z["delta_max"]
        scored[idx] = z["scored"]
        seen[idx] = True
    assert seen.all(), f"{(~seen).sum()} spliceai variants unseen"
    return delta, scored

def _score_st(n: int) -> np.ndarray:
    """SpliceTransformer native dsplice indexed to anchored order (row_idx)."""
    z = np.load(ST_NPZ)
    s = np.full(n, np.nan)
    s[z["row_idx"]] = z["dsplice"]
    return s

def _score_ssnet(n: int, key2i: dict) -> np.ndarray:
    """SpliceSelectNet native ensemble score joined by variant key."""
    z = np.load(SSNET_NPZ, allow_pickle=True)
    chrom = _dec(z["chrom"])
    chrom = np.array([c if c.startswith("chr") else f"chr{c}" for c in chrom])
    k = _keys(chrom, z["pos"], _dec(z["ref"]), _dec(z["alt"]))
    s = np.full(n, np.nan)
    for kk, sc in zip(k, z["score"]):
        i = key2i.get(kk)
        if i is not None:
            s[i] = sc
    return s
