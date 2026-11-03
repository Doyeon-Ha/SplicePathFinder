"""Every ClinVar table column scored on the 107,426-variant v50 MANE universe.

The manuscript's result grids were measured on the 104,811-variant reported-gene
canonical cohort while Tables 2 and 4 report the GENCODE v50 MANE Select
splice-region universe. This puts all of them on the MANE universe: it collects
each column from the source that covers those rows, applies the Table-9 distance
strata computed from the same v50 MANE coordinate fields, and writes one metrics
JSON per stratum.

A column whose source does not cover the universe is listed as pending with the
number of rows it is missing; nothing is filled in.

    python scripts/assemble_v50mane_universe_tables.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from table9_strata import table9_masks  # noqa: E402

NAS = Path(".")
EVAL = NAS / "data/eval"
PAPER = EVAL / "recomb2027/ep8_reportedgene_paper_20261102/predictions.npz"
MANE = EVAL / "clinvar_pb_2026_split/coord_splice_fields_v50mane_seed42.npz"
ARMDIR = EVAL / "recomb2027/v50mane_ablation_arms_20261102/deltamax"
PROBE = EVAL / "clinvar_linear_probe"
SPF_OOF = EVAL / "spf_multichunk"

# Columns the paper predictions file already carries over the whole release.
FROM_PAPER = [
    "Pangolin", "SpliceAI", "ST", "SSNet",
    "SPF", "SPF_nocons", "SPF_Pangolin", "SPF_SpliceAI",
    "phylop|anchor", "phastcons|anchor", "mean_norm|anchor",
    "phylop|interval_max", "phastcons|interval_max", "mean_norm|interval_max",
    "phylop|interval_mean", "phastcons|interval_mean", "mean_norm|interval_mean",
    "SplicePath_ep12", "SPF_ep12",
]
# Label-free backbone readouts, read from each arm's full variant-table cache so
# that the whole block shares one anchor policy and one chunk grid.
FROM_ARMS = {
    "SplicePath": "FULL_sp_v50mane_hnet_ep10",
    "minus_region": "FULL_sp_v50mane_hnet_noregion_ep2",
    "minus_region_ep10": "FULL_sp_v50mane_hnet_noregion_ep10",
    "minus_bidir": "FULL_sp_v50mane_hnet_nobidir_ep10",
    "minus_hybrid": "FULL_sp_v50mane_hnet_nohybrid_ep10",
}
# Heads trained on the v50 ep10 backbone — the pair Table 2 reports.
FROM_OOF = {
    "SPF_v50": ("cv5_sp_v50mane_hnet_ep10_reportedgene_cc/"
                "drcons_cv5_oof_result_v50mane_reportedgene_preds.npz"),
    "SPF_nocons_v50": ("cv5_sp_v50mane_hnet_ep10_reportedgene_cc_nocons/"
                       "drcons_cv5_oof_result_v50mane_reportedgene_nocons_preds.npz"),
}
# Fully connected probes refitted on this universe's folds.
FROM_PROBE = {
    "FC_P": "cv5_oof_fulltrain_sorted_pangolin_gsplitcc_v50mane_universe",
    "FC_S": "cv5_oof_fulltrain_sorted_spliceai_gsplitcc_v50mane_universe",
    "FC_ST": "cv5_oof_fulltrain_sorted_splicetransformer_gsplitcc_v50mane_universe",
    "FC_SN": "cv5_oof_fulltrain_sorted_ssnet_gsplitcc_v50mane_universe",
    "FC_SP": ("cv5_oof_fulltrain_sorted_sp_v50mane_hnet_ep10_reportedgene"
              "_mc8750_gsplitcc_v50mane_universe"),
    "FCc_SP": ("cv5_oof_fulltrain_sorted_cons_sp_v50mane_hnet_ep10_reportedgene"
               "_mc8750_gsplitcc_v50mane_universe"),
}
# Table-9 stratum -> the short name the manuscript tables print.
SHORT = {
    "canonical": "canonical", "essential": "ess",
    "splice_region_pooled": "SR-all", "splice_region_donor": "SR-don",
    "splice_region_acceptor": "SR-acc", "acc_branch_le40": "acc-br",
    "proximal_mid_intronic": "prox-mid", "deep_intronic_pooled": "deep-intronic",
    "deep_intronic_acceptor": "deep-acc",
}


def dec(a: np.ndarray) -> np.ndarray:
    """Return a byte-or-object string array as plain text, as an object array."""
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a],
                    dtype=object)


def arm_on_release(tag: str, where: dict[str, int], n: int) -> np.ndarray:
    """Lift one arm's v50-table delta max onto the release row space."""
    path = ARMDIR / f"{tag}_deltamax.npz"
    if not path.exists():
        return np.full(n, np.nan)
    with np.load(path, allow_pickle=True) as z:
        dm = z["delta_max"].astype(np.float64)
        # the release key drops the "chr" prefix the v50 table carries
        chrom = [c[3:] if c.startswith("chr") else c for c in dec(z["chrom"])]
        key = [f"{c}:{p}:{r}:{a}" for c, p, r, a in
               zip(chrom, z["pos"], dec(z["ref"]), dec(z["alt"]))]
    out = np.full(n, np.nan)
    rows = np.array([where.get(k, -1) for k in key])
    take = rows >= 0
    out[rows[take]] = dm[take]
    return out


def main() -> None:
    """Write the metrics of every column on the MANE universe's strata."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paper = np.load(PAPER, allow_pickle=True)
    release_key = np.asarray(paper["key"], dtype=object)
    where = {k: i for i, k in enumerate(release_key)}
    n = len(release_key)
    cf = np.load(MANE, allow_pickle=True)
    y = cf["labels"].astype(np.int64)
    np.testing.assert_array_equal(y, paper["y"].astype(np.int64))
    universe = cf["splice"].astype(bool)
    is_snv = np.array([len(a) == 1 and len(b) == 1 for a, b in
                       (k.split(":")[2:4] for k in release_key)])

    t9 = table9_masks(cf["intron_nt"], cf["exon_nt"], cf["akind"])
    masks = {SHORT[k]: v & universe for k, v in t9.items() if k in SHORT}
    masks.update(overall=universe, SNV=universe & is_snv, indel=universe & ~is_snv)

    scores: dict[str, np.ndarray] = {k: paper[k].astype(np.float64) for k in FROM_PAPER}
    for name, tag in FROM_ARMS.items():
        scores[name] = arm_on_release(tag, where, n)
    for name, stem in FROM_PROBE.items():
        path = PROBE / f"{stem}.npz"
        if not path.exists():
            scores[name] = np.full(n, np.nan)
            continue
        with np.load(path, allow_pickle=True) as z:
            np.testing.assert_array_equal(z["splice"].astype(bool), universe)
            np.testing.assert_array_equal(z["label"].astype(np.int64), y)
            s = np.full(n, np.nan)
            s[universe] = z["score"][universe].astype(np.float64)
        scores[name] = s
    for name, stem in FROM_OOF.items():
        path = SPF_OOF / stem
        if not path.exists():
            scores[name] = np.full(n, np.nan)
            continue
        with np.load(path, allow_pickle=True) as z:
            np.testing.assert_array_equal(z["y"].astype(np.int64), y)
            scores[name] = z["p"].astype(np.float64)
    scores["FC_cons"] = scores["FCc_SP"]

    rows: dict[str, dict] = {name: {} for name in masks}
    pending: dict[str, dict] = {}
    for label, s in scores.items():
        miss = int((universe & ~np.isfinite(s)).sum())
        if miss:
            pending[label] = dict(missing_predictions=miss)
            print(f"pending {label:24s} missing {miss:,}", flush=True)
            continue
        for name, m in masks.items():
            yy, ss = y[m], s[m]
            rows[name][label] = dict(
                n=int(m.sum()), n_pos=int(yy.sum()),
                auroc=float(roc_auc_score(yy, ss)),
                auprc=float(average_precision_score(yy, ss)))
        o = rows["overall"][label]
        print(f"{label:24s} overall ROC {o['auroc']:.4f} PRC {o['auprc']:.4f}", flush=True)

    meta = dict(
        universe="v50mane_splice_region",
        universe_n=int(universe.sum()),
        universe_pos=int(y[universe].sum()),
        coord=str(MANE),
        sources=dict(paper=str(PAPER), arms=str(ARMDIR), probe=str(PROBE)),
        strata={name: dict(n=int(m.sum()), n_pos=int(y[m].sum()))
                for name, m in masks.items()},
    )
    (args.out_dir / "clinvar_metrics_v50mane_universe.json").write_text(
        json.dumps(dict(meta=meta, rows=rows, pending=pending), indent=2) + "\n")
    print(f"\nwrote {args.out_dir / 'clinvar_metrics_v50mane_universe.json'}", flush=True)


if __name__ == "__main__":
    main()
