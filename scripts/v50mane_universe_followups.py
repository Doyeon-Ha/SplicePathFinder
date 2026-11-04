"""Era subsets, operating points and paired intervals on the v50 MANE universe.

The tables that carry counts and confidence intervals rather than plain metrics
need their own pass once the universe changes. This recomputes, on the
107,426-variant universe and its Table-9 strata: the label-era subsets, SPF's
fixed-threshold operating points, and the gene-component paired bootstrap for
the comparisons the prose quotes.

    python scripts/v50mane_universe_followups.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.table9_strata import table9_masks  # noqa: E402
from scripts.aggregate_t9_precirc import _era_masks  # noqa: E402
from scripts import spf_clinical_cutoffs as cut  # noqa: E402

NAS = Path(".")
EVAL = NAS / "data/eval"
PAPER = EVAL / "recomb2027/ep8_reportedgene_paper_20261102/predictions.npz"
MANE = EVAL / "clinvar_pb_2026_split/coord_splice_fields_v50mane_seed42.npz"
ARMDIR = EVAL / "recomb2027/v50mane_ablation_arms_20261102/deltamax"
PROBE = EVAL / "clinvar_linear_probe"

SHORT = {"canonical": "canonical", "essential": "ess",
         "splice_region_pooled": "SR-all", "splice_region_donor": "SR-don",
         "splice_region_acceptor": "SR-acc", "acc_branch_le40": "acc-br",
         "proximal_mid_intronic": "prox-mid", "deep_intronic_pooled": "deep-intronic",
         "deep_intronic_acceptor": "deep-acc"}
TEMPORAL_MODELS = ["Pangolin", "SpliceAI", "ST", "SSNet", "SplicePath", "SPF"]
# (label, model, reference) for the paired intervals the prose quotes.
PAIRS = [("SPF_vs_SplicePath", "SPF", "SplicePath"),
         ("SPF_vs_Pangolin", "SPF", "Pangolin"),
         ("SPF_vs_SpliceAI", "SPF", "SpliceAI"),
         ("SPF_vs_SPF_nocons", "SPF", "SPF_nocons"),
         ("SPF_vs_FC_cons", "SPF", "FC_cons"),
         ("SPF_vs_SPF_ep12", "SPF", "SPF_ep12"),
         ("SplicePath_vs_ep12", "SplicePath", "SplicePath_ep12")]
THRESHOLDS = [0.5, 0.75, 0.9]
RESAMPLES = 2000
SEED = 42


def dec(a: np.ndarray) -> np.ndarray:
    """Return a byte-or-object string array as plain text, as an object array."""
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a],
                    dtype=object)


def arm_on_release(tag: str, where: dict[str, int], n: int) -> np.ndarray:
    """Lift one arm's v50-table delta max onto the release row space."""
    with np.load(ARMDIR / f"{tag}_deltamax.npz", allow_pickle=True) as z:
        dm = z["delta_max"].astype(np.float64)
        chrom = [c[3:] if c.startswith("chr") else c for c in dec(z["chrom"])]
        key = [f"{c}:{p}:{r}:{a}" for c, p, r, a in
               zip(chrom, z["pos"], dec(z["ref"]), dec(z["alt"]))]
    out = np.full(n, np.nan)
    rows = np.array([where.get(k, -1) for k in key])
    take = rows >= 0
    out[rows[take]] = dm[take]
    return out


def fast_auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC from ranks, called once per resample."""
    r = rankdata(s)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def component_bootstrap(y, a, b, groups):
    """Resample whole gene components and return the paired AUROC/AUPRC deltas."""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(SEED)
    d_roc, d_prc = np.empty(RESAMPLES), np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[j] for j in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            d_roc[i] = d_prc[i] = np.nan
            continue
        d_roc[i] = fast_auroc(yy, a[idx]) - fast_auroc(yy, b[idx])
        d_prc[i] = (average_precision_score(yy, a[idx])
                    - average_precision_score(yy, b[idx]))
    out = {}
    for name, d, point in (("auroc", d_roc, roc_auc_score(y, a) - roc_auc_score(y, b)),
                           ("auprc", d_prc, average_precision_score(y, a)
                            - average_precision_score(y, b))):
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        out[name] = dict(delta=float(point), ci_low=float(lo), ci_high=float(hi),
                         significant=bool(lo > 0 or hi < 0))
    return out


def main() -> None:
    """Write the era, operating-point and paired-interval JSON files."""
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
    universe = cf["splice"].astype(bool)
    is_snv = np.array([len(a) == 1 and len(b) == 1 for a, b in
                       (k.split(":")[2:4] for k in release_key)])
    groups = np.asarray(paper["component"])

    t9 = table9_masks(cf["intron_nt"], cf["exon_nt"], cf["akind"])
    masks = {SHORT[k]: v & universe for k, v in t9.items() if k in SHORT}
    masks.update(overall=universe, SNV=universe & is_snv, indel=universe & ~is_snv)

    scores = {k: paper[k].astype(np.float64) for k in
              ["Pangolin", "SpliceAI", "ST", "SSNet", "SPF", "SPF_nocons",
               "SplicePath_ep12", "SPF_ep12"]}
    scores["SplicePath"] = arm_on_release("FULL_sp_v50mane_hnet_ep10", where, n)
    with np.load(PROBE / ("cv5_oof_fulltrain_sorted_cons_sp_v50mane_hnet_ep10"
                          "_reportedgene_mc8750_gsplitcc_v50mane_universe.npz"),
                 allow_pickle=True) as z:
        s = np.full(n, np.nan)
        s[universe] = z["score"][universe].astype(np.float64)
    scores["FC_cons"] = s

    # label-era subsets
    era = {}
    for ename, em in _era_masks(y).items():
        era[ename] = {}
        for pname, m in masks.items():
            sel = m & em
            if int(sel.sum()) < 20 or y[sel].min() == y[sel].max():
                continue
            era[ename][pname] = {
                k: dict(n=int(sel.sum()), n_pos=int(y[sel].sum()),
                        auroc=float(roc_auc_score(y[sel], scores[k][sel])),
                        auprc=float(average_precision_score(y[sel], scores[k][sel])))
                for k in TEMPORAL_MODELS}
        print(f"era {ename:28s} strata={len(era[ename])}", flush=True)
    (args.out_dir / "temporal_metrics_v50mane.json").write_text(
        json.dumps(era, indent=2) + "\n")

    # SPF fixed-threshold operating points
    ops = {pname: {f"{t:g}": cut.at(y[m], scores["SPF"][m], t) for t in THRESHOLDS}
           for pname, m in masks.items()}
    ops["native_reference"] = {
        k: {f"{t:g}": cut.at(y[universe], scores[k][universe], t) for t in (0.2, 0.5, 0.8)}
        for k in ("Pangolin", "SpliceAI")}
    (args.out_dir / "operating_points_v50mane.json").write_text(
        json.dumps(ops, indent=2) + "\n")
    o = ops["overall"]["0.5"]
    print(f"SPF@0.5 overall sens={o['sens']:.3f} spec={o['spec']:.3f} "
          f"tp={o['tp']} fp={o['fp']} fn={o['fn']} tn={o['tn']}", flush=True)

    # gene-component paired intervals
    paired = {}
    for label, a, b in PAIRS:
        paired[label] = {}
        for pname in ("overall", "deep-intronic", "SR-acc", "prox-mid"):
            m = masks[pname]
            ok = m & np.isfinite(scores[a]) & np.isfinite(scores[b])
            if int(ok.sum()) != int(m.sum()):
                paired[label][pname] = dict(missing=int(m.sum() - ok.sum()))
                continue
            paired[label][pname] = component_bootstrap(
                y[ok], scores[a][ok], scores[b][ok], groups[ok])
            r = paired[label][pname]
            print(f"  {label:22s} {pname:14s} "
                  f"ROC {r['auroc']['delta']:+.4f} "
                  f"[{r['auroc']['ci_low']:+.4f},{r['auroc']['ci_high']:+.4f}]  "
                  f"PRC {r['auprc']['delta']:+.4f} "
                  f"[{r['auprc']['ci_low']:+.4f},{r['auprc']['ci_high']:+.4f}]", flush=True)
    (args.out_dir / "paired_intervals_v50mane.json").write_text(
        json.dumps(paired, indent=2) + "\n")

    # gene-component intervals for SPF's fixed-threshold operating characteristics
    thr = {}
    for pname in ("overall", "SR-acc", "prox-mid", "deep-intronic"):
        m = masks[pname]
        yy, ss, gg = y[m], scores["SPF"][m], groups[m]
        uniq, inv = np.unique(gg, return_inverse=True)
        members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
        thr[pname] = {}
        for t_ in THRESHOLDS:
            rng = np.random.default_rng(SEED)
            prec, rec, spec = (np.empty(RESAMPLES) for _ in range(3))
            for i in range(RESAMPLES):
                idx = np.concatenate([members[j] for j in
                                      rng.integers(0, len(members), len(members))])
                tp, fp, fn, tn = cut.counts(yy[idx], ss[idx] >= t_)
                prec[i] = tp / (tp + fp) if tp + fp else np.nan
                rec[i] = tp / (tp + fn) if tp + fn else np.nan
                spec[i] = tn / (tn + fp) if tn + fp else np.nan
            thr[pname][f"{t_:g}"] = {
                k: dict(ci_low=float(np.nanpercentile(v, 2.5)),
                        ci_high=float(np.nanpercentile(v, 97.5)))
                for k, v in (("precision", prec), ("recall", rec), ("specificity", spec))}
            c = thr[pname][f"{t_:g}"]
            print(f"  thr {pname:14s} t={t_:g} "
                  f"prec [{c['precision']['ci_low']:.3f},{c['precision']['ci_high']:.3f}] "
                  f"rec [{c['recall']['ci_low']:.3f},{c['recall']['ci_high']:.3f}] "
                  f"spec [{c['specificity']['ci_low']:.3f},{c['specificity']['ci_high']:.3f}]",
                  flush=True)
    (args.out_dir / "threshold_intervals_v50mane.json").write_text(
        json.dumps(thr, indent=2) + "\n")
    print(f"\nwrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
