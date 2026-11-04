"""Render the era, operating-point and interval tables on the v50 MANE universe.

Formatting only; every number comes from the follow-up job's JSON.

    python scripts/render_v50mane_universe_followups.py --dir <out-dir> --manuscript <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import render_recomb2027_results as R

UNIVERSE = "the 107,426-variant GENCODE v50 MANE Select splice-region universe"
TEMPORAL = ["Pangolin", "SpliceAI", "ST", "SSNet", "SplicePath", "SPF"]
SECONDARY = [("2018 archive, 2-star", "pre_spliceai_release_2star"),
             ("Pre-SpliceAI submission date", "pre_spliceai_subdate"),
             ("2021 archive, 2-star", "pre_pangolin_release_2star"),
             ("Pre-Pangolin submission date", "pre_pangolin_subdate")]
CURRENT_PAIRS = [("SPF-SplicePath", "SPF_vs_SplicePath"),
                 ("SPF-Pangolin", "SPF_vs_Pangolin"),
                 ("SPF-SpliceAI", "SPF_vs_SpliceAI")]
STRATA4 = ["overall", "deep-intronic", "SR-acc", "prox-mid"]


def ci(d: dict) -> str:
    """Format one interval the way the manuscript prints it."""
    return f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


def main() -> None:
    """Write the three follow-up blocks into the manuscript."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--manuscript", type=Path, required=True)
    args = ap.parse_args()
    era = json.loads((args.dir / "temporal_metrics_v50mane.json").read_text())
    ops = json.loads((args.dir / "operating_points_v50mane.json").read_text())
    paired = json.loads((args.dir / "paired_intervals_v50mane.json").read_text())
    thr = json.loads((args.dir / "threshold_intervals_v50mane.json").read_text())
    dest = args.manuscript / "results"

    def write(name: str, text: str, source: str) -> None:
        (dest / f"{name}.tex").write_text(
            f"% Source: {args.dir / source}\n" + text.strip() + "\n")

    # ---- era subsets -------------------------------------------------------
    prose = (
        f"The primary era-restricted subsets of {UNIVERSE} retain variants present in "
        f"the 2018 or 2021\nClinVar archive with the same pathogenic/benign tier as the "
        f"current release,\npreceding the reference tools' publication windows. They probe "
        f"label-era sensitivity using the same current predictions;\nthey do not remove "
        f"every source of ascertainment or computational-evidence circularity.\nThe SPF "
        f"head is not retrained on historical releases, so this is not a prospective\n"
        f"training-time holdout. Small positive counts and shifted prevalence limit "
        f"comparisons\nbetween eras and the complete release.\n")
    t1 = R.table(era["pre_spliceai_release"], TEMPORAL,
                 r"""Current predictions on the 2018 ClinVar archive (pre-SpliceAI); AUROC/AUPRC.
No-calls remain zero and each stratum uses its complete era-specific denominator.""",
                 ["tab:precirc-spliceai"], R.STRATA)
    t2 = R.table(era["pre_pangolin_release"], TEMPORAL,
                 r"""Current predictions on the 2021 ClinVar archive (pre-Pangolin); AUROC/AUPRC.
No-calls remain zero and each stratum uses its complete era-specific denominator.""",
                 ["tab:precirc-pangolin"], R.STRATA)
    mid = ("The secondary checks additionally require at least two stars in the historical\n"
           "archive, or restrict the last concordant submission date to before 2019-01-17\n"
           "(SpliceAI) or 2021-07-06 (Pangolin). They retain the same fitted predictions.\n")
    sec = {label: era[key]["overall"] for label, key in SECONDARY}
    t3 = R.table(sec, ["Pangolin", "SpliceAI", "SplicePath", "SPF"],
                 r"""Pooled temporal sensitivity checks, AUROC/AUPRC; subset labels describe the
historical-archive or submission-date criterion.""",
                 ["tab:precirc-secondary"], list(sec))
    write("temporal", prose + t1 + t2 + mid + t3, "temporal_metrics_v50mane.json")

    # ---- fixed-threshold counts and their intervals -------------------------
    lines = [r"\begin{table*}[t]\centering{\normalsize\begin{tabular}{lrrrrrrr}\toprule",
             r"Stratum & $t$ & TP & FP & FN & TN & Precision & Recall \\\midrule"]
    for part in STRATA4:
        for t_ in ("0.5", "0.75", "0.9"):
            c = ops[part][t_]
            lines.append(f"{part} & {t_} & {c['tp']} & {c['fp']} & {c['fn']} & "
                         f"{c['tn']} & {c['ppv']:.3f} & {c['sens']:.3f} " + r"\\")
    lines += [r"\bottomrule\end{tabular}}",
              r"\caption{Fixed-threshold counts from refitted test-fold SPF predictions on "
              r"reported-gene inputs over " + UNIVERSE + r". TP/FP/FN/TN denote true "
              r"positive, false positive, false negative and true negative. All variants "
              r"are retained.}", r"\label{tab:spf-cutoffs}\end{table*}",
              r"\begin{table}[t]\centering{\normalsize\begin{tabular}{rccc}\toprule",
              r"$t$ & Precision CI & Recall CI & Specificity CI \\\midrule"]
    for t_ in ("0.5", "0.75", "0.9"):
        c = thr["overall"][t_]
        lines.append(
            f"{t_} & [{c['precision']['ci_low']:.3f}, {c['precision']['ci_high']:.3f}]"
            f" & [{c['recall']['ci_low']:.3f}, {c['recall']['ci_high']:.3f}]"
            f" & [{c['specificity']['ci_low']:.3f}, {c['specificity']['ci_high']:.3f}] "
            + r"\\")
    lines += [r"\bottomrule\end{tabular}}",
              r"\caption{Gene-component bootstrap 95\% confidence intervals (CI) at the "
              r"fixed thresholds; 2,000 replicates. These estimates describe this "
              r"benchmark prevalence.}", r"\label{tab:spf-cutoffs-ci}\end{table}"]
    write("cutoffs_extended", "\n".join(lines), "operating_points_v50mane.json")

    # ---- paired intervals ---------------------------------------------------
    u = [r"""Uncertainty is assessed by resampling connected components of original gene tokens and normalized
canonical anchor genes, preserving
within-component dependence and pairing the same resample across models. The intervals
below condition on the fitted models.
An interval crossing zero does not establish a ranking difference. Fixed-threshold
bootstrap intervals are recorded with the operating-point artifact.""",
         r"\begin{table*}[t]\centering{\normalsize\begin{tabular}{llcc}\toprule",
         r"Stratum & Comparison & $\Delta$AUROC 95\% CI & $\Delta$AUPRC 95\% CI \\\midrule"]
    for part in STRATA4:
        for label, key in CURRENT_PAIRS:
            r_ = paired[key][part]
            u.append(f"{part} & {label} & {ci(r_['auroc'])} & {ci(r_['auprc'])} " + r"\\")
    u += [r"\bottomrule\end{tabular}}",
          r"\caption{Paired gene-component bootstrap confidence intervals (CI) for current "
          r"SPF minus each comparator on " + UNIVERSE + r". The same resampled components "
          r"are used for both scores (2,000 replicates; seed 42). These intervals condition "
          r"on the saved fitted models and exclude retraining variability.}",
          r"\label{tab:bootstrap-current}\end{table*}", "",
          r"% Former-backbone comparison on the same universe",
          r"\begin{table*}[t]\centering{\normalsize\begin{tabular}{llcc}\toprule",
          r"Stratum & Readout & $\Delta$AUROC 95\% CI & $\Delta$AUPRC 95\% CI \\\midrule"]
    for part in STRATA4:
        for label, key in (("SplicePath", "SplicePath_vs_ep12"), ("SPF", "SPF_vs_SPF_ep12")):
            r_ = paired[key][part]
            # the table reports epoch 12 minus the current version
            flip = {m: dict(ci_low=-r_[m]["ci_high"], ci_high=-r_[m]["ci_low"])
                    for m in ("auroc", "auprc")}
            u.append(f"{part} & {label} & {ci(flip['auroc'])} & {ci(flip['auprc'])} " + r"\\")
    u += [r"\bottomrule\end{tabular}}",
          r"\caption{Epoch 12 minus the current v50 version on " + UNIVERSE + r" or the "
          r"named stratum. The paired bootstrap uses the union of original gene tokens and "
          r"normalized anchor genes (2,000 replicates; seed 42). Backbones differ in "
          r"training recipe; the heads also differ in training-anchor policy and fitting "
          r"eligibility; the current version uses the reported-gene refit, while epoch 12 "
          r"retains its existing heads. These intervals condition on the fitted checkpoints "
          r"and exclude retraining variability.}",
          r"\label{tab:epoch12-canonical-ci}\end{table*}"]
    write("uncertainty", "\n".join(u), "paired_intervals_v50mane.json")

    # ---- the head paragraph's quoted interval -------------------------------
    head = (dest / "head.tex").read_text()
    gain = paired["SPF_vs_SPF_nocons"]["overall"]["auprc"]
    old = "without conservation. The sorted fully connected"
    assert head.count(old) == 1
    head = head.replace(old, "without conservation. The paired gene-component 95\\%\n"
                        f"interval for the AUPRC gain is [{gain['ci_low']:+.3f}, "
                        f"{gain['ci_high']:+.3f}]. The sorted fully connected")
    (dest / "head.tex").write_text(head)
    print(f"rendered temporal, cutoffs_extended, uncertainty and the head interval")


if __name__ == "__main__":
    main()
