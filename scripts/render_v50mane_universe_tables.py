"""Render the ClinVar result grids from the v50 MANE universe metrics.

Formatting only. It reuses the shared table writer so column order, marking and
spacing stay identical to the rest of the manuscript, and writes the grids whose
every column is measured on the 107,426-variant universe.

    python scripts/render_v50mane_universe_tables.py \
        --metrics <clinvar_metrics_v50mane_universe.json> --manuscript <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import render_recomb2027_results as R

UNIVERSE = (r"the 107,426-variant GENCODE v50 MANE Select splice-region universe, "
            r"the same variants and strata as Table~\ref{tab:clinvar-cv5-region-fine}")


def main() -> None:
    """Write each grid into the manuscript's results directory."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--manuscript", type=Path, required=True)
    ap.add_argument("--stage-dir", type=Path, help="also write the supplementary fragment here")
    args = ap.parse_args()

    meta = json.loads(args.metrics.read_text())
    cv = meta["rows"]
    # the head grid and its prose read the two heads trained on the v50 ep10
    # backbone, so the ablation pairs with the model Table 2 reports
    head_cv = {p: {**r, "SPF": r.get("SPF_v50", r.get("SPF")),
                   "SPF_nocons": r.get("SPF_nocons_v50", r.get("SPF_nocons"))}
               for p, r in cv.items()}
    dest = args.manuscript / "results"
    dest.mkdir(exist_ok=True)

    def write(name: str, text: str) -> None:
        (dest / f"{name}.tex").write_text(
            f"% Source: {args.metrics}\n" + text.strip() + "\n")

    # the conservation grid renames its readout columns
    cm = {p: {"phyloP": v["phylop|interval_max"],
              "phastCons": v["phastcons|interval_max"],
              "Average": v["mean_norm|interval_max"],
              "SPF": v["SPF"]} for p, v in cv.items()}

    o = cv["overall"]
    deep = cv["deep-intronic"]
    prose = (
        f"A raw conservation score ranks variants without fitting a classifier.\n"
        f"The interval-maximum average of normalized phyloP and phastCons reaches\n"
        f"{o['mean_norm|interval_max']['auroc']:.3f}/"
        f"{o['mean_norm|interval_max']['auprc']:.3f} AUROC/AUPRC on "
        f"{UNIVERSE.split(',')[0]}.\n"
        f"Its deep-intronic estimate is {deep['mean_norm|interval_max']['auroc']:.3f}/"
        f"{deep['mean_norm|interval_max']['auprc']:.3f}, compared with\n"
        f"{deep['SPF']['auroc']:.3f}/{deep['SPF']['auprc']:.3f} for refitted SPF. "
        f"This comparison does not isolate conservation's\n"
        f"contribution within SPF. Undefined whole intervals receive zero; all variants\n"
        f"are retained. For phyloP indels, interval maximum, record position and interval\n"
        f"mean give AUROC {cv['indel']['phylop|interval_max']['auroc']:.3f},\n"
        f"{cv['indel']['phylop|anchor']['auroc']:.3f} and "
        f"{cv['indel']['phylop|interval_mean']['auroc']:.3f}, respectively.\n")
    write("conservation", prose + R.table(
        cm, ["phyloP", "phastCons", "Average", "SPF"],
        r"""Training-free raw conservation
and current SPF on identical variants, AUROC/AUPRC. Conservation uses the maximum
over the affected reference interval; Average is the mean of phastCons and phyloP
clipped to $[-10,10]$ and divided by 10. Undefined intervals score zero. SPF's trained
six-feature input still uses genomic-variant-position features and a $\pm2{,}500$-nt window.
All columns are measured on """ + UNIVERSE + ".",
        ["tab:cons-raw"], R.STRATA))

    write("head_extended", R.table(
        head_cv, ["SPF", "SPF_SRtrain", "Cons6_MLP", "Cons6_logreg"],
        r"""Full head/control grid, AUROC/AUPRC.
The heads are refitted on reported-gene tracks with identical fitting
eligibility and frozen gene-component folds. A1 is the sorted fully connected
(FC) probe; A2 appends six conservation scalars (Cons6) at the FC input; A3 removes
conservation from SPF. Both Mamba-2 heads are trained on the v50 ep10 backbone,
the model Table~\ref{tab:clinvar-cv5-region-fine} reports, and were trained on
different GPU architectures, whose measured effect on this readout is about
0.004 AUROC, so A3 and All are not separated by conservation alone at that scale.
FC validates on the splice-region universe; SPF validates on
all eligible rows of the validation fold. SR train is the splice-region-only
control; MLP is a multilayer perceptron. Every reported column covers """
        + UNIVERSE + ".",
        ["tab:deltareader-ablation"], R.STRATA))

    write("cons_probe", R.table(
        cv, ["FC_P", "FCc_P", "FC_S", "FCc_S", "FC_ST", "FCc_ST",
             "FC_SN", "FCc_SN", "FC_SP", "FCc_SP"],
        r"""Sorted fully connected (FC) probes without and with (FC$^c$) six conservation
scalars, AUROC/AUPRC. SplicePath FC$^c$ concatenates conservation at the input;
its two SplicePath controls share reported-gene tracks and fitting eligibility.
External FC probes use stored-anchor profiles and all release training rows;
their conservation-augmented variants are pending. Every probe is refitted and
pooled out of fold on """ + UNIVERSE + r""". These inputs and fitting populations
differ from SPF's full eligible-fold validation.""",
        ["tab:clinvar-cv5-region-fine-cons"], R.STRATA))

    write("baseswap", R.table(
        cv, ["SPF_Pangolin", "SPF_SpliceAI", "SPF"],
        r"""The SPF head (Mamba-2 with the six
conservation scalars) trained on each backbone's frozen tracks on the same connected-component
folds, AUROC/AUPRC. Pangolin and SpliceAI keep their single-window caches, whereas SplicePath
(SPF) uses merged 8,750-position tracks, so the feature source and the input construction both
differ. All three columns are pooled out of fold on """ + UNIVERSE + r""".
These differences prevent isolating backbone choice.""",
        ["tab:clinvar-cv5-baseswap"], R.STRATA))

    write("dnalm", R.table(
        cv, ["Evo2", "LDARNet", "CaduceusPh", "CrossDNA", "DNABERT2",
             "SplicePath", "SPF"],
        r"""Random forests over frozen DNA language-model embeddings, compared with the
label-free SplicePath readout and SPF on """ + UNIVERSE + r""".
The DNA-model columns are pending on this universe; no estimate measured on a
different universe is substituted.""",
        ["tab:clinvar-cv5-region-fine-dnalm"], R.STRATA))

    write("unsort_linear_probe", R.table(
        cv, ["FCu_P", "FCu_S", "FCu_ST", "FCu_SN", "FCu_SP"],
        r"""Position-fixed (unsorted) fully connected probes on """ + UNIVERSE + r""".
Each probe reads per-position profiles in genomic order rather than sorted by
score, on the same connected-component folds; three folds train, one validates.""",
        ["tab:clinvar-cv5-region-fine-unsort"], R.STRATA))

    oo, dd, pm = (head_cv["overall"], head_cv["deep-intronic"], head_cv["prox-mid"])
    write("head", (
        f"SPF combines frozen SplicePath tracks, a Mamba-2 head and six conservation\n"
        f"features (Cons6). The reported-gene refit reaches {oo['SPF']['auroc']:.3f}/"
        f"{oo['SPF']['auprc']:.3f} pooled AUROC/AUPRC,\n"
        f"compared with {oo['SPF_nocons']['auroc']:.3f}/{oo['SPF_nocons']['auprc']:.3f} "
        f"without conservation. The sorted fully connected\n"
        f"probe gives {oo['FC_SP']['auroc']:.3f}/{oo['FC_SP']['auprc']:.3f} without "
        f"conservation and {oo['FC_cons']['auroc']:.3f}/{oo['FC_cons']['auprc']:.3f} "
        f"with Cons6 at its\ninput. The raw conservation average gives "
        f"{oo['mean_norm|interval_max']['auroc']:.3f}/"
        f"{oo['mean_norm|interval_max']['auprc']:.3f}.\n"
        f"These controls support a pooled benefit from the combined system, but the FC\n"
        f"controls validate on this universe whereas SPF validates on all eligible\n"
        f"rows of the validation fold. Architecture and validation policy are therefore\n"
        f"not isolated. Region-specific ordering also differs: in deep introns, FC$+$cons.\n"
        f"gives {dd['FC_cons']['auroc']:.3f}/{dd['FC_cons']['auprc']:.3f} and SPF "
        f"{dd['SPF']['auroc']:.3f}/{dd['SPF']['auprc']:.3f}; in proximal-mid introns, "
        f"SPF without\nconservation gives {pm['SPF_nocons']['auroc']:.3f}/"
        f"{pm['SPF_nocons']['auprc']:.3f} versus SPF's {pm['SPF']['auroc']:.3f}/"
        f"{pm['SPF']['auprc']:.3f}.\n"
        f"Both heads read the v50 ep10 backbone; they were trained on different GPU\n"
        f"architectures, which moves this readout by about 0.004 AUROC on its own.\n"
        f"Conservation-only fitted classifiers remain pending.\n"))

    # the supplementary backbone grid is spliced into supplementary.tex by hand
    frag = R.table(
        cv, ["SplicePath", "minus_region", "minus_bidir", "minus_hybrid"],
        r"""Backbone comparison on """ + UNIVERSE + r""",
using all-genes training, real genomic flanks and merged 8,750-position delta max.
Every arm is read from its own full variant-table cache, so the four columns share
one anchor policy and one chunk grid. $-$region is selected at epoch 2 by
validation AUROC; $-$bidir and $-$hybrid are still improving at epoch 10, the last
trained for them.""",
        ["tab:backbone-ablation"], R.STRATA)
    out = args.stage_dir or args.metrics.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "backbone_ablation_grid.tex").write_text(frag)
    print(f"wrote results grids into {dest} and the supplementary fragment into {out}")


if __name__ == "__main__":
    main()
