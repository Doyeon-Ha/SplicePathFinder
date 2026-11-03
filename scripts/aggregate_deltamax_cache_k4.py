"""Score and optionally register the K=4 target-only 8,750-position cache readout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

NAS = Path(".")
COORD = NAS / "data/eval/clinvar_pb_2026_split/coord_splice_fields_seed42.npz"
REGISTRY = NAS / "results_registry/registry_cli.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--dataset", choices=("splicevardb_full", "clinvar_20260628"),
                        required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--clinvar-source", choices=("release", "splice_region"), default="release",
        help="ClinVar only: the cache rows are the full release (scored on its "
             "splice-region mask) or the splice-region table itself",
    )
    parser.add_argument("--model", default="med_ei_ac_ep14")
    parser.add_argument("--git-commit", default="uncommitted")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--clinvar-regions", action="store_true",
                        help="also report the canonical Table-9 distance strata")
    parser.add_argument(
        "--allow-short-indels", action="store_true",
        help="Admit 1-3 chunks only for alleles that exceed the genomic "
             "input edge; verify the exact expected count and record it",
    )
    args = parser.parse_args()
    if args.clinvar_regions and args.dataset != "clinvar_20260628":
        parser.error("--clinvar-regions requires --dataset clinvar_20260628")

    with np.load(args.scores) as z:
        scores = z["delta_max"].astype(np.float64)
        masked = bool(z["masked"])
        application = str(z["mask_application"])
        if (int(z["window_left"]), int(z["window_right"])) != (4375, 4375):
            raise ValueError("delta max must use the complete [-4375,+4375) cache window")
    if masked and application != "premerge":
        raise ValueError("masked scores must use premerge per-chunk masking")
    if not np.isfinite(scores).all():
        raise ValueError("non-finite scores")

    with h5py.File(args.cache) as f:
        if f["ref_probs"].shape[:2] != (len(scores), 8750):
            raise ValueError("cache shape does not match scores")
        if int(f.attrs["stride"]) != 1250:
            raise ValueError("cache is not aligned to the K=4 stride")
        if not np.array_equal(f["row_index"][:], np.arange(len(scores))):
            raise ValueError("cache rows do not match the source row order")
        chunks = f["n_chunks_used"][:] if "n_chunks_used" in f else None
        if (chunks is not None and not np.all(chunks == 4)
                and not args.allow_short_indels):
            raise ValueError("at least one variant was not scored with K=4")
    with h5py.File(args.source_h5) as f:
        labels = f["label"][:].astype(np.int8)
        kind = f["variant_kind"][:]
    if len(labels) != len(scores):
        raise ValueError("source labels and scores have different lengths")
    snv = kind == b"snv"
    indel = kind == b"indel"
    if not np.all(snv | indel):
        raise ValueError("unexpected variant kind")
    short_note = ""
    if chunks is not None and not np.all(chunks == 4):
        short = np.flatnonzero(chunks != 4)
        if np.any((chunks < 1) | (chunks > 4)) or not indel[short].all():
            raise ValueError("a missing call or non-indel has fewer than four chunks")
        with h5py.File(args.source_h5) as f:
            for i in short:
                offset = int(f["variant_offset"][i])
                clps = np.arange(offset % 1250, 15000, 1250)
                clps = clps[(clps >= 5000) & (clps < 10000)]
                strand_key = "source_strand" if "source_strand" in f else "strand"
                strand = f[strand_key][i]
                l_ref = len(f["ref"][i])
                capacity = clps + 1 if strand == b"-" else 15000 - clps
                expected = int(np.sum(l_ref <= capacity))
                if expected != chunks[i]:
                    raise ValueError(f"row {i}: {chunks[i]} chunks; expected {expected}")
        short_note = (f" {len(short)} long indels have 1-3 chunks due to "
                      f"the input edge; rows={short.tolist()}; "
                      f"chunks={chunks[short].tolist()}.")
        print(short_note, flush=True)

    if args.dataset == "splicevardb_full":
        universe = np.ones(len(scores), dtype=bool)
        dataset_n = len(scores)
        names = ("overall", "snv", "indel")
    else:
        with np.load(COORD) as coord:
            splice = coord["splice"].astype(bool)
        dataset_n = len(splice)
        if args.clinvar_source == "release":
            universe = splice
            if len(universe) != len(scores):
                raise ValueError(
                    "ClinVar splice-region mask and scores differ in length"
                )
        else:
            # Every row must be one of the release's splice-region rows, each once.
            with h5py.File(args.source_h5) as f:
                release_rows = f["_source_row_index"][:]
            if (len(release_rows) != int(splice.sum())
                    or len(np.unique(release_rows)) != len(release_rows)
                    or not splice[release_rows].all()):
                raise ValueError("source is not the release's splice-region subset")
            universe = np.ones(len(scores), dtype=bool)
        names = ("t9_overall", "t9_SNV", "t9_indel")
    masks = (universe, universe & snv, universe & indel)
    partitions = list(zip(names, masks))
    if args.clinvar_regions:
        from scripts.table9_strata import table9_masks

        coord_rows = slice(None) if args.clinvar_source == "release" else release_rows
        with np.load(COORD) as coord:
            if not np.array_equal(coord["labels"][coord_rows], labels):
                raise ValueError("coordinate fields and source labels are not row-aligned")
            strata = table9_masks(coord["intron_nt"][coord_rows],
                                  coord["exon_nt"][coord_rows],
                                  coord["akind"][coord_rows])
        for name, mask in strata.items():
            if np.any(mask & ~universe):
                raise ValueError(f"stratum {name} leaves the splice-region universe")
            partitions.append((f"t9_{name}", mask))
    rows = []
    for name, mask in partitions:
        y, s = labels[mask], scores[mask]
        if len(np.unique(y)) != 2:
            raise ValueError(f"{name} has only one class")
        row = {"partition": name, "n": int(mask.sum()),
               "positive": int(y.sum()),
               "auroc": float(roc_auc_score(y, s)),
               "auprc": float(average_precision_score(y, s))}
        rows.append(row)
        print(f"{name}: n={row['n']} AUROC={row['auroc']:.6f} "
              f"AUPRC={row['auprc']:.6f}", flush=True)

    summary = args.scores.with_suffix(".metrics.json")
    summary.write_text(json.dumps(rows, indent=2) + "\n")
    if args.register:
        scoring = ("v4_aligned_k4_cache_masked" if masked
                   else "v4_aligned_k4_cache")
        data_paths = [str(args.source_h5), str(args.cache)]
        if args.dataset == "clinvar_20260628":
            data_paths.append(str(COORD))
        for row in rows:
            subprocess.run([
                sys.executable, str(REGISTRY), "add",
                "--model", args.model, "--model-family", "ours",
                "--dataset", args.dataset, "--dataset-n", str(dataset_n),
                "--partition", row["partition"], "--partition-n", str(row["n"]),
                "--scoring", scoring, "--window-bp", "4375",
                "--auroc", str(row["auroc"]), "--auprc", str(row["auprc"]),
                "--source", str(args.scores), "--script",
                "scripts/aggregate_deltamax_cache_k4.py",
                "--checkpoint", args.checkpoint, "--git-commit", args.git_commit,
                "--notes", ("K=4 supervised-target chunks; v4-aligned indels; "
                            "8,750-position [-4375,+4375) window; canonical "
                            f"masking {'premerge' if masked else 'off'}." + short_note),
                "--data-path", *data_paths,
            ], check=True)


if __name__ == "__main__":
    main()
