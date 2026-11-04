"""Per-variant ClinVar *era* annotation for the pre-circularity (pre-tool) analysis.

Circularity risk: ClinVar submitters have used SpliceAI (Cell, online 2019-01-17)
and Pangolin (bioRxiv 2021-07-06) as evidence (ACMG PP3/BP4), so a splice model
evaluated on present-day ClinVar labels is partly scored against its own
predictions. This script annotates every variant of the ClinVar-2026 evaluation
universe with *when* its classification existed, so the benchmark can be restricted
to variants whose labels predate the tools.

Two independent date signals per variant (both written; masks are formed downstream):
  1. archive presence  -- was the variant already in the ClinVar GRCh38 release of
     2018-12-17 (pre-SpliceAI) / 2021-06-26 (pre-Pangolin), and with which
     classification tier + review star *at that time*?
  2. submission dates  -- from the current ``submission_summary.txt.gz``, the
     min/max ``DateLastEvaluated`` over all submissions and over the submissions
     that agree with the variant's current binary tier.

Row-aligned to the canonical ClinVar-2026 cache (378,543 rows) used by
``scripts/aggregate_cv5_table9_union_toprows.py``, so the output npz can be used
directly as a boolean filter on the Table-6 per-variant score columns.
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
import time

import h5py
import numpy as np

_T0 = time.time()


def _tick(msg: str) -> None:
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


NAS = "."
CACHE = f"{NAS}/data/eval/sav_pernt_cache/clinvar_20260628_med_ei_ac_ep14/cache.h5"
ARCH = f"{NAS}/data/clinvar_archive"
CUR_VCF = f"{ARCH}/clinvar_20260628.vcf.gz"
# release just before each tool became public (see module docstring)
ERA_VCF = {"2018": f"{ARCH}/clinvar_20181217.vcf.gz", "2021": f"{ARCH}/clinvar_20210626.vcf.gz"}
SUBSUM = f"{ARCH}/submission_summary.txt.gz"
OUT = f"{NAS}/data/eval/clinvar_pb_2026_split/precirc_era_annotation.npz"

_PATH = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic",
         "pathogenic/likely pathogenic/pathogenic, low penetrance",
         "pathogenic/likely pathogenic/established risk allele"}
_BEN = {"benign", "likely benign", "benign/likely benign"}


def _tier(clnsig: str) -> int:
    """ClinVar significance string -> 1 (pathogenic tier) / 0 (benign tier) / -1."""
    norm = clnsig.replace("_", " ").strip().lower()
    if norm in _PATH:
        return 1
    if norm in _BEN:
        return 0
    return -1


def _star(revstat: str) -> int:
    """CLNREVSTAT / ReviewStatus text -> ClinVar review star (0-4)."""
    s = revstat.replace("_", " ").strip().lower()
    if "practice guideline" in s:
        return 4
    if "expert panel" in s:
        return 3
    if "criteria provided" in s:
        # conflicting = 1 star even with multiple submitters
        if "conflict" in s:
            return 1
        return 2 if "multiple submitters" in s else 1
    return 0


def _bare(c: str) -> str:
    c = str(c)
    return c[3:] if c.startswith("chr") else c


def _key(c, p, r, a) -> str:
    return f"{_bare(c)}:{int(p)}:{r}:{a}"


def _dec(arr):
    return [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]


def _parse_date(text: str) -> int:
    """ClinVar date text -> YYYYMMDD int (0 when absent/unparseable).

    Handles the two shapes seen in the tab-delimited dumps: ``Jun 29, 2015`` and
    ``2015-06-29``.
    """
    t = text.strip()
    if not t or t == "-":
        return 0
    if len(t) >= 10 and t[4] == "-" and t[7] == "-":
        try:
            return int(t[:4] + t[5:7] + t[8:10])
        except ValueError:
            return 0
    mon = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    try:
        head, year = t.split(",")
        mname, day = head.split()
        return int(year) * 10000 + mon[mname[:3].lower()] * 100 + int(day)
    except (ValueError, KeyError):
        return 0


def _vcf_rows(path: str):
    """Yield (ident, info_dict_str, vcf_id) per ACGT-only ALT allele of a ClinVar VCF."""
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            chrom, pos, vid, ref, alt, info = f[0], f[1], f[2], f[3], f[4], f[7]
            for one in alt.split(","):
                yield _key(chrom, pos, ref.upper(), one.upper()), info, vid


def _info_get(info: str, field: str) -> str:
    """Pull one INFO field value without building the whole dict (hot loop)."""
    tag = field + "="
    for part in info.split(";"):
        if part.startswith(tag):
            return part[len(tag):]
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    with h5py.File(args.cache, "r") as f:
        chrom = _dec(f["chrom"][:])
        pos = f["pos"][:].astype(np.int64)
        ref = _dec(f["ref"][:])
        alt = _dec(f["alt"][:])
        lab = f["labels"][:].astype(np.int8)
    n = len(chrom)
    ident = [_key(chrom[i], pos[i], ref[i], alt[i]) for i in range(n)]
    row_of = {k: i for i, k in enumerate(ident)}
    assert len(row_of) == n, f"duplicate identity keys: {n - len(row_of)}"
    _tick(f"cache universe n={n}")

    # ---- current release: identity -> VariationID (VCF ID column) ----
    varid = np.zeros(n, dtype=np.int64)
    hit = 0
    for k, _info, vid in _vcf_rows(CUR_VCF):
        i = row_of.get(k)
        if i is not None and varid[i] == 0:
            try:
                varid[i] = int(vid)
                hit += 1
            except ValueError:
                pass
    _tick(f"VariationID mapped {hit}/{n} ({100.0 * hit / n:.1f}%)")

    # ---- archive releases: presence + tier + star at that time ----
    era = {}
    for tag, path in ERA_VCF.items():
        present = np.zeros(n, dtype=bool)
        tier = np.full(n, -1, dtype=np.int8)
        star = np.zeros(n, dtype=np.int8)
        for k, info, _vid in _vcf_rows(path):
            i = row_of.get(k)
            if i is None or present[i]:
                continue
            present[i] = True
            tier[i] = _tier(_info_get(info, "CLNSIG"))
            star[i] = _star(_info_get(info, "CLNREVSTAT"))
        era[tag] = (present, tier, star)
        _tick(f"{tag}: present={int(present.sum())} concordant="
              f"{int((present & (tier == lab)).sum())} concordant2star="
              f"{int((present & (tier == lab) & (star >= 2)).sum())}")

    # ---- submission dates (current release, per SCV) ----
    want = {int(v): i for i, v in enumerate(varid) if v}
    nsub = np.zeros(n, dtype=np.int32)
    nsub_conc = np.zeros(n, dtype=np.int32)
    d_min_conc = np.full(n, 0, dtype=np.int32)
    d_max_conc = np.zeros(n, dtype=np.int32)
    d_max_any = np.zeros(n, dtype=np.int32)
    n_undated = 0
    with gzip.open(SUBSUM, "rt") as fh:
        header = None
        for line in fh:
            if line.startswith("#"):
                header = line
                continue
            f = line.rstrip("\n").split("\t")
            try:
                i = want[int(f[0])]
            except (KeyError, ValueError):
                continue
            tier = _tier(f[1])
            date = _parse_date(f[2])
            nsub[i] += 1
            if date == 0:
                n_undated += 1
            d_max_any[i] = max(d_max_any[i], date)
            if tier == lab[i]:
                nsub_conc[i] += 1
                d_max_conc[i] = max(d_max_conc[i], date)
                if date and (d_min_conc[i] == 0 or date < d_min_conc[i]):
                    d_min_conc[i] = date
    _tick(f"submissions: header={header.split(chr(9))[:3] if header else None} "
          f"variants with >=1 sub={int((nsub > 0).sum())} undated rows={n_undated}")

    np.savez_compressed(
        args.out,
        ident=np.array(ident, dtype=object),
        label=lab,
        varid=varid,
        present_2018=era["2018"][0], tier_2018=era["2018"][1], star_2018=era["2018"][2],
        present_2021=era["2021"][0], tier_2021=era["2021"][1], star_2021=era["2021"][2],
        nsub=nsub, nsub_conc=nsub_conc,
        date_min_conc=d_min_conc, date_max_conc=d_max_conc, date_max_any=d_max_any,
    )
    _tick(f"wrote {args.out}")

    # ---- summary of the candidate era masks ----
    p18, t18, s18 = era["2018"]
    p21, t21, s21 = era["2021"]
    masks = {
        "all": np.ones(n, bool),
        "pre_spliceai_release": p18 & (t18 == lab),
        "pre_spliceai_release_2star": p18 & (t18 == lab) & (s18 >= 2),
        "pre_spliceai_subdate": (d_max_conc > 0) & (d_max_conc < 20190117),
        "pre_pangolin_release": p21 & (t21 == lab),
        "pre_pangolin_release_2star": p21 & (t21 == lab) & (s21 >= 2),
        "pre_pangolin_subdate": (d_max_conc > 0) & (d_max_conc < 20210706),
    }
    print(f"\n{'mask':32s} {'n':>8s} {'pos':>7s} {'neg':>7s}")
    for name, m in masks.items():
        print(f"{name:32s} {int(m.sum()):8d} {int(lab[m].sum()):7d} {int((lab[m] == 0).sum()):7d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
