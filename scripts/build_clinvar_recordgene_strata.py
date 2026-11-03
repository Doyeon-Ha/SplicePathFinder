"""Splice-site distance fields of the ClinVar 2026 release, as ClinVar describes them.

Each variant is placed by the HGVS in NCBI's preferred variant name, the c./n.
position in variant_summary Name ``NM_...(GENE):c....``. This is not necessarily
the submitter's original transcript. An intronic position (c.N+k or c.N-k,
also in the UTRs) gives the side, the site kind (+ donor, - acceptor) and
``intron_nt`` = k; a range inside an intron takes its end nearer the exon. An
exonic position (c.N, c.-N, c.*N) is measured on the ClinVar gene's transcript in
the GENCODE v49 SpliceAI annotation, from the boundary exon base where the backbone
puts its peak, so ``exon_nt`` 1 is the last base before a donor or the first after
an acceptor. When that transcript has no exon there, or the gene has no intron in
the annotation, the exon distance is unknown and the row stays outside the
universe. A Name without one c./n. side (g./m. Names, uncertain positions, ranges
across an exon boundary) is placed on the ClinVar gene's annotation transcript, or
on the nearest site of any annotated gene when that gene has no intron there.

``hgvs_class`` records the path per row: 0 annotation only, 1 HGVS intronic, 2 HGVS
exonic measured on the ClinVar gene, 3 HGVS exonic without an exon distance.
``site_source`` records the annotation path: 0 ClinVar gene, 1 any annotated gene,
2 no annotated site on the chromosome.

The splice-region universe is ``intron_nt >= 1`` or ``exon_nt`` 1-3, and the Table 9
strata follow from ``intron_nt`` / ``exon_nt`` / ``akind`` (scripts/table9_strata.py).
The output keeps the keys of coord_splice_fields_seed42.npz, row-aligned to the
release table, adds ``record_gene``, ``site_source`` and ``hgvs_class``, and writes a
JSON summary next to it. Before anything is written, the same nearest-site search
under the stored site convention and over every gene must reproduce
coord_splice_fields_seed42.npz row for row.

    python -m scripts.build_clinvar_recordgene_strata
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np

from scripts.table9_strata import table9_masks

NAS = Path(".")
RELEASE = (NAS / "data/eval/clinvar_20260628_gencode_anchored_indel"
           / "clinvar_20260628_gencode_anchored_indel.h5")
STORED = NAS / "data/eval/clinvar_pb_2026_split/coord_splice_fields_seed42.npz"
ANNOT = NAS / "spliceai_train/reference/spliceai_annotation_v49.txt"
SUMMARY = NAS / "data/clinvar_archive/variant_summary.txt.gz"
OUT = (NAS / "data/eval/clinvar_pb_2026_split"
       / "coord_splice_fields_recordgene_hgvs_seed42.npz")

DONOR, ACCEPTOR = 0, 1
ANNOTATION, HGVS_INTRON, HGVS_EXON, HGVS_EXON_UNMEASURED = 0, 1, 2, 3
NAME_RE = re.compile(r"^[^(:\s]+(?:\(([^)]+)\))?:([cnmg])\.(.*)$")
POS_RE = re.compile(r"^[-*]?\d+(?:([+-])(\d+))?")
Sites = tuple[np.ndarray, np.ndarray, np.ndarray]


def annotation_sites(
    lines: Iterable[str], convention: str = "boundary_exon"
) -> tuple[dict[str, Sites], dict[tuple[str, str], Sites]]:
    """Index annotation splice sites per chromosome and per (gene, chromosome).

    Args:
        lines: Annotation rows without the header (NAME CHROM STRAND TX_START
            TX_END EXON_START EXON_END; 0-based starts, 1-based ends).
        convention: ``boundary_exon`` puts every site on its boundary exon base;
            ``stored`` reproduces coord_splice_fields_seed42.npz, which takes the
            0-based EXON_START (the last intron base) for + strand acceptors and
            - strand donors.

    Returns:
        Site position, kind (0 donor, 1 acceptor) and sign (+1 when the intron
        lies above the site), sorted by position with ties in annotation order.
        Genes without an intron add no sites.
    """
    shift = {"boundary_exon": 1, "stored": 0}[convention]
    by_chrom = collections.defaultdict(list)
    by_gene = collections.defaultdict(list)
    for line in lines:
        p = line.rstrip("\n").split("\t")
        name, chrom, strand = p[0], p[1], p[2]
        starts = [int(x) for x in p[5].split(",") if x]
        ends = [int(x) for x in p[6].split(",") if x]
        if len(starts) < 2:
            continue
        if strand == "+":
            sites = ([(e, DONOR, 1) for e in ends[:-1]]
                     + [(s + shift, ACCEPTOR, -1) for s in starts[1:]])
        else:
            sites = ([(s + shift, DONOR, -1) for s in starts[1:]]
                     + [(e, ACCEPTOR, 1) for e in ends[:-1]])
        by_chrom[chrom].extend(sites)
        by_gene[(name, chrom)].extend(sites)

    def index(sites: list) -> Sites:
        a = np.array(sorted(sites, key=lambda t: t[0]), dtype=np.int64)
        a = a.reshape(-1, 3)
        return a[:, 0], a[:, 1], a[:, 2]

    return ({c: index(v) for c, v in by_chrom.items()},
            {g: index(v) for g, v in by_gene.items()})


def nearest_site(sites: Sites, pos: int) -> tuple[int, int, bool]:
    """Return the nearest site's distance and kind, and whether pos is intronic.

    Equal distances go to the lower coordinate, as in the stored fields.
    """
    sp, sk, ss = sites
    i = int(np.searchsorted(sp, pos))
    best = None
    for j in (i - 1, i):
        if 0 <= j < sp.size:
            d = abs(pos - int(sp[j]))
            if best is None or d < best[0]:
                best = (d, int(sk[j]), (pos - int(sp[j])) * int(ss[j]) > 0)
    return best


def record_gene(name: str) -> str:
    """Return the gene of the preferred transcript in a Name, '' if none."""
    m = NAME_RE.match(name or "")
    return (m.group(1) or "") if m else ""


def record_position(name: str) -> tuple[str, int, int] | None:
    """Return the side a c./n. Name gives, with its intronic offset and site kind.

    ``('intron', k, kind)`` for c.N+k or c.N-k; a range inside an intron takes the
    end nearer the exon. ``('exon', -1, -1)`` for c.N, c.-N, c.*N and exonic ranges.
    None for any other Name: no c./n. position, an uncertain position such as
    c.(123+1_124-1), or a range across an exon boundary.
    """
    m = NAME_RE.match(name or "")
    if not m or m.group(2) not in "cn":
        return None
    rest, ends = m.group(3), []
    while True:
        p = POS_RE.match(rest)
        if not p:
            return None
        ends.append((int(p.group(2)), DONOR if p.group(1) == "+" else ACCEPTOR)
                    if p.group(1) else None)
        rest = rest[p.end():]
        if len(ends) == 2 or not rest.startswith("_"):
            break
        rest = rest[1:]
    introns = [e for e in ends if e is not None]
    if not introns:
        return "exon", -1, -1
    if len(introns) < len(ends):
        return None
    offset, kind = min(introns)
    return "intron", offset, kind


def decode(a: np.ndarray) -> np.ndarray:
    """Return an object array of str from a bytes or str array."""
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a],
                    dtype=object)


def record_names(keys: set[str]) -> tuple[dict[str, str], int]:
    """Return the GRCh38 Name per chrom:pos:ref:alt key and the conflicting rows."""
    names, conflicts = {}, 0
    cols = ("Name", "Assembly", "Chromosome", "PositionVCF",
            "ReferenceAlleleVCF", "AlternateAlleleVCF")
    with gzip.open(SUMMARY, "rt") as fh:
        head = fh.readline().lstrip("#").rstrip("\n").split("\t")
        ix = {k: head.index(k) for k in cols}
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if c[ix["Assembly"]] != "GRCh38":
                continue
            k = ":".join(c[ix[f]] for f in cols[2:])
            if k not in keys:
                continue
            if k in names and names[k] != c[ix["Name"]]:
                conflicts += 1
                continue
            names.setdefault(k, c[ix["Name"]])
    return names, conflicts


def strata_counts(intron_nt: np.ndarray, exon_nt: np.ndarray, akind: np.ndarray,
                  labels: np.ndarray) -> dict:
    """Return n and pathogenic n of the universe and its Table 9 strata."""
    universe = (intron_nt >= 1) | ((exon_nt >= 1) & (exon_nt <= 3))
    masks = {"overall": universe, **table9_masks(intron_nt, exon_nt, akind)}
    return {k: {"n": int(m.sum()), "n_pos": int(labels[m].sum())}
            for k, m in masks.items()}


def fields(dist: np.ndarray, is_intron: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return intron_nt and exon_nt (-1 where n/a) from distance and side."""
    return (np.where(is_intron & (dist >= 1), dist, -1),
            np.where(~is_intron & (dist >= 0), dist + 1, -1))


def main() -> None:
    """Write the HGVS-first fields and their summary."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    with h5py.File(RELEASE, "r") as f:
        chrom, pos = decode(f["chrom"][:]), f["pos"][:].astype(np.int64)
        ref, alt = decode(f["ref"][:]), decode(f["alt"][:])
        labels = f["label"][:].astype(np.int64)
        kind = np.array([k.lower() for k in decode(f["variant_kind"][:])])
    stored = np.load(STORED)
    n = len(labels)
    assert np.array_equal(stored["labels"], labels), "release rows != stored fields"
    assert np.array_equal(stored["kind"], kind), "variant kinds != stored fields"
    stored_universe = (stored["intron_nt"] >= 1) | (
        (stored["exon_nt"] >= 1) & (stored["exon_nt"] <= 3))
    assert np.array_equal(stored_universe, stored["splice"])

    with open(ANNOT) as fh:
        lines = fh.readlines()[1:]
    old_chrom, _ = annotation_sites(lines, "stored")
    new_chrom, new_gene = annotation_sites(lines, "boundary_exon")

    keys = [f"{c.removeprefix('chr')}:{p}:{r}:{a}"
            for c, p, r, a in zip(chrom, pos, ref, alt)]
    names, conflicts = record_names(set(keys))

    # Annotation placement: the ClinVar gene's transcript, else any annotated gene.
    a_dist = np.full(n, -1, np.int64)
    a_kind = np.full(n, -1, np.int8)
    a_intron = np.zeros(n, bool)
    source = np.full(n, 2, np.int8)
    rec_gene = np.empty(n, dtype=object)
    old = np.full((n, 3), -1, np.int64)
    for i in range(n):
        c, p = chrom[i], int(pos[i])
        rec_gene[i] = record_gene(names.get(keys[i], ""))
        if c in old_chrom:
            old[i] = nearest_site(old_chrom[c], p)
        sites, source[i] = new_gene.get((rec_gene[i], c)), 0
        if sites is None:
            sites, source[i] = new_chrom.get(c), 1
        if sites is None:
            source[i] = 2
            continue
        a_dist[i], a_kind[i], a_intron[i] = nearest_site(sites, p)
    for j, field in enumerate(("dist", "akind", "is_intron")):
        same = int((old[:, j] == stored[field].astype(np.int64)).sum())
        assert same == n, f"stored-convention {field} reproduces {same}/{n} rows"

    # The preferred Name's HGVS decides the side and the intronic distance.
    dist, akind, is_intron = a_dist.copy(), a_kind.copy(), a_intron.copy()
    hgvs = np.full(n, ANNOTATION, np.int8)
    for i in range(n):
        placed = record_position(names.get(keys[i], ""))
        if placed is None:
            continue
        side, offset, site_kind = placed
        if side == "intron":
            dist[i], akind[i], is_intron[i] = offset, site_kind, True
            hgvs[i] = HGVS_INTRON
        elif source[i] == 0 and not a_intron[i]:
            hgvs[i] = HGVS_EXON
        else:
            dist[i], akind[i], is_intron[i] = -1, -1, False
            hgvs[i] = HGVS_EXON_UNMEASURED

    intron_nt, exon_nt = fields(dist, is_intron)
    splice = (intron_nt >= 1) | ((exon_nt >= 1) & (exon_nt <= 3))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, dist=dist, is_intron=is_intron, akind=akind,
             intron_nt=intron_nt, exon_nt=exon_nt, splice=splice, labels=labels,
             kind=kind, record_gene=rec_gene.astype(str), site_source=source,
             hgvs_class=hgvs)

    a_intron_nt, a_exon_nt = fields(a_dist, a_intron)
    a_universe = (a_intron_nt >= 1) | ((a_exon_nt >= 1) & (a_exon_nt <= 3))
    hgvs_intron = hgvs == HGVS_INTRON
    entrants, leavers = splice & ~stored_universe, ~splice & stored_universe

    def count(m: np.ndarray) -> dict:
        return {"n": int(m.sum()), "n_pos": int(labels[m].sum())}

    summary = {
        "n": n,
        "record_name_found": len(names),
        "record_name_conflicts": conflicts,
        "hgvs_class": {
            name: {**count(hgvs == v), "in_universe": count((hgvs == v) & splice)}
            for name, v in (("annotation", ANNOTATION), ("hgvs_intron", HGVS_INTRON),
                            ("hgvs_exon", HGVS_EXON),
                            ("hgvs_exon_unmeasured", HGVS_EXON_UNMEASURED))},
        "site_source": {"record_gene": int((source == 0).sum()),
                        "any_gene_fallback": int((source == 1).sum()),
                        "none": int((source == 2).sum())},
        "hgvs_intron_vs_annotation": {
            "annotation_exonic": count(hgvs_intron & ~a_intron),
            "annotation_other_kind": count(hgvs_intron & a_intron & (a_kind != akind)),
            "annotation_other_offset": count(hgvs_intron & a_intron & (a_kind == akind)
                                             & (a_dist != dist)),
        },
        "universe": {
            "hgvs_first": count(splice),
            "annotation_only": count(a_universe),
            "stored": count(stored_universe),
            "entrants_vs_stored": count(entrants),
            "leavers_vs_stored": count(leavers),
        },
        "strata": {
            "hgvs_first": strata_counts(intron_nt, exon_nt, akind, labels),
            "annotation_only": strata_counts(a_intron_nt, a_exon_nt, a_kind, labels),
            "stored": strata_counts(stored["intron_nt"], stored["exon_nt"],
                                    stored["akind"], labels),
        },
        "inputs": [str(p) for p in (RELEASE, STORED, ANNOT, SUMMARY)],
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
