"""ClinVar 변이를 각자의 v50 MANE Select 전사체 위에 놓아 거리 층을 만든다.

기존 층은 두 가지뿐이다 — GENCODE v49 canonical 전사체 기준(원고 Table 2)과
NCBI 대표 HGVS 기준(인트론 거리만 MANE에 가깝고 엑손 거리는 v49). v50 MANE
백본이 실제로 보는 전사체 기준의 층은 없었다.

여기서는 변이마다 **보고유전자 anchor 정책이 고른 그 유전자의 MANE Select
전사체**에서 최근접 splice site를 찾는다. 좌표 규약은 기존 층과 같아야 비교가
성립하므로 ``build_clinvar_recordgene_strata``의 ``annotation_sites``(boundary
exon)·``nearest_site``·``fields``를 그대로 쓴다.

출력은 릴리스 378,543행에 정렬되며 ``coord_splice_fields_*`` 와 같은 키를 갖는다.

    python -m scripts.build_clinvar_v50mane_strata --out <npz>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from process_input.variant_identity import variant_keys, require_equal, row_digest

from scripts.build_clinvar_recordgene_strata import (
    annotation_sites,
    fields,
    nearest_site,
    strata_counts,
)

# MANE regions table: name, ?, chrom, strand, tx_start, tx_end, intron starts,
# intron ends, exon starts, exon ends, ... Only these seven reach the site index.
COLUMNS = (0, 2, 3, 4, 5, 8, 9)


def dec(a) -> np.ndarray:
    """바이트 또는 문자열 배열을 str 배열로 바꾼다."""
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def mane_lines(path: Path) -> list[str]:
    """MANE regions 표를 annotation_sites가 읽는 7컬럼 형식으로 바꾼다."""
    out = []
    with path.open() as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 10:
                raise SystemExit(f"MANE row has {len(p)} columns: {p[:3]}")
            out.append("\t".join(p[c] for c in COLUMNS) + "\n")
    return out


def main() -> None:
    """MANE Select 전사체 기준 거리 층을 쓴다."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument('--release-h5', type=Path, required=True)
    ap.add_argument('--mane-table', type=Path, required=True)
    ap.add_argument('--anchor-selection', type=Path, required=True)
    args = ap.parse_args()
    RELEASE, MANE, SEL = args.release_h5, args.mane_table, args.anchor_selection

    with h5py.File(RELEASE) as f:
        chrom, pos = dec(f["chrom"][:]), f["pos"][:].astype(np.int64)
        labels = f["label"][:].astype(np.int8)
        key = variant_keys(f)
    n = len(labels)
    z = np.load(SEL, allow_pickle=True)
    anchor = np.asarray(z["gene"]).astype(str)
    require_equal(key, z['key'], 'anchor selection')
    if len(anchor) != n:
        raise SystemExit(f"anchor rows {len(anchor)} != release rows {n}")

    lines = mane_lines(MANE)
    by_chrom, by_gene = annotation_sites(lines, "boundary_exon")
    print(f"MANE transcripts: {len(lines)}; chromosomes with sites: {len(by_chrom)}",
          flush=True)

    dist = np.full(n, -1, np.int64)
    akind = np.full(n, -1, np.int8)
    is_intron = np.zeros(n, bool)
    source = np.full(n, 2, np.int8)  # 0 anchor gene, 1 any MANE gene, 2 none
    for i in range(n):
        c, p, g = chrom[i], int(pos[i]), anchor[i]
        sites = by_gene.get((g, c)) if g else None
        if sites is not None:
            source[i] = 0
        else:
            sites, source[i] = by_chrom.get(c), 1
        if sites is None:
            source[i] = 2
            continue
        dist[i], akind[i], is_intron[i] = nearest_site(sites, p)
        if i % 50000 == 0:
            print(f"  placed {i}/{n}", flush=True)

    intron_nt, exon_nt = fields(dist, is_intron)
    splice = (intron_nt >= 1) | ((exon_nt >= 1) & (exon_nt <= 3))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, dist=dist, is_intron=is_intron, akind=akind,
             key=key, row_key_sha256=row_digest(key),
             intron_nt=intron_nt, exon_nt=exon_nt, splice=splice, labels=labels,
             anchor_gene=anchor, site_source=source)
    summary = {
        "n": int(n),
        "placed_on_anchor_gene": int((source == 0).sum()),
        "placed_on_any_mane_gene": int((source == 1).sum()),
        "unplaced": int((source == 2).sum()),
        "strata": strata_counts(intron_nt, exon_nt, akind, labels),
        "inputs": [str(RELEASE), str(MANE), str(SEL)],
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
