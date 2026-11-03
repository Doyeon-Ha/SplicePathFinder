"""BRCA Exchange 릴리스 → 변이표 H5 (SSNet BRCA 벤치마크 재현).

BRCA Exchange(`built_with_change_types.tsv`)의 BRCA1/BRCA2 변이를 004 gencode-anchored
빌더가 먹는 변이표 H5로 만든다. SSNet(Miyachi & Nakai 2025)의 BRCA 벤치마크에 해당.

두 라벨 subset을 지원한다(SSNet과 정합):
  * ``expert``: ``Pathogenicity_expert``(ENIGMA 전문가 심사)만 — Pathogenic/Likely
    Pathogenic → 1, Benign*/Likely benign → 0. (SSNet "expert label" ≈ 4909P/2504N)
  * ``all``: ``Pathogenicity_all``(ENIGMA+ClinVar 통합) — 문자열에 pathogenic만 있으면
    1, benign만 있으면 0, VUS/충돌/혼재는 제외. (SSNet "all label" ≈ 7959P/6566N)

좌표는 hg38(``Genomic_Coordinate_hg38`` 또는 ``chr:POS:REF>ALT``)를 파싱한다. indel은
VCF-anchored(좌측 앵커) 형식이라 그대로 004 빌더에 넣는다. 과도하게 큰 구조 변이는
``--max-indel``로 제외한다.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from process_input.clinvar_to_splicevardb_format import _resolve_strand  # noqa: E402
from process_input.gene_strand_lookup import load_gene_strand_from_gtf  # noqa: E402

logger = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)

_DEFAULT_GTF = "./gencode/gencode.v49.annotation.gtf"
_ACGT_RE = re.compile(r"^[ACGT]+$")
# "chr13:g.32314514:C>T" 또는 "chr13:32315532:C>T"
_COORD_RE = re.compile(r"^(chr[\dXY]+):(?:g\.)?(\d+):([ACGT]*)>([ACGT]*)$")

_EXPERT_PATH = {"Pathogenic", "Likely Pathogenic"}
_EXPERT_BENIGN = {
    "Benign / Little Clinical Significance", "Likely benign", "Likely Benign",
}


def _label_expert(val: str) -> int | None:
    """``Pathogenicity_expert`` 값을 이진 label로 변환한다.

    Args:
        val (str): BRCA Exchange 전문가 심사 결과.

    Returns:
        int | None: 1=pathogenic, 0=benign, None=미심사/VUS(제외).
    """
    if val in _EXPERT_PATH:
        return 1
    if val in _EXPERT_BENIGN:
        return 0
    return None


def _label_all(val: str) -> int | None:
    """``Pathogenicity_all``(ENIGMA+ClinVar 통합) 값을 이진 label로 변환한다.

    pathogenic 표기만 있으면 1, benign 표기만 있으면 0. 둘 다/VUS/충돌은 제외.

    Args:
        val (str): 통합 임상 유의성 문자열.

    Returns:
        int | None: 1/0/None.
    """
    if not val or val == "-":
        return None
    low = val.lower()
    has_path = "pathogenic" in low
    has_benign = "benign" in low
    has_vus = "uncertain" in low or "conflicting" in low
    if has_path and not has_benign and not has_vus:
        return 1
    if has_benign and not has_path and not has_vus:
        return 0
    return None


def _parse_coord(coord: str) -> tuple[str, int, str, str] | None:
    """HGVS 유사 좌표를 (chrom, pos, ref, alt)로 파싱한다.

    Args:
        coord (str): ``chr13:g.32314514:C>T`` 또는 ``chr13:32315532:C>T``.

    Returns:
        tuple | None: (chrom, pos, ref, alt) 또는 파싱 실패 시 None.
    """
    m = _COORD_RE.match(coord.strip())
    if not m:
        return None
    chrom, pos, ref, alt = m.group(1), int(m.group(2)), m.group(3), m.group(4)
    if not (_ACGT_RE.match(ref) and _ACGT_RE.match(alt)):
        return None
    return chrom, pos, ref, alt


def build_columns(tsv_path: str, subset: str, max_indel: int) -> dict[str, list]:
    """BRCA Exchange TSV를 파싱해 변이표 컬럼 dict를 만든다.

    Args:
        tsv_path (str): ``built_with_change_types.tsv`` 경로.
        subset (str): ``expert`` 또는 ``all``.
        max_indel (int): 최대 indel 길이(max(len(ref),len(alt))). 초과 시 제외.

    Returns:
        dict[str, list]: chrom/pos/ref/alt/label/gene_symbol/variant_id 컬럼.
    """
    labeler = _label_expert if subset == "expert" else _label_all
    cols: dict[str, list] = {
        k: [] for k in
        ("chrom", "pos", "ref", "alt", "label", "gene_symbol", "variant_id")
    }
    n_total = n_label = n_coord = n_big = 0
    with open(tsv_path, newline="") as fh:
        r = csv.reader(fh, delimiter="\t")
        hdr = next(r)
        idx = {h: i for i, h in enumerate(hdr)}
        c_gene = idx["Gene_Symbol"]
        c_coord = idx["Genomic_Coordinate_hg38"]
        c_lab = idx["Pathogenicity_expert" if subset == "expert" else "Pathogenicity_all"]
        for row in r:
            n_total += 1
            label = labeler(row[c_lab])
            if label is None:
                continue
            n_label += 1
            parsed = _parse_coord(row[c_coord])
            if parsed is None:
                continue
            chrom, pos, ref, alt = parsed
            if max(len(ref), len(alt)) > max_indel:
                n_big += 1
                continue
            n_coord += 1
            cols["chrom"].append(chrom)
            cols["pos"].append(pos)
            cols["ref"].append(ref)
            cols["alt"].append(alt)
            cols["label"].append(label)
            cols["gene_symbol"].append(row[c_gene])
            cols["variant_id"].append(row[c_coord])
    logger.info(
        "subset=%s: 총 %d, label 있음 %d, 좌표 OK %d (large-indel drop %d)",
        subset, n_total, n_label, n_coord, n_big,
    )
    logger.info(
        "label dist: pathogenic=%d benign=%d",
        sum(cols["label"]), len(cols["label"]) - sum(cols["label"]),
    )
    return cols


def write_variant_table(cols: dict[str, list], gtf_path: str, out_h5: str) -> None:
    """변이표 컬럼에 strand를 더해 004 빌더 호환 H5로 저장한다.

    Args:
        cols (dict[str, list]): `build_columns` 산출 컬럼.
        gtf_path (str): gene_symbol→strand 조회용 GENCODE GTF.
        out_h5 (str): 출력 H5 경로.
    """
    gene_strand = load_gene_strand_from_gtf(gtf_path)
    strands = np.array(
        [_resolve_strand(g, gene_strand) for g in cols["gene_symbol"]], dtype=object
    )
    n = len(cols["pos"])
    logger.info("strand 해결 %d/%d", int((strands != "").sum()), n)
    empty = np.array([""] * n, dtype=object)
    str_dt = h5py.string_dtype("utf-8")
    out_path = Path(out_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("label", data=np.array(cols["label"], dtype=np.int8))
        h5.create_dataset("pos", data=np.array(cols["pos"], dtype=np.int64))
        for key in ("chrom", "ref", "alt", "gene_symbol", "variant_id"):
            h5.create_dataset(key, data=np.array(cols[key], dtype=object), dtype=str_dt)
        h5.create_dataset("strand", data=strands, dtype=str_dt)
        h5.create_dataset("gene_location", data=empty, dtype=str_dt)  # 004 필수(미사용)
        h5.create_dataset("gene_region", data=empty, dtype=str_dt)
        h5.attrs["source"] = "brca_exchange"
        h5.attrs["n"] = n
    logger.info("변이표 저장: %s (n=%d)", out_path, n)


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다.

    Returns:
        argparse.Namespace: 파싱된 인자.
    """
    p = argparse.ArgumentParser(
        description="BRCA Exchange → 변이표 H5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tsv", required=True)
    p.add_argument("--out-h5", required=True, dest="out_h5")
    p.add_argument("--subset", choices=["expert", "all"], default="expert")
    p.add_argument("--gtf", default=_DEFAULT_GTF)
    p.add_argument("--max-indel", type=int, default=50, dest="max_indel")
    return p.parse_args()


def main() -> None:
    """진입점: TSV 파싱 → strand → 변이표 H5."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    cols = build_columns(args.tsv, args.subset, args.max_indel)
    write_variant_table(cols, args.gtf, args.out_h5)


if __name__ == "__main__":
    main()
