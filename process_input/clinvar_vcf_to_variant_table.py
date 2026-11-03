"""최신 ClinVar VCF → SpliceVarDB-format 변이표 H5 (spec 016).

NCBI ClinVar GRCh38 VCF(gzip)를 파싱해, 004 gencode-anchored indel 빌더
(`splicevardb_gencode_anchored_allchroms_indel_builder.py`)가 먹는 변이표 H5를
생성한다. **SNV와 indel을 모두 유지**한다(기존 ClinVar 빌드의 SNV-only 필터 제거).

라벨/strand 로직은 `clinvar_to_splicevardb_format`의 것을 재사용한다
(`PATHOGENIC_TIER`/`BENIGN_TIER`/`_resolve_strand`). `gene_location`/`gene_region`은
ClinVar에 없으므로 빈 문자열(004 빌더는 anchoring에 사용 안 함). 구버전 ClinVar
산출물은 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import re
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from process_input.clinvar_to_splicevardb_format import (  # noqa: E402
    BENIGN_TIER,
    PATHOGENIC_TIER,
    _resolve_strand,
)
from process_input.gene_strand_lookup import (  # noqa: E402
    load_gene_strand_from_gtf,
)

logger = logging.getLogger(__name__)

# ClinVar VCF는 ALT="."(no-alt)·"<DEL>"/"<DUP>"(symbolic) 레코드를 포함하므로
# ref/alt가 순수 ACGT 문자열인 변이만 유지한다(004 빌더는 ACGT만 인코딩 가능).
_ACGT_RE = re.compile(r"^[ACGTacgt]+$")
_DEFAULT_GTF = "./gencode/gencode.v49.annotation.gtf"
# CLNREVSTAT 텍스트 → ClinVar review star.
_REVSTAT_TO_STAR: dict[str, int] = {
    "practice_guideline": 4,
    "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_single_submitter": 1,
    "criteria_provided,_conflicting_classifications": 1,
    "no_assertion_criteria_provided": 0,
    "no_classification_provided": 0,
    "no_classifications_from_unflagged_records": 0,
    "no_classification_for_the_single_variant": 0,
}


def _parse_info(info: str) -> dict[str, str]:
    """VCF INFO 필드를 dict로 파싱한다.

    Args:
        info (str): 세미콜론 구분 INFO 문자열.

    Returns:
        dict[str, str]: key→value (flag는 value="").
    """
    out: dict[str, str] = {}
    for field in info.split(";"):
        if "=" in field:
            key, val = field.split("=", 1)
            out[key] = val
        else:
            out[field] = ""
    return out


def _clnsig_to_label(clnsig: str) -> int | None:
    """CLNSIG를 이진 label로 변환한다.

    Args:
        clnsig (str): ClinVar ``CLNSIG`` (underscore 구분).

    Returns:
        int | None: 1=pathogenic tier, 0=benign tier, None=그 외(제외).
    """
    norm = clnsig.replace("_", " ").strip()
    if norm in PATHOGENIC_TIER:
        return 1
    if norm in BENIGN_TIER:
        return 0
    return None


def _gene_symbol(geneinfo: str) -> str:
    """GENEINFO("GENE:ID|GENE2:ID2")에서 gene name을 ``|``로 결합한다.

    Args:
        geneinfo (str): VCF ``GENEINFO`` 값.

    Returns:
        str: ``"GENE|GENE2"`` 형태 (없으면 "").
    """
    if not geneinfo:
        return ""
    names = [piece.split(":", 1)[0] for piece in geneinfo.split("|") if piece]
    return "|".join(names)


def parse_clinvar_vcf(
    vcf_path: str, min_star: int, keep_indels: bool
) -> dict[str, list]:
    """ClinVar VCF를 파싱해 변이표 컬럼 dict를 반환한다.

    Args:
        vcf_path (str): ClinVar VCF(.gz) 경로.
        min_star (int): 최소 review star (이상만 유지).
        keep_indels (bool): False면 SNV만 유지.

    Returns:
        dict[str, list]: 컬럼명→리스트 (chrom/pos/ref/alt/label/star/
            gene_symbol/molecular_consequence/variant_vcv).
    """
    cols: dict[str, list] = {
        k: []
        for k in (
            "chrom",
            "pos",
            "ref",
            "alt",
            "label",
            "star",
            "gene_symbol",
            "molecular_consequence",
            "variant_vcv",
        )
    }
    n_total = n_label = n_star = n_kept = 0
    n_snv = n_indel = 0
    with gzip.open(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            n_total += 1
            f = line.rstrip("\n").split("\t")
            chrom, pos, vid, ref, alt = f[0], f[1], f[2], f[3], f[4]
            if "," in alt:  # multi-allelic (ClinVar는 드묾) — skip
                continue
            if not (_ACGT_RE.match(ref) and _ACGT_RE.match(alt)):
                continue  # ALT="."·symbolic(<DEL> 등)·N 포함 제외
            info = _parse_info(f[7])
            label = _clnsig_to_label(info.get("CLNSIG", ""))
            if label is None:
                continue
            n_label += 1
            star = _REVSTAT_TO_STAR.get(info.get("CLNREVSTAT", ""), 0)
            if star < min_star:
                continue
            n_star += 1
            is_snv = len(ref) == 1 and len(alt) == 1
            if not is_snv and not keep_indels:
                continue
            n_snv += int(is_snv)
            n_indel += int(not is_snv)
            cols["chrom"].append(chrom if chrom.startswith("chr") else f"chr{chrom}")
            cols["pos"].append(int(pos))
            cols["ref"].append(ref.upper())
            cols["alt"].append(alt.upper())
            cols["label"].append(label)
            cols["star"].append(star)
            cols["gene_symbol"].append(_gene_symbol(info.get("GENEINFO", "")))
            cols["molecular_consequence"].append(info.get("MC", ""))
            cols["variant_vcv"].append(vid)
            n_kept += 1
    logger.info(
        "parsed total=%d, label-tier=%d, star>=%d=%d, kept=%d (snv=%d, indel=%d)",
        n_total,
        n_label,
        min_star,
        n_star,
        n_kept,
        n_snv,
        n_indel,
    )
    logger.info(
        "label dist: pathogenic=%d benign=%d",
        sum(cols["label"]),
        len(cols["label"]) - sum(cols["label"]),
    )
    return cols


def write_variant_table(cols: dict[str, list], gtf_path: str, out_h5: str) -> None:
    """변이표 컬럼에 strand enrichment를 더해 H5로 저장한다.

    Args:
        cols (dict[str, list]): `parse_clinvar_vcf` 산출 컬럼.
        gtf_path (str): gene_symbol→strand 조회용 GENCODE GTF.
        out_h5 (str): 출력 H5 경로(신버전 경로).
    """
    logger.info("strand 조회 GTF 로드: %s", gtf_path)
    gene_strand = load_gene_strand_from_gtf(gtf_path)
    strands = np.array(
        [_resolve_strand(g, gene_strand) for g in cols["gene_symbol"]],
        dtype=object,
    )
    n = len(cols["pos"])
    n_no_strand = int((strands == "").sum())
    logger.info(
        "strand 해결 %d/%d (미해결 %d → 004 빌더에서 drop 가능)",
        n - n_no_strand,
        n,
        n_no_strand,
    )
    empty = np.array([""] * n, dtype=object)
    str_dt = h5py.string_dtype("utf-8")
    out_path = Path(out_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("label", data=np.array(cols["label"], dtype=np.int8))
        h5.create_dataset("pos", data=np.array(cols["pos"], dtype=np.int64))
        h5.create_dataset("star", data=np.array(cols["star"], dtype=np.int8))
        for key in (
            "chrom",
            "ref",
            "alt",
            "gene_symbol",
            "molecular_consequence",
            "variant_vcv",
        ):
            h5.create_dataset(key, data=np.array(cols[key], dtype=object), dtype=str_dt)
        h5.create_dataset("strand", data=strands, dtype=str_dt)
        # 004 빌더 _read_input 필수 키 — ClinVar엔 없어 빈 문자열(anchoring 미사용)
        h5.create_dataset("gene_location", data=empty, dtype=str_dt)
        h5.create_dataset("gene_region", data=empty, dtype=str_dt)
        h5.attrs["source_vcf"] = str(out_path)
        h5.attrs["n"] = n
    logger.info("변이표 저장: %s (n=%d)", out_path, n)


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다.

    Returns:
        argparse.Namespace: 파싱된 인자.
    """
    p = argparse.ArgumentParser(
        description="ClinVar VCF → SpliceVarDB-format 변이표 H5 (SNV+indel).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--vcf", required=True)
    p.add_argument("--out-h5", required=True, dest="out_h5")
    p.add_argument("--gtf", default=_DEFAULT_GTF)
    p.add_argument("--min-star", type=int, default=2, dest="min_star")
    p.add_argument("--snv-only", action="store_true", help="indel 제외(기본은 포함)")
    return p.parse_args()


def main() -> None:
    """진입점: VCF 파싱 → strand enrichment → 변이표 H5."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    cols = parse_clinvar_vcf(
        args.vcf, min_star=args.min_star, keep_indels=not args.snv_only
    )
    write_variant_table(cols, args.gtf, args.out_h5)


if __name__ == "__main__":
    main()
