"""ClinVar VCF → 변이표 파서 단위 테스트 (spec 016 US1)."""

import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from process_input import clinvar_vcf_to_variant_table as cv

_VCF_HEADER = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"


def _info(clnsig: str, revstat: str, gene: str, mc: str = "SO:1|x") -> str:
    return f"CLNSIG={clnsig};CLNREVSTAT={revstat};GENEINFO={gene};MC={mc}"


def _write_vcf(tmp_path: Path) -> str:
    rows = [
        # SNV pathogenic, 2-star → keep label 1
        ("1", "100", "VCV1", "C", "T",
         _info("Pathogenic", "criteria_provided,_multiple_submitters,_no_conflicts", "GENEA:1")),
        # indel (del) benign, 2-star → keep label 0
        ("2", "200", "VCV2", "AG", "A",
         _info("Benign", "criteria_provided,_multiple_submitters,_no_conflicts", "GENEB:2")),
        # indel (ins) likely_pathogenic, 1-star → dropped by min_star=2
        ("3", "300", "VCV3", "A", "AT",
         _info("Likely_pathogenic", "criteria_provided,_single_submitter", "GENEC:3")),
        # VUS → dropped (no label tier)
        ("4", "400", "VCV4", "G", "A",
         _info("Uncertain_significance", "reviewed_by_expert_panel", "GENED:4")),
        # ALT="." (no-alt) pathogenic 2-star → dropped by ACGT filter
        ("5", "500", "VCV5", "G", ".",
         _info("Pathogenic", "criteria_provided,_multiple_submitters,_no_conflicts", "GENEE:5")),
        # symbolic <DEL> → dropped by ACGT filter
        ("6", "600", "VCV6", "G", "<DEL>",
         _info("Pathogenic", "criteria_provided,_multiple_submitters,_no_conflicts", "GENEF:6")),
    ]
    p = tmp_path / "mini.vcf.gz"
    with gzip.open(p, "wt") as f:
        f.write("##fileformat=VCFv4.1\n")
        f.write(_VCF_HEADER)
        for r in rows:
            f.write("\t".join([*r[:5], ".", ".", r[5]]) + "\n")
    return str(p)


def test_clnsig_label() -> None:
    assert cv._clnsig_to_label("Pathogenic") == 1
    assert cv._clnsig_to_label("Likely_pathogenic") == 1
    assert cv._clnsig_to_label("Benign") == 0
    assert cv._clnsig_to_label("Uncertain_significance") is None


def test_gene_symbol() -> None:
    assert cv._gene_symbol("OR4F5:79501") == "OR4F5"
    assert cv._gene_symbol("A:1|B:2") == "A|B"
    assert cv._gene_symbol("") == ""


def test_revstat_star() -> None:
    assert cv._REVSTAT_TO_STAR["reviewed_by_expert_panel"] == 3
    assert cv._REVSTAT_TO_STAR["criteria_provided,_single_submitter"] == 1


def test_parse_keeps_snv_and_indel_filters_star_and_vus(tmp_path: Path) -> None:
    vcf = _write_vcf(tmp_path)
    cols = cv.parse_clinvar_vcf(vcf, min_star=2, keep_indels=True)
    # VCV1(snv,2*) + VCV2(indel,2*) 유지; VCV3(1*) VCV4(VUS) 제외
    assert len(cols["pos"]) == 2
    assert cols["chrom"] == ["chr1", "chr2"]
    assert cols["label"] == [1, 0]
    kinds = [len(r) == 1 and len(a) == 1 for r, a in zip(cols["ref"], cols["alt"])]
    assert kinds == [True, False]  # snv, indel


def test_snv_only_drops_indel(tmp_path: Path) -> None:
    vcf = _write_vcf(tmp_path)
    cols = cv.parse_clinvar_vcf(vcf, min_star=2, keep_indels=False)
    assert len(cols["pos"]) == 1  # SNV만
    assert cols["chrom"] == ["chr1"]
