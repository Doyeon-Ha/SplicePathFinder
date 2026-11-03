"""Tests for the SpliceAI-style H5 data build pipeline.

Covers all three stages of the pipeline plus the dataset
loader and end-to-end integration:

  Stage 1: canonical_dataset_builder
  Stage 2: h5_datafile_builder
  Stage 3: h5_dataset_builder + h5_utils
  Loader:  SplicingH5Dataset
  E2E:     Full pipeline with tiny mock data
"""

from math import ceil
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from process_input.canonical_dataset_builder import (
    compute_ucsc_cds_spans_and_frames,
    get_canonical_score,
    load_level_from_gtf,
    load_paralog_flags,
    merge_stop_codon,
    parse_gencode_gtf,
    parse_gtf_attributes,
    parse_ucsc_tables,
    select_canonical_transcripts,
    write_canonical_dataset,
)
from process_input.h5_datafile_builder import (
    build_datafile,
    extract_sequence,
)
from process_input.h5_dataset_builder import build_dataset
from process_input.h5_utils import (
    EXON_FIRST,
    EXON_IGNORE,
    EXON_INTERNAL,
    EXON_LAST,
    REGION_CDS_PHASE0,
    REGION_IGNORE,
    REGION_INTRON,
    REGION_NCRNA_EXON,
    REGION_UTR3,
    REGION_UTR5,
    build_exon_position_label_array,
    build_region_label_array,
    ceil_div,
    clip_datapoints,
    create_datapoints,
    reformat_data,
)
from scripts.verify_realflank_h5 import _expected_window, _n_points
from train.dataset import SplicingH5Dataset

# ────────────────────────────────────────────────────
# Shared test constants
# ────────────────────────────────────────────────────
CL_MAX = 10
SL = 20


# ────────────────────────────────────────────────────
# Helper: tiny FASTA via pyfaidx
# ────────────────────────────────────────────────────
def _write_fasta(path: Path, records: dict[str, str]) -> Path:
    """Write a minimal FASTA file.

    Args:
        path (Path): File path.
        records (dict[str, str]): chrom -> sequence.

    Returns:
        Path: The written file path.
    """
    with open(path, "w") as f:
        for name, seq in records.items():
            f.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")
    return path


# ────────────────────────────────────────────────────
# Helper: mock pyfaidx.Fasta via dict
# ────────────────────────────────────────────────────
class _MockChrom:
    """Slice-able chromosome sequence mock."""

    def __init__(self, seq: str) -> None:
        self._seq = seq

    def __len__(self) -> int:
        return len(self._seq)

    def __getitem__(self, key: slice) -> str:
        return self._seq[key]

    def __str__(self) -> str:
        return self._seq


class MockFasta:
    """Minimal pyfaidx.Fasta replacement."""

    def __init__(self, records: dict[str, str]) -> None:
        self._records = {k: _MockChrom(v) for k, v in records.items()}

    def __getitem__(self, key: str) -> _MockChrom:
        return self._records[key]


# ════════════════════════════════════════════════════
#  Stage 1 — canonical_dataset_builder
# ════════════════════════════════════════════════════


class TestParseGtfAttributes:
    """T-01: GTF attribute parsing."""

    def test_basic_attributes(self) -> None:
        raw = (
            'gene_id "ENSG00000000001.1"; '
            'gene_name "TESTGENE"; '
            'transcript_type "protein_coding";'
        )
        attrs = parse_gtf_attributes(raw)
        assert attrs["gene_id"] == "ENSG00000000001.1"
        assert attrs["gene_name"] == "TESTGENE"
        assert attrs["transcript_type"] == "protein_coding"

        return

    def test_empty_string(self) -> None:
        assert parse_gtf_attributes("") == {}

        return

    def test_repeated_tags_kept(self) -> None:
        """GENCODE writes one ``tag`` entry per tag; none is dropped."""
        raw = (
            'gene_id "G1"; transcript_id "T1"; '
            'tag "basic"; tag "Ensembl_canonical"; '
            'tag "MANE_Select"; tag "appris_principal_1"; tag "CCDS";'
        )
        attrs = parse_gtf_attributes(raw)
        assert attrs["tag"].split(",") == [
            "basic",
            "Ensembl_canonical",
            "MANE_Select",
            "appris_principal_1",
            "CCDS",
        ]
        assert attrs["transcript_id"] == "T1"

        return


class TestGetCanonicalScore:
    """T-02: Representative transcript tiers."""

    def test_mane_select(self) -> None:
        """MANE_Select also carries Ensembl_canonical."""
        attrs = {"tag": "basic,Ensembl_canonical,MANE_Select"}
        assert get_canonical_score(attrs) == (1, 0, 1)

        return

    def test_mane_plus_clinical(self) -> None:
        """MANE_Plus_Clinical alone sets the second tier."""
        attrs = {"tag": "basic,MANE_Plus_Clinical"}
        assert get_canonical_score(attrs) == (0, 1, 0)

        return

    def test_ensembl_canonical(self) -> None:
        """Ensembl_canonical alone sets the third tier."""
        attrs = {"tag": "basic,Ensembl_canonical"}
        assert get_canonical_score(attrs) == (0, 0, 1)

        return

    def test_tier_order(self) -> None:
        """Select > Plus Clinical > Ensembl canonical > untagged."""
        select = get_canonical_score({"tag": "MANE_Select"})
        plus = get_canonical_score({"tag": "MANE_Plus_Clinical"})
        canonical = get_canonical_score({"tag": "Ensembl_canonical"})
        assert select > plus > canonical > (0, 0, 0)

        return

    def test_other_tags_ignored(self) -> None:
        """APPRIS, basic and CCDS do not rank a transcript."""
        attrs = {"tag": "basic,appris_principal_1,CCDS"}
        assert get_canonical_score(attrs) == (0, 0, 0)

        return

    def test_no_tags(self) -> None:
        """No tag scores zero."""
        assert get_canonical_score({}) == (0, 0, 0)

        return

    def test_mane_select_found_in_any_tag_position(self) -> None:
        """MANE_Select is found when it is not the last tag of the line."""
        raw = 'tag "MANE_Select"; tag "basic"; tag "CCDS";'
        assert get_canonical_score(parse_gtf_attributes(raw))[0] == 1

        return


class TestParseUcscTables:
    """T-03 ~ T-05: UCSC table parsing."""

    @pytest.fixture()
    def ucsc_files(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create minimal UCSC table files.

        Args:
            tmp_path (Path): pytest tmp dir.

        Returns:
            tuple[Path, Path]: (knownCanonical, gencodeBasic).
        """
        kc = tmp_path / "knownCanonical.txt"
        kc.write_text(
            "#chrom\tchromStart\tchromEnd\tclusterId"
            "\ttranscript\tprotein\n"
            "chr1\t100\t500\t1\tENST001\tENSP001\n"
            "chr1\t600\t900\t2\tENST002\tENSP002\n"
        )

        gb = tmp_path / "gencodeBasic.txt"
        #   0:bin 1:name 2:chrom 3:strand
        #   4:txStart 5:txEnd 6:cdsStart 7:cdsEnd
        #   8:exonCount 9:exonStarts 10:exonEnds
        #   11:score 12:name2 13:cdsStartStat
        #   14:cdsEndStat 15:exonFrames
        gb.write_text(
            "#bin\tname\tchrom\tstrand\ttxStart\t"
            "txEnd\tcdsStart\tcdsEnd\texonCount\t"
            "exonStarts\texonEnds\tscore\tname2\t"
            "cdsStartStat\tcdsEndStat\texonFrames\n"
            "0\tENST001\tchr1\t+\t100\t500\t"
            "120\t480\t3\t100,200,350,\t"
            "180,300,500,\t0\tGENEA\tcmpl\t"
            "cmpl\t0,1,2,\n"
            "0\tENST002\tchr1\t-\t600\t900\t"
            "600\t900\t2\t600,800,\t"
            "700,900,\t0\tGENEB\tcmpl\t"
            "cmpl\t0,1,\n"
        )
        return kc, gb

    def test_returns_two_genes(self, ucsc_files: tuple[Path, Path]) -> None:
        """T-03: Two protein-coding genes parsed."""
        kc, gb = ucsc_files
        result = parse_ucsc_tables(str(kc), str(gb))
        assert len(result) == 2

        return

    def test_coordinate_conversion(self, ucsc_files: tuple[Path, Path]) -> None:
        """T-04: 0-based half-open -> 1-based conversion."""
        kc, gb = ucsc_files
        result = parse_ucsc_tables(str(kc), str(gb))
        gene_a = next(t for t in result if t["gene_name"] == "GENEA")
        assert gene_a["tx_start"] == 101
        assert gene_a["tx_end"] == 500

        # FR-004: Check exact junction coordinate conversion
        # exonStarts: 100,200,350 -> jn_end: 201,351 (1-based)
        assert gene_a["jn_end_str"] == "201,351,"
        # exonEnds: 180,300,500 -> jn_start: 180,300
        assert gene_a["jn_start_str"] == "180,300,"

        return

    def test_junction_strings(self, ucsc_files: tuple[Path, Path]) -> None:
        """T-05: Junction strings have trailing comma."""
        kc, gb = ucsc_files
        result = parse_ucsc_tables(str(kc), str(gb))
        gene_a = next(t for t in result if t["gene_name"] == "GENEA")
        assert gene_a["jn_start_str"].endswith(",")
        assert gene_a["jn_end_str"].endswith(",")

        return

    def test_non_coding_excluded(self, tmp_path: Path) -> None:
        """T-04b: Non-coding transcripts excluded."""
        kc = tmp_path / "kc.txt"
        kc.write_text("#h\nchr1\t0\t100\t1\tENST_NC\tENSP\n")
        gb = tmp_path / "gb.txt"
        gb.write_text(
            "#h\n"
            "0\tENST_NC\tchr1\t+\t0\t100\t"
            "0\t0\t2\t0,50,\t40,100,\t0\t"
            "NCGENE\tcmpl\tcmpl\t0,1,\n"
        )
        result = parse_ucsc_tables(str(kc), str(gb))
        assert len(result) == 0

        return

    def test_single_exon_excluded(self, tmp_path: Path) -> None:
        """T-04c: Single-exon transcripts excluded."""
        kc = tmp_path / "kc.txt"
        kc.write_text("#h\nchr1\t0\t200\t1\tENST_1EX\tENSP\n")
        gb = tmp_path / "gb.txt"
        gb.write_text(
            "#h\n"
            "0\tENST_1EX\tchr1\t+\t0\t200\t"
            "10\t190\t1\t0,\t200,\t0\t"
            "ONEGENE\tcmpl\tcmpl\t0,\n"
        )
        result = parse_ucsc_tables(str(kc), str(gb))
        assert len(result) == 0

        return

    def test_non_standard_chrom_excluded(self, tmp_path: Path) -> None:
        """T-04d: Non-standard chromosomes excluded."""
        kc = tmp_path / "kc.txt"
        kc.write_text("#h\nchrUn\t0\t200\t1\tE1\tP1\n")
        gb = tmp_path / "gb.txt"
        gb.write_text(
            "#h\n"
            "0\tE1\tchrUn\t+\t0\t200\t"
            "10\t190\t2\t0,100,\t90,200,\t0\t"
            "BADCHR\tcmpl\tcmpl\t0,1,\n"
        )
        result = parse_ucsc_tables(str(kc), str(gb))
        assert len(result) == 0

        return


class TestParseGencodeGtf:
    """T-06 ~ T-07: GTF mode parsing."""

    @pytest.fixture()
    def gtf_path(self, tmp_path: Path) -> Path:
        """Create a minimal GENCODE GTF.

        Args:
            tmp_path (Path): pytest tmp dir.

        Returns:
            Path: GTF file path.
        """
        gtf = tmp_path / "test.gtf"
        lines = [
            "##description: test",
            # transcript line
            "chr1\tENSEMBL\ttranscript\t101\t500\t.\t+\t.\t"
            'gene_id "G1"; transcript_id "T1"; '
            'gene_type "protein_coding"; '
            'transcript_type "protein_coding"; '
            'gene_name "GENEA"; '
            'tag "basic"; tag "Ensembl_canonical"; '
            'tag "MANE_Select"; tag "appris_principal_1";',
            # exon 1
            "chr1\tENSEMBL\texon\t101\t200\t.\t+\t.\t"
            'gene_id "G1"; transcript_id "T1"; '
            'gene_type "protein_coding"; '
            'transcript_type "protein_coding"; '
            'gene_name "GENEA";',
            # exon 2
            "chr1\tENSEMBL\texon\t350\t500\t.\t+\t.\t"
            'gene_id "G1"; transcript_id "T1"; '
            'gene_type "protein_coding"; '
            'transcript_type "protein_coding"; '
            'gene_name "GENEA";',
        ]
        gtf.write_text("\n".join(lines) + "\n")
        return gtf

    def test_parses_gene(self, gtf_path: Path) -> None:
        """T-06: GTF parsed to gene structure."""
        genes = parse_gencode_gtf(str(gtf_path))
        assert "GENEA" in genes
        assert "T1" in genes["GENEA"]

        return

    def test_exon_coordinates_0based(self, gtf_path: Path) -> None:
        """T-07: GTF exons stored as 0-based start."""
        genes = parse_gencode_gtf(str(gtf_path))
        tx = genes["GENEA"]["T1"]
        starts = sorted(e[0] for e in tx["exons"])
        assert starts == [100, 349]

        return

    def test_cds_features_with_frame_collected(self, tmp_path: Path) -> None:
        """CDS lines populate the per-transcript ``cds`` list with frames."""
        gtf = tmp_path / "with_cds.gtf"
        common_attrs = (
            'gene_id "G2"; transcript_id "T2"; '
            'gene_type "protein_coding"; '
            'transcript_type "protein_coding"; '
            'gene_name "GENEB";'
        )
        lines = [
            f"chr2\tENSEMBL\ttranscript\t101\t500\t.\t+\t.\t{common_attrs}",
            f"chr2\tENSEMBL\texon\t101\t200\t.\t+\t.\t{common_attrs}",
            f"chr2\tENSEMBL\tCDS\t150\t200\t.\t+\t0\t{common_attrs}",
            f"chr2\tENSEMBL\texon\t350\t500\t.\t+\t.\t{common_attrs}",
            f"chr2\tENSEMBL\tCDS\t350\t450\t.\t+\t2\t{common_attrs}",
        ]
        gtf.write_text("\n".join(lines) + "\n")

        genes = parse_gencode_gtf(str(gtf))
        tx = genes["GENEB"]["T2"]
        cds_sorted = sorted(tx["cds"], key=lambda x: x[0])
        # CDS positions stored as (start_0, end_1based, frame).
        assert cds_sorted == [(149, 200, 0), (349, 450, 2)]

        return

    def test_tx_start_1based(self, gtf_path: Path) -> None:
        """tx_start follows the UCSC mode (1-based) convention."""
        tx = parse_gencode_gtf(str(gtf_path))["GENEA"]["T1"]
        assert tx["tx_start"] == 101
        assert tx["tx_end"] == 500

        return

    def test_tier_tags_read_from_separate_entries(self, gtf_path: Path) -> None:
        """Tier tags on separate GTF entries are all read."""
        tx = parse_gencode_gtf(str(gtf_path))["GENEA"]["T1"]
        assert tx["score"] == (1, 0, 1)

        return

    def test_stop_codon_and_readthrough_collected(self, tmp_path: Path) -> None:
        """stop_codon segments and the readthrough tag are kept."""
        gtf = tmp_path / "stop.gtf"
        attrs = (
            'gene_id "G3"; transcript_id "T3"; gene_type "protein_coding"; '
            'transcript_type "protein_coding"; gene_name "GENEC-GENED";'
        )
        lines = [
            f"chr3\tENSEMBL\ttranscript\t101\t500\t.\t+\t.\t{attrs} "
            'tag "basic"; tag "readthrough_transcript";',
            f"chr3\tENSEMBL\texon\t101\t500\t.\t+\t.\t{attrs}",
            f"chr3\tENSEMBL\tCDS\t150\t197\t.\t+\t0\t{attrs}",
            f"chr3\tENSEMBL\tstop_codon\t198\t200\t.\t+\t0\t{attrs}",
        ]
        gtf.write_text("\n".join(lines) + "\n")
        tx = parse_gencode_gtf(str(gtf))["GENEC-GENED"]["T3"]
        assert tx["stop_codon"] == [(197, 200, 0)]
        assert tx["readthrough"] is True

        return

    def test_keeps_non_protein_coding_gene_types(self, tmp_path: Path) -> None:
        """IG/TR genes reach selection; the CDS requirement applies there."""
        gtf = tmp_path / "ig.gtf"
        attrs = (
            'gene_id "G4"; transcript_id "T4"; gene_type "IG_C_gene"; '
            'transcript_type "IG_C_gene"; gene_name "IGHM";'
        )
        lines = [
            f"chr14\tENSEMBL\ttranscript\t101\t500\t.\t-\t.\t{attrs}",
            f"chr14\tENSEMBL\texon\t101\t200\t.\t-\t.\t{attrs}",
        ]
        gtf.write_text("\n".join(lines) + "\n")
        assert "T4" in parse_gencode_gtf(str(gtf))["IGHM"]

        return


class TestMergeStopCodon:
    """GTF stop codons are folded back into the CDS segments."""

    def test_plus_strand_extends_last_segment(self) -> None:
        """A + strand stop codon extends the last CDS segment."""
        cds = [(120, 180, 0), (350, 477, 2)]
        assert merge_stop_codon(cds, [(477, 480, 0)]) == [(120, 180, 0), (350, 480, 2)]

        return

    def test_minus_strand_extends_first_segment(self) -> None:
        """A - strand stop codon extends the first CDS segment."""
        cds = [(603, 700, 2), (800, 900, 0)]
        assert merge_stop_codon(cds, [(600, 603, 0)]) == [(600, 700, 2), (800, 900, 0)]

        return

    def test_stop_codon_in_its_own_exon(self) -> None:
        """A stop codon past an exon boundary is its own segment."""
        cds = [(120, 180, 0)]
        assert merge_stop_codon(cds, [(300, 303, 0)]) == [(120, 180, 0), (300, 303, 0)]

        return

    def test_split_stop_codon(self) -> None:
        """A stop codon split by an intron extends one segment and adds one."""
        cds = [(120, 179, 0)]
        merged = merge_stop_codon(cds, [(179, 180, 0), (300, 302, 2)])
        assert merged == [(120, 180, 0), (300, 302, 2)]

        return

    def test_no_stop_codon(self) -> None:
        """Without a stop codon the CDS is unchanged."""
        assert merge_stop_codon([(10, 20, 1)], []) == [(10, 20, 1)]

        return


def _gtf_tx(
    chrom: str,
    strand: str,
    gene: str,
    tx_id: str,
    exons: list[tuple[int, int]],
    cds: list[tuple[int, int, int]],
    stop: list[tuple[int, int, int]],
    tags: tuple[str, ...] = ("basic", "Ensembl_canonical", "MANE_Select"),
) -> list[str]:
    """GENCODE-style GTF lines for one transcript (1-based closed coordinates).

    Args:
        chrom (str): Chromosome.
        strand (str): ``'+'`` or ``'-'``.
        gene (str): gene_name (also used as gene_id).
        tx_id (str): transcript_id.
        exons (list[tuple[int, int]]): Exon (start, end) pairs.
        cds (list[tuple[int, int, int]]): CDS (start, end, phase).
        stop (list[tuple[int, int, int]]): stop_codon (start, end, phase).
        tags (tuple[str, ...]): Transcript tags, one entry each.

    Returns:
        list[str]: GTF lines.
    """
    attrs = (
        f'gene_id "{gene}"; transcript_id "{tx_id}"; gene_type "protein_coding"; '
        f'transcript_type "protein_coding"; gene_name "{gene}";'
    )
    tag_str = " ".join(f'tag "{t}";' for t in tags)

    def row(feature: str, s: int, e: int, phase: str, extra: str = "") -> str:
        cols = (chrom, "ENSEMBL", feature, s, e, ".", strand, phase, attrs + extra)
        return "\t".join(str(c) for c in cols)

    start = min(s for s, _ in exons)
    end = max(e for _, e in exons)
    lines = [row("transcript", start, end, ".", f" {tag_str}")]
    lines += [row("exon", s, e, ".") for s, e in exons]
    lines += [row("CDS", s, e, str(p)) for s, e, p in cds]
    lines += [row("stop_codon", s, e, str(p)) for s, e, p in stop]
    return lines


def _select_from_gtf(tmp_path: Path, lines: list[str]) -> list[dict]:
    """Write ``lines`` as a GTF and run GTF mode selection.

    Args:
        tmp_path (Path): pytest tmp dir.
        lines (list[str]): GTF lines.

    Returns:
        list[dict]: Selected transcripts.
    """
    gtf = tmp_path / "sel.gtf"
    gtf.write_text("\n".join(lines) + "\n")
    return select_canonical_transcripts(parse_gencode_gtf(str(gtf)))


class TestSelectCanonicalTranscripts:
    """T-08: Representative transcript selection from GTF."""

    def test_selects_highest_score(self) -> None:
        genes = {
            "GENE1": {
                "TX_MANE": {
                    "chrom": "chr1",
                    "strand": "+",
                    "tx_start": 100,
                    "tx_end": 500,
                    "exons": [(100, 200), (300, 500)],
                    "cds": [(150, 200, 0), (300, 400, 1)],
                    "score": (1, 0, 1),
                    "transcript_type": "protein_coding",
                    "tsl": None,
                },
                "TX_OTHER": {
                    "chrom": "chr1",
                    "strand": "+",
                    "tx_start": 100,
                    "tx_end": 500,
                    "exons": [(100, 250), (300, 500)],
                    "cds": [(150, 250, 0), (300, 400, 1)],
                    "score": (0, 0, 0),
                    "transcript_type": "protein_coding",
                    "tsl": None,
                },
            }
        }
        result = select_canonical_transcripts(genes)
        assert len(result) == 1
        assert result[0]["transcript_id"] == "TX_MANE"

        return

    def test_selects_longest_exonic_length_tiebreak(self) -> None:
        """FR-003: Tiebreak by longest exonic length when scores are equal."""
        genes = {
            "GENE1": {
                "TX_SHORT": {
                    "chrom": "chr1",
                    "strand": "+",
                    "tx_start": 100,
                    "tx_end": 500,
                    "exons": [(100, 200), (400, 500)],  # length = 100 + 100 = 200
                    "cds": [(150, 200, 0), (400, 450, 1)],
                    "score": (0, 0, 1),
                    "transcript_type": "protein_coding",
                    "tsl": None,
                },
                "TX_LONG": {
                    "chrom": "chr1",
                    "strand": "+",
                    "tx_start": 100,
                    "tx_end": 600,
                    "exons": [(100, 200), (300, 600)],  # length = 100 + 300 = 400
                    "cds": [(150, 200, 0), (300, 450, 1)],
                    "score": (0, 0, 1),
                    "transcript_type": "protein_coding",
                    "tsl": None,
                },
            }
        }
        result = select_canonical_transcripts(genes)
        assert len(result) == 1
        assert result[0]["transcript_id"] == "TX_LONG"

        return

    def test_emits_region_columns(self) -> None:
        """Selected transcript carries exon/CDS/frame strings."""
        genes = {
            "GENE1": {
                "TX1": {
                    "chrom": "chr1",
                    "strand": "+",
                    "tx_start": 100,
                    "tx_end": 500,
                    "exons": [(99, 200), (349, 500)],
                    "cds": [(120, 200, 0), (349, 480, 1)],
                    "score": (1, 0, 1),
                    "transcript_type": "protein_coding",
                    "tsl": None,
                },
            }
        }
        result = select_canonical_transcripts(genes)
        assert len(result) == 1
        tx = result[0]
        assert tx["jn_start_str"] == "200,"
        assert tx["jn_end_str"] == "350,"
        assert tx["exon_start_str"] == "100,350,"
        assert tx["exon_end_str"] == "200,500,"
        assert tx["cds_start_str"] == "121,350,"
        assert tx["cds_end_str"] == "200,480,"
        assert tx["cds_frame_str"] == "0,1,"

        return

    def test_excludes_single_exon(self) -> None:
        genes = {
            "GENE1": {
                "TX1": {
                    "chrom": "chr1",
                    "strand": "+",
                    "tx_start": 0,
                    "tx_end": 200,
                    "exons": [(0, 200)],
                    "cds": [(20, 180, 0)],
                    "score": (1, 0, 1),
                    "transcript_type": "protein_coding",
                    "tsl": None,
                }
            }
        }
        result = select_canonical_transcripts(genes)
        assert len(result) == 0

        return

    def test_excludes_gene_without_tier_tag(self, tmp_path: Path) -> None:
        """A gene without any tier tag is excluded."""
        lines = _gtf_tx(
            "chr1", "+", "GENE1", "T1", [(101, 200), (301, 400)],
            [(121, 200, 0), (301, 380, 1)], [], tags=("basic",),
        )
        assert _select_from_gtf(tmp_path, lines) == []

        return

    def test_excludes_readthrough(self, tmp_path: Path) -> None:
        """A readthrough gene is excluded."""
        lines = _gtf_tx(
            "chr1", "+", "GENEA-GENEB", "T1", [(101, 200), (301, 400)],
            [(121, 200, 0), (301, 380, 1)], [],
            tags=("basic", "Ensembl_canonical", "readthrough_transcript"),
        )
        assert _select_from_gtf(tmp_path, lines) == []

        return

    def test_keeps_mane_select_readthrough(self, tmp_path: Path) -> None:
        """A readthrough transcript that is the MANE_Select is kept."""
        lines = _gtf_tx(
            "chr17", "+", "FOXOL", "T1", [(101, 200), (301, 400)],
            [(121, 200, 0), (301, 380, 1)], [],
            tags=(
                "basic", "Ensembl_canonical", "MANE_Select", "readthrough_transcript"
            ),
        )
        assert [r["transcript_id"] for r in _select_from_gtf(tmp_path, lines)] == ["T1"]

        return

    def test_excludes_without_cds(self, tmp_path: Path) -> None:
        """A gene whose selected transcript has no CDS is excluded."""
        lines = _gtf_tx(
            "chr1", "+", "LNC1", "T1", [(101, 200), (301, 400)], [], [],
            tags=("basic", "Ensembl_canonical"),
        )
        assert _select_from_gtf(tmp_path, lines) == []

        return

    def test_par_chrx_mane_beats_chry_copy(self, tmp_path: Path) -> None:
        """A PAR gene keeps the chrX copy carrying MANE_Select."""
        lines = _gtf_tx(
            "chrX", "+", "SHOXL", "TX_X", [(1001, 1100), (1201, 1300)],
            [(1051, 1100, 0), (1201, 1250, 2)], [],
        ) + _gtf_tx(
            "chrY", "+", "SHOXL", "TX_Y", [(1001, 1100), (1151, 1180), (1201, 1300)],
            [(1051, 1100, 0), (1151, 1180, 2), (1201, 1250, 2)], [],
            tags=("basic", "Ensembl_canonical"),
        )
        result = _select_from_gtf(tmp_path, lines)
        assert [(r["transcript_id"], r["chrom"]) for r in result] == [("TX_X", "chrX")]

        return

    def test_readthrough_sharing_a_name_loses_to_mane(self, tmp_path: Path) -> None:
        """A readthrough gene named like a MANE gene does not displace it."""
        lines = _gtf_tx(
            "chr9", "-", "KYATL", "TX_MANE", [(1001, 1100), (1201, 1300)],
            [(1051, 1100, 0), (1201, 1250, 2)], [],
        ) + _gtf_tx(
            "chr9", "-", "KYATL", "TX_RT",
            [(501, 600), (701, 800), (1001, 1100), (1201, 1300)],
            [(551, 600, 0), (701, 800, 2), (1001, 1100, 0), (1201, 1250, 2)], [],
            tags=("basic", "Ensembl_canonical", "readthrough_transcript"),
        )
        result = _select_from_gtf(tmp_path, lines)
        assert [r["transcript_id"] for r in result] == ["TX_MANE"]

        return

    def test_matches_ucsc_mode_columns(self, tmp_path: Path) -> None:
        """The same transcripts give UCSC mode's table columns on both strands."""
        kc = tmp_path / "knownCanonical.txt"
        kc.write_text(
            "#chrom\tchromStart\tchromEnd\tclusterId\ttranscript\tprotein\n"
            "chr1\t100\t500\t1\tENST001\tENSP001\n"
            "chr1\t600\t900\t2\tENST002\tENSP002\n"
        )
        gb = tmp_path / "gencodeBasic.txt"
        gb.write_text(
            "#bin\tname\tchrom\tstrand\ttxStart\ttxEnd\tcdsStart\tcdsEnd\t"
            "exonCount\texonStarts\texonEnds\tscore\tname2\t"
            "cdsStartStat\tcdsEndStat\texonFrames\n"
            "0\tENST001\tchr1\t+\t100\t500\t120\t480\t3\t100,200,350,\t"
            "180,300,500,\t0\tGENEA\tcmpl\tcmpl\t0,0,2,\n"
            "0\tENST002\tchr1\t-\t600\t900\t600\t900\t2\t600,800,\t"
            "700,900,\t0\tGENEB\tcmpl\tcmpl\t2,0,\n"
        )
        ucsc = {t["gene_name"]: t for t in parse_ucsc_tables(str(kc), str(gb))}
        lines = _gtf_tx(
            "chr1", "+", "GENEA", "ENST001", [(101, 180), (201, 300), (351, 500)],
            [(121, 180, 0), (201, 300, 0), (351, 477, 2)], [(478, 480, 0)],
        ) + _gtf_tx(
            "chr1", "-", "GENEB", "ENST002", [(601, 700), (801, 900)],
            [(604, 700, 2), (801, 900, 0)], [(601, 603, 0)],
        )
        gtf = {t["gene_name"]: t for t in _select_from_gtf(tmp_path, lines)}
        keys = (
            "chrom", "strand", "tx_start", "tx_end", "jn_start_str", "jn_end_str",
            "exon_start_str", "exon_end_str", "cds_start_str", "cds_end_str",
            "cds_frame_str",
        )
        for gene in ("GENEA", "GENEB"):
            assert {k: gtf[gene][k] for k in keys} == {k: ucsc[gene][k] for k in keys}

        return

    def test_tsl_filter(self) -> None:
        genes = {
            "GENE1": {
                "TX_GOOD": {
                    "chrom": "chr1",
                    "strand": "+",
                    "tx_start": 0,
                    "tx_end": 200,
                    "exons": [(0, 100), (150, 200)],
                    "cds": [(10, 100, 0), (150, 190, 0)],
                    "score": (0, 0, 1),
                    "transcript_type": "protein_coding",
                    "tsl": 1,
                },
                "TX_BAD": {
                    "chrom": "chr1",
                    "strand": "+",
                    "tx_start": 0,
                    "tx_end": 300,
                    "exons": [(0, 100), (200, 300)],
                    "cds": [(10, 100, 0), (200, 290, 0)],
                    "score": (1, 0, 1),
                    "transcript_type": "protein_coding",
                    "tsl": 5,
                },
            }
        }
        result = select_canonical_transcripts(genes, tsl_max=2)
        assert len(result) == 1
        assert result[0]["transcript_id"] == "TX_GOOD"

        return


class TestWriteCanonicalDataset:
    """T-09: Splice table output format."""

    def test_output_columns(self, tmp_path: Path) -> None:
        transcripts = [
            {
                "gene_name": "GENEA",
                "chrom": "chr1",
                "strand": "+",
                "tx_start": 101,
                "tx_end": 500,
                "jn_start_str": "200,",
                "jn_end_str": "351,",
                "exon_start_str": "101,351,",
                "exon_end_str": "200,500,",
                "cds_start_str": "120,351,",
                "cds_end_str": "200,480,",
                "cds_frame_str": "0,1,",
            }
        ]
        out = tmp_path / "canonical.txt"
        write_canonical_dataset(transcripts, str(out), test_chroms={"chr1"})
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 1
        cols = lines[0].split("\t")
        assert len(cols) == 13
        assert cols[0] == "GENEA"
        assert cols[2] == "chr1"
        assert cols[3] == "+"
        assert cols[4] == "101"
        assert cols[5] == "500"
        assert cols[8] == "101,351,"
        assert cols[9] == "200,500,"
        assert cols[10] == "120,351,"
        assert cols[11] == "200,480,"
        assert cols[12] == "0,1,"

        return

    def test_paralog_flag_from_test_chroms(self, tmp_path: Path) -> None:
        transcripts = [
            {
                "gene_name": "G1",
                "chrom": "chr1",
                "strand": "+",
                "tx_start": 1,
                "tx_end": 100,
                "jn_start_str": "50,",
                "jn_end_str": "60,",
            },
            {
                "gene_name": "G2",
                "chrom": "chr2",
                "strand": "+",
                "tx_start": 1,
                "tx_end": 100,
                "jn_start_str": "50,",
                "jn_end_str": "60,",
            },
        ]
        out = tmp_path / "canonical.txt"
        write_canonical_dataset(
            transcripts,
            str(out),
            test_chroms={"chr1"},
        )
        lines = out.read_text().strip().split("\n")
        assert lines[0].split("\t")[1] == "0"
        assert lines[1].split("\t")[1] == "1"

        return

    def test_paralog_flag_from_dict(self, tmp_path: Path) -> None:
        transcripts = [
            {
                "gene_name": "GENEA",
                "chrom": "chr1",
                "strand": "+",
                "tx_start": 1,
                "tx_end": 100,
                "jn_start_str": "50,",
                "jn_end_str": "60,",
            }
        ]
        out = tmp_path / "canonical.txt"
        write_canonical_dataset(
            transcripts,
            str(out),
            paralog_flags={"GENEA": 1},
        )
        lines = out.read_text().strip().split("\n")
        assert lines[0].split("\t")[1] == "1"

        return


class TestLoadParalogFlags:
    """T-09b: Paralog flag loading."""

    def test_load(self, tmp_path: Path) -> None:
        tsv = tmp_path / "paralogs.tsv"
        tsv.write_text("gene_name\tparalog_flag\nGENEA\t0\nGENEB\t1\n")
        flags = load_paralog_flags(str(tsv))
        assert flags == {"GENEA": 0, "GENEB": 1}

        return


# ════════════════════════════════════════════════════
#  Stage 2 — h5_datafile_builder
# ════════════════════════════════════════════════════


class TestExtractSequence:
    """T-10 ~ T-13: Genome sequence extraction."""

    @pytest.fixture()
    def genome(self) -> MockFasta:
        """50 bp mock genome.

        Returns:
            MockFasta: Mock genome dict.
        """
        return MockFasta({"chr1": "A" * 50, "chr2": "C" * 30})

    def test_normal_extraction(self, genome: MockFasta) -> None:
        """T-10: Normal range returns correct length."""
        seq = extract_sequence(genome, "chr1", 10, 20)
        assert len(seq) == 10
        assert seq == "A" * 10

        return

    def test_left_boundary_padding(self, genome: MockFasta) -> None:
        """T-11: Negative start pads with N."""
        seq = extract_sequence(genome, "chr1", -5, 10)
        assert len(seq) == 15
        assert seq[:5] == "NNNNN"
        assert seq[5:] == "A" * 10

        return

    def test_right_boundary_padding(self, genome: MockFasta) -> None:
        """T-12: Past-end request pads with N."""
        seq = extract_sequence(genome, "chr2", 25, 35)
        assert len(seq) == 10
        assert seq[:5] == "C" * 5
        assert seq[5:] == "NNNNN"

        return

    def test_both_boundaries(self, genome: MockFasta) -> None:
        """T-13: Both sides overflow pads correctly."""
        seq = extract_sequence(genome, "chr2", -3, 33)
        assert len(seq) == 36
        assert seq[:3] == "NNN"
        assert seq[-3:] == "NNN"

        return


class TestBuildDatafile:
    """T-14 ~ T-17: Intermediate H5 datafile creation."""

    @pytest.fixture()
    def datafile_env(self, tmp_path: Path) -> tuple[str, str]:
        """Set up splice table, FASTA, and config.

        Args:
            tmp_path (Path): pytest tmp dir.

        Returns:
            tuple[str, str]: (config_path, output_path).
        """
        genome_seq = "ACGTACGTACGT" * 10
        fa_path = _write_fasta(
            tmp_path / "ref.fa",
            {"chr1": genome_seq},
        )

        splice_table = tmp_path / "canonical.txt"
        tx_start = 11
        tx_end = 30
        splice_table.write_text(f"GENE1\t0\tchr1\t+\t{tx_start}\t{tx_end}\t20,\t25,\n")

        out_path = str(tmp_path / "datafile.h5")

        cfg = OmegaConf.create(
            {
                "cl_max": CL_MAX,
                "sl": SL,
                "train_chromosomes": ["chr1"],
                "test_chromosomes": ["chr3"],
                "paths": {
                    "ref_genome": str(fa_path),
                    "splice_table": str(splice_table),
                    "output_dir": str(tmp_path),
                },
            }
        )
        cfg_path = str(tmp_path / "cfg.yaml")
        OmegaConf.save(cfg, cfg_path)

        return cfg_path, out_path

    def test_creates_h5(self, datafile_env: tuple[str, str]) -> None:
        """T-14: H5 file is created on disk."""
        cfg_path, out_path = datafile_env
        cfg = OmegaConf.load(cfg_path)
        result = build_datafile(cfg, "train", "all", out_path)
        assert Path(result).exists()

        return

    def test_real_flank_reads_sl_more_each_side(
        self, datafile_env: tuple[str, str]
    ) -> None:
        """flank_fill=real reads ``sl`` extra bases per side and records it."""
        cfg_path, out_path = datafile_env
        cfg = OmegaConf.load(cfg_path)
        build_datafile(cfg, "train", "all", out_path)
        with h5py.File(out_path, "r") as f:
            n_flank_len = len(f["SEQ"][0])
            assert f.attrs["flank_fill"] == "N"
            assert int(f.attrs["extra_flank"]) == 0

        cfg.flank_fill = "real"
        build_datafile(cfg, "train", "all", out_path)
        with h5py.File(out_path, "r") as f:
            assert len(f["SEQ"][0]) == n_flank_len + 2 * SL
            assert f.attrs["flank_fill"] == "real"
            assert int(f.attrs["extra_flank"]) == SL

        return

    def test_h5_datasets_present(self, datafile_env: tuple[str, str]) -> None:
        """T-15: Required datasets exist in H5."""
        cfg_path, out_path = datafile_env
        cfg = OmegaConf.load(cfg_path)
        build_datafile(cfg, "train", "all", out_path)
        with h5py.File(out_path, "r") as f:
            for key in [
                "NAME",
                "CHROM",
                "STRAND",
                "TX_START",
                "TX_END",
                "JN_START",
                "JN_END",
                "SEQ",
            ]:
                assert key in f, f"Missing dataset: {key}"

        return

    def test_split_filtering(self, datafile_env: tuple[str, str]) -> None:
        """T-16: Split filter excludes chroms."""
        cfg_path, out_path = datafile_env
        cfg = OmegaConf.load(cfg_path)
        build_datafile(cfg, "test", "all", out_path)
        with h5py.File(out_path, "r") as f:
            assert f["NAME"].shape[0] == 0

        return

    def test_paralog_filtering(self, tmp_path: Path) -> None:
        """T-17: Paralog filter keeps matching rows."""
        genome_seq = "ACGT" * 30
        fa_path = _write_fasta(tmp_path / "ref.fa", {"chr1": genome_seq})

        splice = tmp_path / "spl.txt"
        splice.write_text(
            "G1\t0\tchr1\t+\t11\t30\t20,\t25,\nG2\t1\tchr1\t+\t11\t30\t20,\t25,\n"
        )

        cfg = OmegaConf.create(
            {
                "cl_max": CL_MAX,
                "sl": SL,
                "train_chromosomes": ["chr1"],
                "test_chromosomes": [],
                "paths": {
                    "ref_genome": str(fa_path),
                    "splice_table": str(splice),
                    "output_dir": str(tmp_path),
                },
            }
        )

        out0 = str(tmp_path / "df_0.h5")
        build_datafile(cfg, "train", "0", out0)
        with h5py.File(out0, "r") as f:
            assert f["NAME"].shape[0] == 1

        out1 = str(tmp_path / "df_1.h5")
        build_datafile(cfg, "train", "1", out1)
        with h5py.File(out1, "r") as f:
            assert f["NAME"].shape[0] == 1

        return


# ════════════════════════════════════════════════════
#  Stage 3 — h5_utils + h5_dataset_builder
# ════════════════════════════════════════════════════


class TestCeilDiv:
    """T-18: Ceiling division utility."""

    @pytest.mark.parametrize(
        "x,y,expected",
        [
            (10, 5, 2),
            (11, 5, 3),
            (0, 5, 0),
            (1, 5, 1),
            (100, 7, 15),
        ],
    )
    def test_ceil_div(self, x: int, y: int, expected: int) -> None:
        assert ceil_div(x, y) == expected

        return


class TestReformatData:
    """T-19 ~ T-21: Block splitting logic."""

    def test_output_shapes(self) -> None:
        """T-19: X has shape (N, SL+CL_max), Y (N, SL)."""
        tx_len = 50
        X0 = np.ones(tx_len + CL_MAX)
        Y0 = np.zeros(tx_len)
        Xd, Yd, _, _ = reformat_data(X0, Y0, CL_MAX, SL)
        n = ceil(tx_len / SL)
        assert Xd.shape == (n, SL + CL_MAX)
        assert Yd.shape == (n, SL)

        return

    def test_single_block_padding(self) -> None:
        """T-20: Short gene pads Y with -1."""
        tx_len = 10
        X0 = np.ones(tx_len + CL_MAX)
        Y0 = np.zeros(tx_len)
        Xd, Yd, _, _ = reformat_data(X0, Y0, CL_MAX, SL)
        assert Xd.shape[0] == 1
        assert np.sum(Yd[0] == -1) == SL - tx_len

        return

    def test_multi_block_count(self) -> None:
        """T-21: Correct number of blocks."""
        tx_len = SL * 3 + 5
        X0 = np.ones(tx_len + CL_MAX)
        Y0 = np.zeros(tx_len)
        Xd, Yd, _, _ = reformat_data(X0, Y0, CL_MAX, SL)
        assert Xd.shape[0] == ceil(tx_len / SL)

        return

    def test_stride_default_matches_sl(self) -> None:
        """Stride defaulting to ``sl`` is bit-identical to legacy."""
        tx_len = SL * 3 + 5
        X0 = np.arange(tx_len + CL_MAX, dtype=np.float32)
        Y0 = np.arange(tx_len, dtype=np.float32)
        Xd_legacy, Yd_legacy, _, _ = reformat_data(X0, Y0, CL_MAX, SL)
        Xd_explicit, Yd_explicit, _, _ = reformat_data(
            X0, Y0, CL_MAX, SL, stride=SL
        )
        np.testing.assert_array_equal(Xd_legacy, Xd_explicit)
        np.testing.assert_array_equal(Yd_legacy, Yd_explicit)

        return

    def test_stride_overlap_block_count(self) -> None:
        """``stride < sl`` yields ceil_div(tx-sl, stride)+1 blocks."""
        tx_len = SL * 3 + 5
        stride = SL // 2
        X0 = np.zeros(tx_len + CL_MAX)
        Y0 = np.zeros(tx_len)
        Xd, Yd, _, _ = reformat_data(
            X0, Y0, CL_MAX, SL, stride=stride
        )
        expected = ceil((tx_len - SL) / stride) + 1
        assert Xd.shape[0] == expected
        assert Yd.shape[0] == expected

        return

    def test_stride_window_starts(self) -> None:
        """Block ``i`` reads from ``stride*i`` and labels stay aligned."""
        tx_len = SL * 4
        stride = SL // 2
        X0 = np.arange(tx_len + CL_MAX, dtype=np.float32)
        Y0 = np.arange(tx_len, dtype=np.float32)
        Xd, Yd, _, _ = reformat_data(
            X0, Y0, CL_MAX, SL, stride=stride
        )
        for i in range(Xd.shape[0]):
            start = stride * i
            assert Xd[i, 0] == start
            if start < tx_len:
                assert Yd[i, 0] == start

        return

    def test_stride_full_position_coverage(self) -> None:
        """Every transcript position appears in some block's labels.

        This is the key property the variant-centred coverage
        guarantee relies on: with ``stride <= sl - 2W`` every
        position ``v`` has a block where ``[v-W, v+W]`` lies in
        the target region.
        """
        tx_len = SL * 5 + 3
        stride = SL // 2
        X0 = np.zeros(tx_len + CL_MAX)
        Y0 = np.arange(1, tx_len + 1, dtype=np.float32)
        _, Yd, _, _ = reformat_data(
            X0, Y0, CL_MAX, SL, stride=stride
        )
        seen: set[int] = set()
        for row in Yd:
            for value in row:
                if value > 0:
                    seen.add(int(value))
        assert seen == set(range(1, tx_len + 1))

        return

    def test_stride_invalid_value_raises(self) -> None:
        """Stride outside ``[1, sl]`` is rejected."""
        X0 = np.zeros(SL + CL_MAX)
        Y0 = np.zeros(SL)
        with pytest.raises(ValueError):
            reformat_data(X0, Y0, CL_MAX, SL, stride=0)
        with pytest.raises(ValueError):
            reformat_data(X0, Y0, CL_MAX, SL, stride=SL + 1)

        return


class TestCreateDatapoints:
    """T-22 ~ T-28: Full datapoint creation."""

    def _make_seq(self, length: int) -> str:
        """Build a repeating ACGT sequence.

        Args:
            length (int): Desired length.

        Returns:
            str: DNA sequence.
        """
        base = "ACGT"
        return (base * (length // 4 + 1))[:length]

    def test_encoding_values(self) -> None:
        """T-22: Sequence encoding 0=N,1=A,2=C,3=G,4=T."""
        tx_start, tx_end = 1, SL
        tx_len = tx_end - tx_start + 1
        seq_len = tx_len + CL_MAX + 1
        seq = self._make_seq(seq_len)

        X, Y, _, _ = create_datapoints(
            seq,
            "+",
            str(tx_start),
            str(tx_end),
            [""],
            [""],
            CL_MAX,
            SL,
        )
        assert set(np.unique(X)).issubset({0, 1, 2, 3, 4})

        return

    def test_exact_encoding_mapping(self) -> None:
        """FR-014: Exact 1:1 mapping N=0, A=1, C=2, G=3, T=4."""
        tx_start, tx_end = 1, SL
        tx_len = tx_end - tx_start + 1
        half_cl = CL_MAX // 2
        seq = "A" * half_cl + "NACGT" + "A" * (tx_len + CL_MAX + 1 - 5 - half_cl)

        X, Y, _, _ = create_datapoints(
            seq,
            "+",
            str(tx_start),
            str(tx_end),
            [""],
            [""],
            CL_MAX,
            SL,
        )
        assert list(X[0, half_cl : half_cl + 5]) == [
            0,
            1,
            2,
            3,
            4,
        ]

        return

    def test_label_values(self) -> None:
        """T-23: Labels are in {-1, 0, 1, 2}."""
        tx_start, tx_end = 1, SL
        tx_len = tx_end - tx_start + 1
        seq = self._make_seq(tx_len + CL_MAX + 1)

        jn_s = f"{tx_start + 5},"
        jn_e = f"{tx_start + 10},"

        X, Y, _, _ = create_datapoints(
            seq,
            "+",
            str(tx_start),
            str(tx_end),
            [jn_s],
            [jn_e],
            CL_MAX,
            SL,
        )
        assert set(np.unique(Y)).issubset({-1, 0, 1, 2})

        return

    def test_donor_acceptor_positions_plus(self) -> None:
        """T-24: + strand donor=2, acceptor=1 placement."""
        tx_s, tx_e = 1, 40
        tx_len = tx_e - tx_s + 1
        seq = self._make_seq(tx_len + CL_MAX + 1)

        donor_pos = 10
        acceptor_pos = 20

        X, Y, _, _ = create_datapoints(
            seq,
            "+",
            str(tx_s),
            str(tx_e),
            [f"{donor_pos},"],
            [f"{acceptor_pos},"],
            CL_MAX,
            SL,
        )
        Y_flat = Y.flatten()[:tx_len]
        assert Y_flat[donor_pos - tx_s] == 2
        assert Y_flat[acceptor_pos - tx_s] == 1

        return

    def test_minus_strand_reverse_complement(self) -> None:
        """T-25: - strand reverses and complements seq."""
        tx_s, tx_e = 1, 20
        tx_len = tx_e - tx_s + 1
        seq = "A" * (tx_len + CL_MAX + 1)

        X, Y, _, _ = create_datapoints(
            seq,
            "-",
            str(tx_s),
            str(tx_e),
            [""],
            [""],
            CL_MAX,
            SL,
        )
        mid = CL_MAX // 2
        inner = X[0, mid : mid + tx_len]
        assert np.all(inner == 4)

        return

    def test_minus_strand_label_reversal(self) -> None:
        """T-26: - strand reverses donor/acceptor coords."""
        tx_s, tx_e = 1, 40
        tx_len = tx_e - tx_s + 1
        seq = self._make_seq(tx_len + CL_MAX + 1)

        donor_pos = 10
        acceptor_pos = 30

        X, Y, _, _ = create_datapoints(
            seq,
            "-",
            str(tx_s),
            str(tx_e),
            [f"{donor_pos},"],
            [f"{acceptor_pos},"],
            CL_MAX,
            SL,
        )
        Y_flat = Y.flatten()[:tx_len]
        assert Y_flat[tx_e - acceptor_pos] == 2
        assert Y_flat[tx_e - donor_pos] == 1

        return

    def test_non_acgtn_replaced(self) -> None:
        """T-27: Non-ACGTN chars become N (0)."""
        tx_s, tx_e = 1, 20
        tx_len = tx_e - tx_s + 1
        seq = "R" * (tx_len + CL_MAX + 1)

        X, Y, _, _ = create_datapoints(
            seq,
            "+",
            str(tx_s),
            str(tx_e),
            [""],
            [""],
            CL_MAX,
            SL,
        )
        assert np.all(X == 0)

        return

    def test_output_dtype_int8(self) -> None:
        """T-28: Output arrays are int8."""
        tx_s, tx_e = 1, 20
        tx_len = tx_e - tx_s + 1
        seq = self._make_seq(tx_len + CL_MAX + 1)
        X, Y, _, _ = create_datapoints(
            seq,
            "+",
            str(tx_s),
            str(tx_e),
            [""],
            [""],
            CL_MAX,
            SL,
        )
        assert X.dtype == np.int8
        assert Y.dtype == np.int8

        return

    def test_no_junctions_all_padding(self) -> None:
        """T-28b: Empty junctions -> all -1 labels."""
        tx_s, tx_e = 1, 20
        tx_len = tx_e - tx_s + 1
        seq = self._make_seq(tx_len + CL_MAX + 1)
        X, Y, _, _ = create_datapoints(
            seq,
            "+",
            str(tx_s),
            str(tx_e),
            [""],
            [""],
            CL_MAX,
            SL,
        )
        assert np.all(Y == -1)

        return


class TestCreateDatapointsRealFlank:
    """``flank_fill='real'`` keeps genome sequence outside the transcript."""

    _COMPLEMENT = str.maketrans("ACGT", "TGCA")

    @staticmethod
    def _encode(seq: str) -> np.ndarray:
        """Encode with the builder's alphabet (N=0, A=1, C=2, G=3, T=4)."""
        return np.array(["NACGT".index(b) for b in seq], dtype=np.int8)

    @staticmethod
    def _genome(length: int) -> str:
        """Random N-free sequence, so any 0 in X can only be padding."""
        rng = np.random.default_rng(0)
        return "".join(rng.choice(list("ACGT"), size=length))

    def test_flanks_keep_genome_sequence(self) -> None:
        """Real fill keeps the flanks; the transcript bases are unchanged."""
        tx_s, tx_e = 1, SL
        tx_len = tx_e - tx_s + 1
        half_cl = CL_MAX // 2
        seq = self._genome(tx_len + CL_MAX)
        args = (seq, "+", str(tx_s), str(tx_e), [""], [""], CL_MAX, SL)

        X_n, _, _, _ = create_datapoints(*args)
        X_r, _, _, _ = create_datapoints(*args, flank_fill="real")

        assert np.all(X_n[0, :half_cl] == 0)
        assert np.all(X_n[0, -half_cl:] == 0)
        np.testing.assert_array_equal(X_r[0], self._encode(seq))
        np.testing.assert_array_equal(
            X_r[0, half_cl:-half_cl], X_n[0, half_cl:-half_cl]
        )

        return

    def test_labels_do_not_depend_on_flank_fill(self) -> None:
        """Y, Y_region and Y_exon are identical under either fill."""
        tx_s, tx_e = 1, 3 * SL
        tx_len = tx_e - tx_s + 1
        seq = self._genome(tx_len + CL_MAX)
        args = (seq, "-", str(tx_s), str(tx_e), ["10,40,"], ["20,50,"], CL_MAX, SL)

        _, Y_n, R_n, E_n = create_datapoints(*args, stride=SL // 2)
        _, Y_r, R_r, E_r = create_datapoints(*args, stride=SL // 2, flank_fill="real")

        np.testing.assert_array_equal(Y_n, Y_r)
        np.testing.assert_array_equal(R_n, R_r)
        np.testing.assert_array_equal(E_n, E_r)

        return

    def test_without_extra_flank_the_last_tail_is_zero_padded(self) -> None:
        """The last block reaches stride - 1 bases past the flank."""
        stride = SL // 2
        tx_s, tx_e = 1, 1 + SL  # tx_len = SL + 1: two blocks
        tx_len = tx_e - tx_s + 1
        seq = self._genome(tx_len + CL_MAX)

        X, _, _, _ = create_datapoints(
            seq, "+", str(tx_s), str(tx_e), [""], [""], CL_MAX, SL,
            stride=stride, flank_fill="real",
        )

        assert X.shape[0] == 2
        assert np.all(X[-1, -(stride - 1):] == 0)
        assert np.all(X[-1, : -(stride - 1)] != 0)

        return

    @pytest.mark.parametrize("strand", ["+", "-"])
    def test_extra_flank_fills_the_last_tail(self, strand: str) -> None:
        """With ``extra_flank`` every block, the last included, is genome
        sequence in transcript orientation."""
        stride = SL // 2
        tx_s, tx_e = 1, 1 + SL
        tx_len = tx_e - tx_s + 1
        # What the datafile reads with extra_flank=SL: SL more on each side.
        genome = self._genome(tx_len + CL_MAX + 2 * SL)

        X, _, _, _ = create_datapoints(
            genome, strand, str(tx_s), str(tx_e), [""], [""], CL_MAX, SL,
            stride=stride, flank_fill="real", extra_flank=SL,
        )

        # Transcript orientation, built independently of create_datapoints.
        if strand == "+":
            oriented = genome[SL:]
        else:
            oriented = genome[:-SL].translate(self._COMPLEMENT)[::-1]
        for i in range(X.shape[0]):
            start = stride * i
            expected = self._encode(oriented[start : start + SL + CL_MAX])
            np.testing.assert_array_equal(X[i], expected)
        assert np.all(X != 0)

        return

    def test_extra_flank_requires_real_fill(self) -> None:
        """An extension is meaningless when the flanks are blanked."""
        seq = self._genome(SL + CL_MAX + 2 * SL)
        with pytest.raises(ValueError, match="extra_flank"):
            create_datapoints(
                seq, "+", "1", str(SL), [""], [""], CL_MAX, SL, extra_flank=SL
            )

        return

    def test_unknown_flank_fill_is_rejected(self) -> None:
        """Only 'N' and 'real' are accepted."""
        seq = self._genome(SL + CL_MAX)
        with pytest.raises(ValueError, match="flank_fill"):
            create_datapoints(
                seq, "+", "1", str(SL), [""], [""], CL_MAX, SL, flank_fill="genome"
            )

        return

    def test_built_windows_match_the_genome(self, tmp_path: Path) -> None:
        """Datafile + dataset builders with real flanks reproduce every window
        the verifier rebuilds from the genome, on both strands."""
        from pyfaidx import Fasta

        stride = SL // 2
        genome_seq = self._genome(400)
        fa_path = _write_fasta(tmp_path / "ref.fa", {"chr1": genome_seq})
        genes = [("GP", "+", 50, 120), ("GM", "-", 200, 331)]
        table = tmp_path / "canonical.txt"
        table.write_text(
            "".join(f"{n}\t0\tchr1\t{s}\t{a}\t{b}\t{a + 5},\t{a + 15},\n" for n, s, a, b in genes)
        )
        cfg = OmegaConf.create(
            {
                "cl_max": CL_MAX,
                "sl": SL,
                "stride": stride,
                "chunk_size": 1,
                "flank_fill": "real",
                "train_chromosomes": ["chr1"],
                "test_chromosomes": ["chr3"],
                "paths": {
                    "ref_genome": str(fa_path),
                    "splice_table": str(table),
                    "output_dir": str(tmp_path),
                },
            }
        )
        datafile = build_datafile(cfg, "train", "all", str(tmp_path / "datafile.h5"))
        dataset = build_dataset(cfg, "train", "all", datafile, str(tmp_path / "dataset.h5"))

        genome = Fasta(str(fa_path))
        with h5py.File(dataset, "r") as f:
            for i, (_, strand, tx_s, tx_e) in enumerate(genes):
                X = f[f"X{i}"][:]
                assert X.shape[0] == _n_points(tx_e - tx_s + 1, SL, stride)
                for w in range(X.shape[0]):
                    expected = _expected_window(
                        genome, "chr1", strand, tx_s, tx_e, w, CL_MAX // 2, SL, stride
                    )
                    np.testing.assert_array_equal(X[w], expected)

        return


class TestBuildRegionLabelArray:
    """Region annotation label generation."""

    def test_plus_strand_protein_coding_layout(self) -> None:
        """+strand: 5'UTR / CDS phases / 3'UTR / intron in correct ranges."""
        labels = build_region_label_array(
            tx_start=1,
            tx_end=50,
            exon_starts_1=[1, 25, 45],
            exon_ends_1=[15, 40, 50],
            cds_starts_1=[8, 25, 45],
            cds_ends_1=[15, 40, 46],
            cds_frames=[0, 1, 0],
            strand="+",
        )
        assert labels.shape == (50,)
        for idx in range(0, 7):
            assert labels[idx] == REGION_UTR5
        for k in range(8):
            assert labels[7 + k] == REGION_CDS_PHASE0 + ((k + 3) % 3)
        for idx in range(15, 24):
            assert labels[idx] == REGION_INTRON
        for k in range(16):
            assert labels[24 + k] == REGION_CDS_PHASE0 + ((k + 2) % 3)
        for idx in range(40, 44):
            assert labels[idx] == REGION_INTRON
        for k in range(2):
            assert labels[44 + k] == REGION_CDS_PHASE0 + ((k + 3) % 3)
        for idx in range(46, 50):
            assert labels[idx] == REGION_UTR3

        return

    def test_minus_strand_utr_sides_flip(self) -> None:
        """-strand: low genomic pos = 3'UTR; high = 5'UTR."""
        labels = build_region_label_array(
            tx_start=1,
            tx_end=50,
            exon_starts_1=[1, 25, 45],
            exon_ends_1=[15, 40, 50],
            cds_starts_1=[8, 25, 45],
            cds_ends_1=[15, 40, 46],
            cds_frames=[0, 1, 0],
            strand="-",
        )
        for idx in range(0, 7):
            assert labels[idx] == REGION_UTR3
        for idx in range(46, 50):
            assert labels[idx] == REGION_UTR5

        return

    def test_minus_strand_codon_phase_progression(self) -> None:
        """-strand: 5' end of a CDS span starts at codon position 0."""
        # Single CDS span on - strand: span (8, 15), frame=0.
        labels = build_region_label_array(
            tx_start=1,
            tx_end=20,
            exon_starts_1=[1, 8],
            exon_ends_1=[5, 20],
            cds_starts_1=[8],
            cds_ends_1=[15],
            cds_frames=[0],
            strand="-",
        )
        # genomic asc indexing (pos - tx_start). pos 15 = idx 14 is the
        # 5'-most base of the CDS span on - strand and must be phase 0.
        assert labels[14] == REGION_CDS_PHASE0
        assert labels[13] == REGION_CDS_PHASE0 + 1
        assert labels[12] == REGION_CDS_PHASE0 + 2
        assert labels[11] == REGION_CDS_PHASE0
        assert labels[7] == REGION_CDS_PHASE0 + 1

        return

    def test_non_coding_transcript(self) -> None:
        """Non-coding: every exon base = ncRNA_exon, between = intron."""
        labels = build_region_label_array(
            tx_start=1,
            tx_end=20,
            exon_starts_1=[1, 10],
            exon_ends_1=[5, 20],
            cds_starts_1=[],
            cds_ends_1=[],
            cds_frames=[],
            strand="+",
        )
        for idx in range(0, 5):
            assert labels[idx] == REGION_NCRNA_EXON
        for idx in range(5, 9):
            assert labels[idx] == REGION_INTRON
        for idx in range(9, 20):
            assert labels[idx] == REGION_NCRNA_EXON

        return


class TestBuildExonPositionLabelArray:
    """Per-base exon position label generation."""

    def test_plus_strand_three_exons(self) -> None:
        labels = build_exon_position_label_array(
            tx_start=1,
            tx_end=30,
            exon_starts_1=[1, 11, 21],
            exon_ends_1=[5, 15, 30],
            strand="+",
        )
        assert np.all(labels[0:5] == EXON_FIRST)
        assert np.all(labels[5:10] == EXON_IGNORE)
        assert np.all(labels[10:15] == EXON_INTERNAL)
        assert np.all(labels[15:20] == EXON_IGNORE)
        assert np.all(labels[20:30] == EXON_LAST)

        return

    def test_minus_strand_flips_first_last(self) -> None:
        labels = build_exon_position_label_array(
            tx_start=1,
            tx_end=30,
            exon_starts_1=[1, 11, 21],
            exon_ends_1=[5, 15, 30],
            strand="-",
        )
        assert np.all(labels[0:5] == EXON_LAST)
        assert np.all(labels[10:15] == EXON_INTERNAL)
        assert np.all(labels[20:30] == EXON_FIRST)

        return

    def test_single_exon_transcript_all_ignore(self) -> None:
        labels = build_exon_position_label_array(
            tx_start=1,
            tx_end=10,
            exon_starts_1=[1],
            exon_ends_1=[10],
            strand="+",
        )
        assert np.all(labels == EXON_IGNORE)

        return


class TestComputeUcscCdsSpansAndFrames:
    """UCSC mode CDS span and GFF3 phase derivation."""

    def test_plus_strand_two_cds_exons(self) -> None:
        spans, frames = compute_ucsc_cds_spans_and_frames(
            exons_0=[(0, 10), (20, 35)],
            cds_start_0=5,
            cds_end=30,
            strand="+",
        )
        assert spans == [(6, 10), (21, 30)]
        # First CDS exon length = 5; (3 - 5 % 3) % 3 = 1.
        assert frames == [0, 1]

        return

    def test_minus_strand_two_cds_exons(self) -> None:
        spans, frames = compute_ucsc_cds_spans_and_frames(
            exons_0=[(0, 10), (20, 35)],
            cds_start_0=5,
            cds_end=30,
            strand="-",
        )
        # 5'->3' order on - strand starts at the genomically last span.
        # Span (21, 30) length 10 -> next phase = (3 - 10 % 3) % 3 = 2.
        assert spans == [(6, 10), (21, 30)]
        assert frames == [2, 0]

        return

    def test_no_cds_when_cds_start_equals_cds_end(self) -> None:
        spans, frames = compute_ucsc_cds_spans_and_frames(
            exons_0=[(0, 10)],
            cds_start_0=10,
            cds_end=10,
            strand="+",
        )
        assert spans == []
        assert frames == []

        return


class TestCreateDatapointsRegions:
    """End-to-end region/exon labels through create_datapoints."""

    def test_plus_strand_region_and_exon_labels(self) -> None:
        tx_s, tx_e = 1, 30
        tx_len = tx_e - tx_s + 1
        seq = "A" * (tx_len + CL_MAX + 1)
        X, Y, Y_region, Y_exon = create_datapoints(
            seq,
            "+",
            str(tx_s),
            str(tx_e),
            ["10,"],
            ["20,"],
            CL_MAX,
            SL,
            exon_start="1,20,",
            exon_end="10,30,",
            cds_start="3,20,",
            cds_end="10,25,",
            cds_frame="0,1,",
        )
        assert Y_region.shape == Y.shape
        assert Y_exon.shape == Y.shape
        # 5'UTR
        assert Y_region[0, 0] == REGION_UTR5
        assert Y_region[0, 1] == REGION_UTR5
        # CDS phase 0 at genomic pos 3 (idx 2)
        assert Y_region[0, 2] == REGION_CDS_PHASE0
        # First exon
        assert np.all(Y_exon[0, 0:10] == EXON_FIRST)
        # Intron region
        assert np.all(Y_exon[0, 10:19] == EXON_IGNORE)
        # Last exon
        assert np.all(Y_exon[0, 19:30] == EXON_LAST)

        return

    def test_minus_strand_region_labels_reversed(self) -> None:
        tx_s, tx_e = 1, SL
        tx_len = tx_e - tx_s + 1
        seq = "A" * (tx_len + CL_MAX + 1)
        # Two exons (1-5) and (15-20) on - strand: in transcript
        # 5'->3' order, exon (15-20) comes first.
        X, Y, Y_region, Y_exon = create_datapoints(
            seq,
            "-",
            str(tx_s),
            str(tx_e),
            [""],
            [""],
            CL_MAX,
            SL,
            exon_start="1,15,",
            exon_end="5,20,",
            cds_start=",",
            cds_end=",",
            cds_frame=",",
        )
        # Block index 0 corresponds to transcript 5' end (genomic 20 on
        # - strand): the high genomic exon should now carry the FIRST
        # label and the low genomic exon should carry LAST.
        assert Y_exon[0, 0] == EXON_FIRST
        assert Y_exon[0, tx_len - 1] == EXON_LAST

        return

    def test_default_region_inputs_yield_ignore_labels(self) -> None:
        tx_s, tx_e = 1, 20
        tx_len = tx_e - tx_s + 1
        seq = "A" * (tx_len + CL_MAX + 1)
        X, Y, Y_region, Y_exon = create_datapoints(
            seq,
            "+",
            str(tx_s),
            str(tx_e),
            [""],
            [""],
            CL_MAX,
            SL,
        )
        assert Y_region.shape == Y.shape
        assert Y_exon.shape == Y.shape
        assert np.all(Y_region == REGION_IGNORE)
        assert np.all(Y_exon == EXON_IGNORE)

        return


class TestClipDatapoints:
    """T-29 ~ T-31: Clip + GPU alignment."""

    def test_no_clip_when_equal(self) -> None:
        """T-29: cl == cl_max -> no clipping."""
        X = np.zeros((10, SL + CL_MAX))
        Y = np.zeros((10, SL))
        Xc, Yc = clip_datapoints(X, Y, CL_MAX, CL_MAX, 1)
        assert Xc.shape == X.shape
        assert Yc.shape == Y.shape

        return

    def test_clip_when_smaller(self) -> None:
        """T-30: cl < cl_max trims X columns."""
        cl = 6
        X = np.zeros((10, SL + CL_MAX))
        Y = np.zeros((10, SL))
        Xc, Yc = clip_datapoints(X, Y, cl, CL_MAX, 1)
        assert Xc.shape[1] == SL + cl

        return

    def test_gpu_alignment(self) -> None:
        """T-31: Rows trimmed for GPU count."""
        X = np.zeros((7, SL + CL_MAX))
        Y = np.zeros((7, SL))
        Xc, Yc = clip_datapoints(X, Y, CL_MAX, CL_MAX, 3)
        assert Xc.shape[0] % 3 == 0
        assert Xc.shape[0] == 6

        return


class TestBuildDataset:
    """T-32 ~ T-33: Final H5 dataset creation."""

    @pytest.fixture()
    def datafile_h5(self, tmp_path: Path) -> tuple[str, str]:
        """Create a minimal intermediate H5.

        Args:
            tmp_path (Path): pytest tmp dir.

        Returns:
            tuple[str, str]: (input_h5_path, cfg_path).
        """
        tx_s, tx_e = 1, 30
        tx_len = tx_e - tx_s + 1
        seq = ("ACGT" * ((tx_len + CL_MAX + 2) // 4 + 1))[: tx_len + CL_MAX + 1]

        inp = tmp_path / "datafile.h5"
        with h5py.File(str(inp), "w") as f:
            f.create_dataset(
                "NAME",
                data=np.array(["GENE1"], dtype="S"),
            )
            f.create_dataset("PARALOG", data=np.array([0]))
            f.create_dataset(
                "CHROM",
                data=np.array(["chr1"], dtype="S"),
            )
            f.create_dataset(
                "STRAND",
                data=np.array(["+"], dtype="S"),
            )
            f.create_dataset(
                "TX_START",
                data=np.array([str(tx_s)], dtype="S"),
            )
            f.create_dataset(
                "TX_END",
                data=np.array([str(tx_e)], dtype="S"),
            )
            f.create_dataset(
                "JN_START",
                data=np.array([["10,"]], dtype="S"),
            )
            f.create_dataset(
                "JN_END",
                data=np.array([["20,"]], dtype="S"),
            )
            f.create_dataset(
                "SEQ",
                data=np.array([seq], dtype="S"),
            )

        cfg = OmegaConf.create(
            {
                "cl_max": CL_MAX,
                "sl": SL,
                "chunk_size": 1,
                "paths": {
                    "output_dir": str(tmp_path),
                },
            }
        )
        cfg_path = str(tmp_path / "cfg.yaml")
        OmegaConf.save(cfg, cfg_path)

        return str(inp), cfg_path

    def test_creates_chunks(self, datafile_h5: tuple[str, str]) -> None:
        """T-32: Output contains X0/Y0 datasets."""
        inp, cfg_path = datafile_h5
        cfg = OmegaConf.load(cfg_path)
        out = str(Path(inp).parent / "dataset.h5")
        build_dataset(cfg, "train", "all", inp, out)

        with h5py.File(out, "r") as f:
            assert "X0" in f
            assert "Y0" in f

        return

    def test_chunk_shapes(self, datafile_h5: tuple[str, str]) -> None:
        """T-33: X/Y chunk dimensions are correct."""
        inp, cfg_path = datafile_h5
        cfg = OmegaConf.load(cfg_path)
        out = str(Path(inp).parent / "dataset.h5")
        build_dataset(cfg, "train", "all", inp, out)

        with h5py.File(out, "r") as f:
            assert f["X0"].shape[1] == SL + CL_MAX
            assert f["Y0"].shape[1] == SL

        return

    def test_flank_fill_mismatch_with_the_datafile_is_rejected(
        self, datafile_h5: tuple[str, str]
    ) -> None:
        """A real-flank config cannot read an N-flank datafile (no attrs)."""
        inp, cfg_path = datafile_h5
        cfg = OmegaConf.load(cfg_path)
        cfg.flank_fill = "real"
        out = str(Path(inp).parent / "dataset.h5")
        with pytest.raises(ValueError, match="flank_fill"):
            build_dataset(cfg, "train", "all", inp, out)

        return


# ════════════════════════════════════════════════════
#  Dataset Loader — SplicingH5Dataset
# ════════════════════════════════════════════════════


def _make_h5_dataset(
    path: Path,
    n_chunks: int = 2,
    n_per_chunk: int = 5,
    sl: int = SL,
    cl_max: int = CL_MAX,
    integer_format: bool = True,
) -> Path:
    """Write a synthetic H5 dataset file.

    Args:
        path (Path): Output path.
        n_chunks (int): Number of chunks.
        n_per_chunk (int): Samples per chunk.
        sl (int): Sequence length.
        cl_max (int): Context length.
        integer_format (bool): Use integer format.

    Returns:
        Path: The written file path.
    """
    with h5py.File(str(path), "w") as f:
        for i in range(n_chunks):
            if integer_format:
                X = np.random.randint(
                    0,
                    5,
                    size=(n_per_chunk, sl + cl_max),
                    dtype=np.int8,
                )
                Y = np.random.choice(
                    [-1, 0, 1, 2],
                    size=(n_per_chunk, sl),
                ).astype(np.int8)
            else:
                X = np.zeros(
                    (n_per_chunk, sl + cl_max, 4),
                    dtype=np.int8,
                )
                for r in range(n_per_chunk):
                    for c in range(sl + cl_max):
                        ch = np.random.randint(0, 4)
                        X[r, c, ch] = 1
                Y = np.zeros(
                    (1, n_per_chunk, sl, 3),
                    dtype=np.int8,
                )
                for r in range(n_per_chunk):
                    Y[0, r, :, 0] = 1
            f.create_dataset(f"X{i}", data=X)
            f.create_dataset(f"Y{i}", data=Y)
    return path


class TestSplicingH5DatasetInteger:
    """T-34 ~ T-37: Integer format loader."""

    @pytest.fixture()
    def ds(self, tmp_path: Path) -> SplicingH5Dataset:
        """Build mock integer-format H5 dataset.

        Args:
            tmp_path (Path): pytest tmp dir.

        Returns:
            SplicingH5Dataset: Dataset instance.
        """
        path = _make_h5_dataset(
            tmp_path / "ds.h5",
            n_chunks=2,
            n_per_chunk=5,
        )
        return SplicingH5Dataset(str(path))

    def test_length(self, ds: SplicingH5Dataset) -> None:
        """T-34: Length == sum of chunk sizes."""
        assert len(ds) == 10

        return

    def test_sample_keys(self, ds: SplicingH5Dataset) -> None:
        """T-35: Each sample has 'ref' and 'labels'."""
        sample = ds[0]
        assert "ref" in sample
        assert "labels" in sample

        return

    def test_ref_is_dna(self, ds: SplicingH5Dataset) -> None:
        """T-36: 'ref' is a string of {N,A,C,G,T}."""
        sample = ds[0]
        assert isinstance(sample["ref"], str)
        assert set(sample["ref"]).issubset({"N", "A", "C", "G", "T"})

        return

    def test_labels_shape(self, ds: SplicingH5Dataset) -> None:
        """T-37: Labels shape is (3, SL)."""
        sample = ds[0]
        assert sample["labels"].shape == (3, SL)

        return

    def test_cross_chunk_indexing(self, ds: SplicingH5Dataset) -> None:
        """T-37b: Index in second chunk works."""
        sample = ds[7]
        assert "ref" in sample
        assert sample["labels"].shape == (3, SL)

        return


class TestSplicingH5DatasetOneHot:
    """T-38 ~ T-39: Legacy one-hot format loader."""

    @pytest.fixture()
    def ds(self, tmp_path: Path) -> SplicingH5Dataset:
        """Build mock one-hot-format H5 dataset.

        Args:
            tmp_path (Path): pytest tmp dir.

        Returns:
            SplicingH5Dataset: Dataset instance.
        """
        path = _make_h5_dataset(
            tmp_path / "oh.h5",
            n_chunks=1,
            n_per_chunk=3,
            integer_format=False,
        )
        return SplicingH5Dataset(str(path))

    def test_length(self, ds: SplicingH5Dataset) -> None:
        """T-38: Length correct for one-hot."""
        assert len(ds) == 3

        return

    def test_ref_is_dna(self, ds: SplicingH5Dataset) -> None:
        """T-39: One-hot decoded to DNA string."""
        sample = ds[0]
        assert isinstance(sample["ref"], str)
        assert len(sample["ref"]) == SL + CL_MAX

        return

    def test_labels_shape(self, ds: SplicingH5Dataset) -> None:
        """T-39b: Labels shape correct for one-hot."""
        sample = ds[0]
        assert sample["labels"].shape == (3, SL)

        return


class TestSplicingH5DatasetLabelMapping:
    """T-40 ~ T-41: Label conversion correctness."""

    def test_integer_label_channels(self, tmp_path: Path) -> None:
        """T-40: Integer labels map to 3-ch tensor."""
        path = tmp_path / "lab.h5"
        with h5py.File(str(path), "w") as f:
            X = np.array(
                [[1, 2, 3, 4] * ((SL + CL_MAX) // 4)],
                dtype=np.int8,
            )
            Y = np.array(
                [[0, 1, 2, -1] + [0] * (SL - 4)],
                dtype=np.int8,
            )
            f.create_dataset("X0", data=X)
            f.create_dataset("Y0", data=Y)

        ds = SplicingH5Dataset(str(path))
        labels = ds[0]["labels"]
        acc, don, nei = labels[0], labels[1], labels[2]

        assert acc[1] == 1
        assert don[2] == 1
        assert nei[0] == 1

        assert acc[3] == 0
        assert don[3] == 0
        assert nei[3] == 0

        return

    def test_padding_excluded_from_channels(self, tmp_path: Path) -> None:
        """T-41: Padding (-1) zeroes all channels."""
        path = tmp_path / "pad.h5"
        with h5py.File(str(path), "w") as f:
            X = np.ones((1, SL + CL_MAX), dtype=np.int8)
            Y = -np.ones((1, SL), dtype=np.int8)
            f.create_dataset("X0", data=X)
            f.create_dataset("Y0", data=Y)

        ds = SplicingH5Dataset(str(path))
        labels = ds[0]["labels"]
        assert (labels.sum(dim=0) == 0).all()

        return

    def test_site_usage_soft_labels(self, tmp_path: Path) -> None:
        """Canonical stays hard while a measured added site becomes soft."""
        path = tmp_path / "site_usage.h5"
        with h5py.File(str(path), "w") as f:
            X = np.ones((1, SL + CL_MAX), dtype=np.int8)
            Y = np.zeros((1, SL), dtype=np.int8)
            Y[0, :2] = [1, 2]
            usage = np.full((1, SL), -1.0, dtype=np.float16)
            usage[0, :2] = [1.0, 0.25]
            f.create_dataset("X0", data=X)
            f.create_dataset("Y0", data=Y)
            f.create_dataset("Y_SITE_USAGE0", data=usage)

        ds = SplicingH5Dataset(str(path), site_usage_soft_labels=True)
        labels = ds[0]["labels"]

        assert labels.dtype == torch.float32
        torch.testing.assert_close(labels[:, 0], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(labels[:, 1], torch.tensor([0.0, 0.25, 0.75]))

        return


# ════════════════════════════════════════════════════
#  Additional Edge-Case Tests (Gap Coverage)
# ════════════════════════════════════════════════════


class TestEmptyDatafile:
    """T-46: build_dataset with 0 genes."""

    def test_empty_datafile_returns_path(self, tmp_path: Path) -> None:
        """Empty intermediate H5 -> returns path, no file."""
        inp = tmp_path / "empty.h5"
        with h5py.File(str(inp), "w") as f:
            for ds_name in [
                "NAME",
                "CHROM",
                "STRAND",
                "TX_START",
                "TX_END",
                "SEQ",
            ]:
                f.create_dataset(
                    ds_name,
                    data=np.array([], dtype="S"),
                )
            f.create_dataset("PARALOG", data=np.array([], dtype=int))
            f.create_dataset(
                "JN_START",
                data=np.empty((0, 1), dtype="S"),
            )
            f.create_dataset(
                "JN_END",
                data=np.empty((0, 1), dtype="S"),
            )

        cfg = OmegaConf.create(
            {
                "cl_max": CL_MAX,
                "sl": SL,
                "chunk_size": 100,
                "paths": {
                    "output_dir": str(tmp_path),
                },
            }
        )
        out = str(tmp_path / "dataset.h5")
        result = build_dataset(cfg, "train", "all", str(inp), out)
        assert result == out
        assert not Path(result).exists()

        return


class TestClMaxOddRejected:
    """T-47: cl_max must be even."""

    def test_odd_cl_max_raises(self, tmp_path: Path) -> None:
        """Odd cl_max triggers assertion error."""
        genome_seq = "ACGT" * 30
        fa_path = _write_fasta(tmp_path / "ref.fa", {"chr1": genome_seq})
        splice = tmp_path / "spl.txt"
        splice.write_text("G1\t0\tchr1\t+\t11\t30\t20,\t25,\n")
        cfg = OmegaConf.create(
            {
                "cl_max": 11,
                "sl": SL,
                "train_chromosomes": ["chr1"],
                "test_chromosomes": [],
                "paths": {
                    "ref_genome": str(fa_path),
                    "splice_table": str(splice),
                    "output_dir": str(tmp_path),
                },
            }
        )
        with pytest.raises(AssertionError):
            build_datafile(
                cfg,
                "train",
                "all",
                str(tmp_path / "df.h5"),
            )

        return


class TestChunkSizeLargerThanGenes:
    """T-48: chunk_size > n_genes edge case."""

    def test_chunk_size_exceeds_gene_count(self, tmp_path: Path) -> None:
        """Verify dataset builds when chunk_size > n_genes.

        When chunk_size exceeds n_genes, the last (only) chunk
        should contain exactly n_genes datapoints without error.
        """
        tx_s, tx_e = 1, 30
        tx_len = tx_e - tx_s + 1
        seq = ("ACGT" * ((tx_len + CL_MAX + 2) // 4 + 1))[: tx_len + CL_MAX + 1]

        inp = tmp_path / "datafile.h5"
        with h5py.File(str(inp), "w") as f:
            f.create_dataset(
                "NAME",
                data=np.array(["GENE1"], dtype="S"),
            )
            f.create_dataset("PARALOG", data=np.array([0]))
            f.create_dataset(
                "CHROM",
                data=np.array(["chr1"], dtype="S"),
            )
            f.create_dataset(
                "STRAND",
                data=np.array(["+"], dtype="S"),
            )
            f.create_dataset(
                "TX_START",
                data=np.array([str(tx_s)], dtype="S"),
            )
            f.create_dataset(
                "TX_END",
                data=np.array([str(tx_e)], dtype="S"),
            )
            f.create_dataset(
                "JN_START",
                data=np.array([["10,"]], dtype="S"),
            )
            f.create_dataset(
                "JN_END",
                data=np.array([["20,"]], dtype="S"),
            )
            f.create_dataset(
                "SEQ",
                data=np.array([seq], dtype="S"),
            )

        cfg = OmegaConf.create(
            {
                "cl_max": CL_MAX,
                "sl": SL,
                "chunk_size": 100,
                "paths": {
                    "output_dir": str(tmp_path),
                },
            }
        )
        out = str(tmp_path / "dataset.h5")
        result = build_dataset(cfg, "train", "all", str(inp), out)

        assert Path(result).exists()
        with h5py.File(result, "r") as f:
            assert "X0" in f
            assert "Y0" in f
            assert f["X0"].shape[0] > 0

        return


class TestParalogFlagMissing:
    """T-49: Missing gene defaults to paralog_flag=1."""

    def test_missing_gene_defaults_to_one(self, tmp_path: Path) -> None:
        """Gene absent from paralog_flags -> flag=1."""
        transcripts = [
            {
                "gene_name": "KNOWN",
                "chrom": "chr1",
                "strand": "+",
                "tx_start": 1,
                "tx_end": 100,
                "jn_start_str": "50,",
                "jn_end_str": "60,",
            },
            {
                "gene_name": "UNKNOWN",
                "chrom": "chr2",
                "strand": "+",
                "tx_start": 1,
                "tx_end": 100,
                "jn_start_str": "50,",
                "jn_end_str": "60,",
            },
        ]
        out = tmp_path / "canonical.txt"
        write_canonical_dataset(
            transcripts,
            str(out),
            paralog_flags={"KNOWN": 0},
        )
        lines = out.read_text().strip().split("\n")
        assert lines[0].split("\t")[1] == "0"
        assert lines[1].split("\t")[1] == "1"

        return


class TestUcscGeneDeduplicate:
    """T-50: UCSC mode picks best transcript per gene."""

    def test_dedup_keeps_most_exons(self, tmp_path: Path) -> None:
        """Two transcripts for same gene -> most exons wins."""
        kc = tmp_path / "kc.txt"
        kc.write_text(
            "#h\nchr1\t100\t500\t1\tTX_SHORT\tP\nchr1\t100\t600\t2\tTX_LONG\tP\n"
        )
        gb = tmp_path / "gb.txt"
        gb.write_text(
            "#h\n"
            "0\tTX_SHORT\tchr1\t+\t100\t500\t"
            "120\t480\t2\t100,300,\t"
            "200,500,\t0\tSAMEGENE\tcmpl\t"
            "cmpl\t0,1,\n"
            "0\tTX_LONG\tchr1\t+\t100\t600\t"
            "120\t580\t4\t100,200,350,450,\t"
            "180,300,430,600,\t0\tSAMEGENE\tcmpl\t"
            "cmpl\t0,1,2,3,\n"
        )
        result = parse_ucsc_tables(str(kc), str(gb))
        assert len(result) == 1
        jn_count = result[0]["jn_start_str"].count(",")
        assert jn_count == 3

        return

    def test_dedup_tiebreak_by_length(self, tmp_path: Path) -> None:
        """Same exon count -> longest exonic length wins."""
        kc = tmp_path / "kc.txt"
        kc.write_text("#h\nchr1\t0\t300\t1\tTX_A\tP\nchr1\t0\t400\t2\tTX_B\tP\n")
        gb = tmp_path / "gb.txt"
        gb.write_text(
            "#h\n"
            "0\tTX_A\tchr1\t+\t0\t300\t"
            "10\t290\t2\t0,200,\t"
            "100,300,\t0\tTIEGENE\tcmpl\t"
            "cmpl\t0,1,\n"
            "0\tTX_B\tchr1\t+\t0\t400\t"
            "10\t390\t2\t0,200,\t"
            "150,400,\t0\tTIEGENE\tcmpl\t"
            "cmpl\t0,1,\n"
        )
        result = parse_ucsc_tables(str(kc), str(gb))
        assert len(result) == 1
        assert result[0]["tx_end"] == 400

        return


class TestLevelFilter:
    """T-51: GENCODE annotation level filtering."""

    def _make_genes_with_levels(
        self,
    ) -> dict[str, dict[str, dict]]:
        """Build gene dict with mixed level values.

        Returns:
            dict[str, dict[str, dict]]: Gene dict with
                level 1, 2, 3 and None transcripts.
        """
        base = {
            "chrom": "chr1",
            "strand": "+",
            "tx_start": 0,
            "tx_end": 200,
            "exons": [(0, 100), (150, 200)],
            "cds": [(10, 100, 0), (150, 190, 0)],
            "score": (0, 0, 1),
            "transcript_type": "protein_coding",
            "tsl": 1,
        }
        return {
            "GENE_L1": {
                "TX1": {**base, "level": 1},
            },
            "GENE_L2": {
                "TX2": {**base, "level": 2},
            },
            "GENE_L3": {
                "TX3": {**base, "level": 3},
            },
            "GENE_NA": {
                "TX4": {**base, "level": None},
            },
        }

    def test_gtf_mode_level_filter(self) -> None:
        """FR-005c: level_max=2 excludes level 3."""
        genes = self._make_genes_with_levels()
        result = select_canonical_transcripts(genes, level_max=2)
        names = {r["gene_name"] for r in result}
        assert "GENE_L1" in names
        assert "GENE_L2" in names
        assert "GENE_L3" not in names

        return

    def test_level_na_excluded(self) -> None:
        """FR-005e: N/A level transcripts excluded."""
        genes = self._make_genes_with_levels()
        result = select_canonical_transcripts(genes, level_max=2)
        names = {r["gene_name"] for r in result}
        assert "GENE_NA" not in names

        return

    def test_level_max_null_no_filter(self) -> None:
        """FR-005f: level_max=None keeps all."""
        genes = self._make_genes_with_levels()
        result = select_canonical_transcripts(genes, level_max=None)
        names = {r["gene_name"] for r in result}
        assert len(names) == 4

        return

    def test_level_max_1_strictest(self) -> None:
        """FR-005c: level_max=1 keeps only validated."""
        genes = self._make_genes_with_levels()
        result = select_canonical_transcripts(genes, level_max=1)
        names = {r["gene_name"] for r in result}
        assert names == {"GENE_L1"}

        return

    def test_combined_tsl_level_filter(self) -> None:
        """FR-005h: AND logic for tsl + level filters."""
        genes = {
            "GENE_PASS": {
                "TX1": {
                    "chrom": "chr1",
                    "strand": "+",
                    "tx_start": 0,
                    "tx_end": 200,
                    "exons": [(0, 100), (150, 200)],
                    "cds": [(10, 100, 0), (150, 190, 0)],
                    "score": (0, 0, 1),
                    "transcript_type": "protein_coding",
                    "tsl": 1,
                    "level": 2,
                },
            },
            "GENE_BAD_TSL": {
                "TX2": {
                    "chrom": "chr1",
                    "strand": "+",
                    "tx_start": 0,
                    "tx_end": 200,
                    "exons": [(0, 100), (150, 200)],
                    "cds": [(10, 100, 0), (150, 190, 0)],
                    "score": (0, 0, 1),
                    "transcript_type": "protein_coding",
                    "tsl": 5,
                    "level": 1,
                },
            },
            "GENE_BAD_LVL": {
                "TX3": {
                    "chrom": "chr1",
                    "strand": "+",
                    "tx_start": 0,
                    "tx_end": 200,
                    "exons": [(0, 100), (150, 200)],
                    "cds": [(10, 100, 0), (150, 190, 0)],
                    "score": (0, 0, 1),
                    "transcript_type": "protein_coding",
                    "tsl": 1,
                    "level": 3,
                },
            },
        }
        result = select_canonical_transcripts(genes, tsl_max=2, level_max=2)
        names = {r["gene_name"] for r in result}
        assert names == {"GENE_PASS"}

        return

    def test_parse_gtf_level_attribute(self, tmp_path: Path) -> None:
        """FR-005a: level attribute parsed from GTF."""
        gtf = tmp_path / "level.gtf"
        lines = [
            "##description: test",
            (
                "chr1\tENSEMBL\ttranscript\t"
                "101\t500\t.\t+\t.\t"
                'gene_id "G1"; '
                'transcript_id "T1"; '
                'gene_type "protein_coding"; '
                'transcript_type "protein_coding"; '
                'gene_name "GENEA"; '
                "level 2;"
            ),
            (
                "chr1\tENSEMBL\texon\t"
                "101\t200\t.\t+\t.\t"
                'gene_id "G1"; '
                'transcript_id "T1"; '
                'gene_type "protein_coding"; '
                'transcript_type "protein_coding"; '
                'gene_name "GENEA";'
            ),
            (
                "chr1\tENSEMBL\texon\t"
                "350\t500\t.\t+\t.\t"
                'gene_id "G1"; '
                'transcript_id "T1"; '
                'gene_type "protein_coding"; '
                'transcript_type "protein_coding"; '
                'gene_name "GENEA";'
            ),
        ]
        gtf.write_text("\n".join(lines) + "\n")
        genes = parse_gencode_gtf(str(gtf))
        tx = genes["GENEA"]["T1"]
        assert tx["level"] == 2

        return

    def test_load_level_from_gtf(self, tmp_path: Path) -> None:
        """FR-005d: load_level_from_gtf cross-ref."""
        gtf = tmp_path / "level_xref.gtf"
        lines = [
            "##description: test",
            (
                "chr1\tENSEMBL\ttranscript\t"
                "101\t500\t.\t+\t.\t"
                'gene_id "G1"; '
                'transcript_id "ENST001"; '
                'gene_type "protein_coding"; '
                'transcript_type "protein_coding"; '
                'gene_name "GENEA"; '
                "level 1;"
            ),
            (
                "chr1\tENSEMBL\ttranscript\t"
                "600\t900\t.\t-\t.\t"
                'gene_id "G2"; '
                'transcript_id "ENST002"; '
                'gene_type "protein_coding"; '
                'transcript_type "protein_coding"; '
                'gene_name "GENEB"; '
                "level 3;"
            ),
        ]
        gtf.write_text("\n".join(lines) + "\n")
        level_map = load_level_from_gtf(str(gtf))
        assert level_map["ENST001"] == 1
        assert level_map["ENST002"] == 3

        return

    def test_ucsc_mode_level_filter(self, tmp_path: Path) -> None:
        """FR-005d: UCSC mode cross-ref level filter."""
        kc = tmp_path / "kc.txt"
        kc.write_text(
            "#h\nchr1\t100\t500\t1\tENST001\tP\nchr1\t600\t900\t2\tENST002\tP\n"
        )
        gb = tmp_path / "gb.txt"
        gb.write_text(
            "#h\n"
            "0\tENST001\tchr1\t+\t100\t500\t"
            "120\t480\t3\t100,200,350,\t"
            "180,300,500,\t0\tGENEA\tcmpl\t"
            "cmpl\t0,1,2,\n"
            "0\tENST002\tchr1\t-\t600\t900\t"
            "600\t900\t2\t600,800,\t"
            "700,900,\t0\tGENEB\tcmpl\t"
            "cmpl\t0,1,\n"
        )
        level_map = {"ENST001": 2, "ENST002": 3}
        result = parse_ucsc_tables(
            str(kc),
            str(gb),
            level_map=level_map,
            level_max=2,
        )
        names = {r["gene_name"] for r in result}
        assert "GENEA" in names
        assert "GENEB" not in names

        return

    def test_ucsc_mode_level_na_excluded(self, tmp_path: Path) -> None:
        """FR-005e: UCSC N/A level transcripts excluded."""
        kc = tmp_path / "kc.txt"
        kc.write_text(
            "#h\nchr1\t100\t500\t1\tENST001\tP\nchr1\t600\t900\t2\tENST_MISSING\tP\n"
        )
        gb = tmp_path / "gb.txt"
        gb.write_text(
            "#h\n"
            "0\tENST001\tchr1\t+\t100\t500\t"
            "120\t480\t3\t100,200,350,\t"
            "180,300,500,\t0\tGENEA\tcmpl\t"
            "cmpl\t0,1,2,\n"
            "0\tENST_MISSING\tchr1\t-\t600\t900\t"
            "600\t900\t2\t600,800,\t"
            "700,900,\t0\tGENEB\tcmpl\t"
            "cmpl\t0,1,\n"
        )
        level_map = {"ENST001": 2}
        result = parse_ucsc_tables(
            str(kc),
            str(gb),
            level_map=level_map,
            level_max=2,
        )
        names = {r["gene_name"] for r in result}
        assert "GENEA" in names
        assert "GENEB" not in names

        return

    def test_level_max_out_of_range(self) -> None:
        """Edge: level_max=0 or 4 rejected."""
        from process_input.canonical_dataset_builder import (
            build_canonical_dataset,
        )

        cfg = OmegaConf.create(
            {
                "canonical_dataset": {
                    "mode": "gtf",
                    "level_max": 0,
                },
                "paths": {"gtf": "/dev/null"},
                "logging": {
                    "level": "INFO",
                    "format": "%(message)s",
                    "date_format": "%H:%M:%S",
                },
                "test_chromosomes": ["chr1"],
            }
        )
        with pytest.raises(ValueError, match="1-3"):
            build_canonical_dataset(cfg)

        return

    def test_level_filter_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        """FR-005g: Level filter logs summary."""
        import logging

        genes = self._make_genes_with_levels()
        with caplog.at_level(logging.INFO):
            select_canonical_transcripts(genes, level_max=2)
        log_text = caplog.text
        assert "Level filter" in log_text
        assert "kept=" in log_text
        assert "dropped=" in log_text
        assert "N/A=" in log_text

        return


class TestConfigInterpolation:
    """FR-024: OmegaConf variable interpolation."""

    def test_omegaconf_interpolation(self) -> None:
        """Config supports ${} interpolation."""
        cfg = OmegaConf.create(
            {
                "base_dir": "/tmp/data",
                "paths": {
                    "ref_genome": "${base_dir}/ref.fa",
                    "output_dir": "${base_dir}/out",
                },
            }
        )
        # Resolve interpolation and verify
        assert cfg.paths.ref_genome == "/tmp/data/ref.fa"
        assert cfg.paths.output_dir == "/tmp/data/out"

        return


# ════════════════════════════════════════════════════
#  E2E Integration Tests
# ════════════════════════════════════════════════════


@pytest.mark.slow
class TestEndToEnd:
    """T-42 ~ T-45: Full pipeline integration."""

    @pytest.fixture()
    def e2e_env(self, tmp_path: Path) -> dict[str, str]:
        """Build a tiny end-to-end environment.

        Creates a 200 bp genome on chr1 and chr2 with
        known exon boundaries, splice table, and config.

        Args:
            tmp_path (Path): pytest tmp dir.

        Returns:
            dict[str, str]: Paths to all artifacts.
        """
        chr1_seq = "ACGT" * 50
        chr2_seq = "TGCA" * 50
        fa_path = _write_fasta(
            tmp_path / "ref.fa",
            {"chr1": chr1_seq, "chr2": chr2_seq},
        )

        splice = tmp_path / "canonical.txt"
        splice.write_text(
            "GENE_P\t0\tchr1\t+\t11\t50"
            "\t25,40,\t30,45,\n"
            "GENE_M\t1\tchr2\t-\t11\t50"
            "\t25,40,\t30,45,\n"
        )

        cfg = OmegaConf.create(
            {
                "cl_max": CL_MAX,
                "sl": SL,
                "chunk_size": 1,
                "train_chromosomes": ["chr1", "chr2"],
                "test_chromosomes": [],
                "paths": {
                    "ref_genome": str(fa_path),
                    "splice_table": str(splice),
                    "output_dir": str(tmp_path),
                },
            }
        )
        cfg_path = str(tmp_path / "cfg.yaml")
        OmegaConf.save(cfg, cfg_path)

        return {
            "cfg_path": cfg_path,
            "tmp_path": str(tmp_path),
        }

    def test_pipeline_produces_loadable_dataset(self, e2e_env: dict[str, str]) -> None:
        """T-42: Full pipeline -> loadable dataset."""
        cfg = OmegaConf.load(e2e_env["cfg_path"])
        tmp = e2e_env["tmp_path"]

        df_path = build_datafile(
            cfg,
            "train",
            "all",
            str(Path(tmp) / "datafile.h5"),
        )
        ds_path = build_dataset(
            cfg,
            "train",
            "all",
            df_path,
            str(Path(tmp) / "dataset.h5"),
        )

        ds = SplicingH5Dataset(ds_path)
        assert len(ds) > 0
        sample = ds[0]
        assert "ref" in sample
        assert "labels" in sample

        return

    def test_e2e_label_values_valid(self, e2e_env: dict[str, str]) -> None:
        """T-43: All label values in {0, 1}."""
        cfg = OmegaConf.load(e2e_env["cfg_path"])
        tmp = e2e_env["tmp_path"]

        df_path = build_datafile(
            cfg,
            "train",
            "all",
            str(Path(tmp) / "datafile.h5"),
        )
        ds_path = build_dataset(
            cfg,
            "train",
            "all",
            df_path,
            str(Path(tmp) / "dataset.h5"),
        )

        ds = SplicingH5Dataset(ds_path)
        for i in range(len(ds)):
            labels = ds[i]["labels"]
            assert labels.min() >= 0
            assert labels.max() <= 1

        return

    def test_e2e_ref_length(self, e2e_env: dict[str, str]) -> None:
        """T-44: Ref string length == SL + CL_MAX."""
        cfg = OmegaConf.load(e2e_env["cfg_path"])
        tmp = e2e_env["tmp_path"]

        df_path = build_datafile(
            cfg,
            "train",
            "all",
            str(Path(tmp) / "datafile.h5"),
        )
        ds_path = build_dataset(
            cfg,
            "train",
            "all",
            df_path,
            str(Path(tmp) / "dataset.h5"),
        )

        ds = SplicingH5Dataset(ds_path)
        for i in range(len(ds)):
            assert len(ds[i]["ref"]) == SL + CL_MAX

        return

    def test_e2e_both_strands_have_donors(self, e2e_env: dict[str, str]) -> None:
        """T-45: Both strands produce donor labels."""
        cfg = OmegaConf.load(e2e_env["cfg_path"])
        tmp = e2e_env["tmp_path"]

        df_path = build_datafile(
            cfg,
            "train",
            "all",
            str(Path(tmp) / "datafile.h5"),
        )
        ds_path = build_dataset(
            cfg,
            "train",
            "all",
            df_path,
            str(Path(tmp) / "dataset.h5"),
        )

        with h5py.File(ds_path, "r") as f:
            Y = f["Y0"][:]
        has_donor = np.any(Y == 2)
        has_acceptor = np.any(Y == 1)
        assert has_donor, "No donor labels found"
        assert has_acceptor, "No acceptor labels found"

        return

    def test_e2e_region_columns_propagate_to_h5(self, tmp_path: Path) -> None:
        """13-column splice table -> Y_REGION/Y_EXON datasets in H5."""
        chr1_seq = "ACGT" * 50
        fa_path = _write_fasta(tmp_path / "ref.fa", {"chr1": chr1_seq})

        # Plus-strand transcript covering positions 11-50 with two
        # exons (11-30) and (40-50); CDS spans 15-30 (frame 0) and
        # 40-45 (frame computed by the splice-table builder).
        splice = tmp_path / "canonical.txt"
        splice.write_text(
            "GENE_P\t0\tchr1\t+\t11\t50\t30,\t40,\t"
            "11,40,\t30,50,\t15,40,\t30,45,\t0,1,\n"
        )

        cfg = OmegaConf.create(
            {
                "cl_max": CL_MAX,
                "sl": SL,
                "chunk_size": 1,
                "train_chromosomes": ["chr1"],
                "test_chromosomes": [],
                "paths": {
                    "ref_genome": str(fa_path),
                    "splice_table": str(splice),
                    "output_dir": str(tmp_path),
                },
            }
        )

        df_path = build_datafile(
            cfg, "train", "all", str(tmp_path / "datafile.h5")
        )
        ds_path = build_dataset(
            cfg, "train", "all", df_path, str(tmp_path / "dataset.h5")
        )

        with h5py.File(ds_path, "r") as f:
            assert "Y_REGION0" in f
            assert "Y_EXON0" in f
            y_region = f["Y_REGION0"][:]
            y_exon = f["Y_EXON0"][:]
            y = f["Y0"][:]

        assert y_region.shape == y.shape
        assert y_exon.shape == y.shape

        region_values = set(np.unique(y_region).tolist())
        # 5'UTR, CDS phases, 3'UTR, intron must all appear inside the
        # transcript window; padding (-1) is allowed at trailing bases.
        for required in (
            REGION_INTRON,
            REGION_UTR5,
            REGION_CDS_PHASE0,
            REGION_UTR3,
        ):
            assert required in region_values

        exon_values = set(np.unique(y_exon).tolist())
        assert EXON_FIRST in exon_values
        assert EXON_LAST in exon_values

        return
