"""Build canonical_dataset.txt from UCSC tables or GENCODE GTF.

Supports two modes controlled by ``canonical_dataset.mode``:

* **ucsc** (recommended) -- Joins UCSC knownCanonical table
  (one canonical transcript per gene) with the GENCODE Basic
  table (exon structure). This matches the original SpliceAI
  training data pipeline.
* **gtf** -- Parses a GENCODE GTF directly and selects
  canonical transcripts via MANE Select > APPRIS principal
  > longest exonic length.

Both modes merge paralog flags from an external TSV and
write the final tab-separated splice table consumed by the
h5 data builders.

An optional Transcript Support Level (TSL) filter can be
enabled via ``canonical_dataset.tsl_max`` (1-5).  When set,
only transcripts with TSL <= the threshold are retained.
In UCSC mode, TSL values are cross-referenced from the
GENCODE GTF specified in ``paths.gtf``.

An optional GENCODE annotation level filter can be
enabled via ``canonical_dataset.level_max`` (1-3).  When
set, only transcripts with level <= the threshold are
retained; transcripts without a level annotation are
excluded.  In UCSC mode, level values are cross-referenced
from the GENCODE GTF specified in ``paths.gtf``.

Usage:
    uv run -m process_input.canonical_dataset_builder \\
        --config-path config/build_data/h5_data.yaml

    uv run -m process_input.canonical_dataset_builder \\
        --config-path config/build_data/h5_data.yaml \\
        -o /path/to/canonical_dataset.txt
"""

import argparse
import gzip
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

STANDARD_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}

CHROM_SORT_KEY = {
    **{f"chr{i}": i for i in range(1, 23)},
    "chrX": 23,
    "chrY": 24,
}

# GTF (GFF2) is a 9-column TSV. Spec: https://mblab.wustl.edu/GTF22.html
#   0: seqname     3: start   6: strand
#   1: source      4: end     7: frame
#   2: feature     5: score   8: attributes (key "value"; ...)
GTF_NUM_COLS = 9
GTF_FEATURE_COL = 2
GTF_ATTRIBUTES_COL = 8


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=("Create canonical_dataset.txt from UCSC tables or GENCODE GTF"),
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=("Output file path (overrides paths.splice_table)"),
    )

    return parser.parse_args()


def setup_logging(cfg: DictConfig) -> None:
    """Configure logging from config.

    Args:
        cfg (DictConfig): Configuration object.
    """
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format,
        datefmt=cfg.logging.date_format,
    )

    return


# ── Shared helpers ──────────────────────────────────


def _sort_key(tx: dict) -> tuple[int, int]:
    """Sort key: chromosome number then position.

    Args:
        tx (dict): Transcript dict with 'chrom'
            and 'tx_start'.

    Returns:
        tuple[int, int]: (chrom_order, tx_start).
    """
    return (
        CHROM_SORT_KEY.get(tx["chrom"], 25),
        tx["tx_start"],
    )


def load_paralog_flags(
    paralog_path: str,
) -> dict[str, int]:
    """Load paralog flags from a TSV file.

    Expects a two-column TSV with header:
    ``gene_name<TAB>paralog_flag``

    Args:
        paralog_path (str): Path to paralog flags TSV.

    Returns:
        dict[str, int]: Gene name -> paralog flag (0 or 1).
    """
    logger.info(f"Loading paralog flags: {paralog_path}")

    df = pd.read_csv(
        paralog_path,
        sep="\t",
        dtype={"paralog_flag": int},
    )
    flags = dict(zip(df["gene_name"], df["paralog_flag"]))

    n_with = sum(1 for v in flags.values() if v == 1)
    n_without = sum(1 for v in flags.values() if v == 0)
    logger.info(f"  Loaded {len(flags)} flags")
    logger.info(f"  With paralogs: {n_with}, Without: {n_without}")

    return flags


def load_tsl_from_gtf(
    gtf_path: str,
) -> dict[str, int]:
    """Load Transcript Support Levels from GENCODE GTF.

    Parses ``transcript`` features and extracts the
    ``transcript_support_level`` attribute.  Returns a
    mapping from versioned transcript ID (e.g.
    ``ENST00000367771.11``) to its integer TSL (1-5).
    Transcripts without a TSL annotation are omitted.

    Args:
        gtf_path (str): Path to GENCODE GTF file
            (plain text or gzipped).

    Returns:
        dict[str, int]: transcript_id -> TSL (1-5).
    """
    logger.info(f"Loading TSL from GTF: {gtf_path}")

    is_gz = gtf_path.endswith(".gz")
    open_func = gzip.open if is_gz else open
    mode = "rt" if is_gz else "r"

    tsl_map: dict[str, int] = {}

    with open_func(gtf_path, mode) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < GTF_NUM_COLS or fields[GTF_FEATURE_COL] != "transcript":
                continue

            attrs = parse_gtf_attributes(fields[GTF_ATTRIBUTES_COL])
            tid = attrs.get("transcript_id")
            tsl = attrs.get("transcript_support_level")

            if tid and tsl and tsl.isdigit():
                tsl_map[tid] = int(tsl)

    logger.info(f"  {len(tsl_map):,} transcripts with TSL loaded")

    return tsl_map


def load_level_from_gtf(
    gtf_path: str,
) -> dict[str, int]:
    """Load GENCODE annotation levels from GTF.

    Parses ``transcript`` features and extracts the
    ``level`` attribute.  Returns a mapping from versioned
    transcript ID (e.g. ``ENST00000367771.11``) to its
    integer level (1-3).  Transcripts without a level
    annotation are omitted.

    Args:
        gtf_path (str): Path to GENCODE GTF file
            (plain text or gzipped).

    Returns:
        dict[str, int]: transcript_id -> level (1-3).
    """
    logger.info(f"Loading level from GTF: {gtf_path}")

    is_gz = gtf_path.endswith(".gz")
    open_func = gzip.open if is_gz else open
    mode = "rt" if is_gz else "r"

    level_map: dict[str, int] = {}

    with open_func(gtf_path, mode) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < GTF_NUM_COLS or fields[GTF_FEATURE_COL] != "transcript":
                continue

            attrs = parse_gtf_attributes(fields[GTF_ATTRIBUTES_COL])
            tid = attrs.get("transcript_id")
            level = attrs.get("level")

            if tid and level and level.isdigit():
                level_map[tid] = int(level)

    logger.info(f"  {len(level_map):,} transcripts with level loaded")

    return level_map


def write_canonical_dataset(
    transcripts: list[dict],
    output_path: str,
    paralog_flags: dict[str, int] | None = None,
    test_chroms: set[str] | None = None,
) -> None:
    """Write canonical dataset in SpliceAI-compatible format.

    Output columns (tab-separated, no header):
        gene_name, paralog_flag, chrom, strand,
        tx_start, tx_end, jn_start, jn_end,
        exon_start, exon_end, cds_start, cds_end, cds_frame

    Columns 0-7 are the legacy SpliceAI columns; columns 8-12
    are append-only region-annotation columns introduced for
    auxiliary region prediction.  All position fields are
    1-based (closed) and use comma-separated strings with a
    trailing comma to match the legacy junction format.
    cds_frame stores the GFF3 phase (0/1/2) for each CDS exon
    in genomic ascending order.  Non-coding transcripts emit a
    bare trailing comma for the cds_* columns.

    Args:
        transcripts (list[dict]): Canonical transcript dicts.
            Each must have keys: gene_name, chrom, strand,
            tx_start, tx_end, jn_start_str, jn_end_str,
            exon_start_str, exon_end_str, cds_start_str,
            cds_end_str, cds_frame_str.
        output_path (str): Output file path.
        paralog_flags (dict[str, int] | None): Gene-name to
            paralog flag mapping.  If None, flags are derived
            from test_chroms (0=test, 1=train).
        test_chroms (set[str] | None): Test chromosome set for
            fallback flag assignment.
    """
    if test_chroms is None:
        test_chroms = {"chr1", "chr3", "chr5", "chr7", "chr9"}

    n_missing_flags = 0

    logger.info(f"Writing {len(transcripts)} transcripts to {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for tx in transcripts:
            gene_name = tx["gene_name"]
            chrom = tx["chrom"]
            strand = tx["strand"]
            tx_start = tx["tx_start"]
            tx_end = tx["tx_end"]
            jn_start_str = tx["jn_start_str"]
            jn_end_str = tx["jn_end_str"]
            exon_start_str = tx.get("exon_start_str", ",")
            exon_end_str = tx.get("exon_end_str", ",")
            cds_start_str = tx.get("cds_start_str", ",")
            cds_end_str = tx.get("cds_end_str", ",")
            cds_frame_str = tx.get("cds_frame_str", ",")

            if paralog_flags is not None:
                if gene_name in paralog_flags:
                    flag = paralog_flags[gene_name]
                else:
                    flag = 1
                    n_missing_flags += 1
            else:
                flag = 0 if chrom in test_chroms else 1

            line = (
                f"{gene_name}\t{flag}\t{chrom}\t"
                f"{strand}\t{tx_start}\t{tx_end}\t"
                f"{jn_start_str}\t{jn_end_str}\t"
                f"{exon_start_str}\t{exon_end_str}\t"
                f"{cds_start_str}\t{cds_end_str}\t"
                f"{cds_frame_str}\n"
            )
            f.write(line)

    if n_missing_flags > 0:
        logger.warning(
            f"{n_missing_flags} genes not found in paralog flags (defaulted to 1)"
        )

    logger.info(f"Done: {len(transcripts)} transcripts written")

    return


# ── UCSC mode ───────────────────────────────────────


def compute_ucsc_cds_spans_and_frames(
    exons_0: list[tuple[int, int]],
    cds_start_0: int,
    cds_end: int,
    strand: str,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Compute per-exon CDS spans and GFF3 phases for UCSC mode.

    Intersects each exon (genomic ascending, 0-based half-open)
    with the contiguous CDS range ``[cds_start_0, cds_end)`` and
    derives the GFF3 phase of every CDS exon.  Phase is computed
    in 5'→3' order using the cumulative sum of preceding CDS
    lengths: ``phase = (3 - sum_so_far % 3) % 3``.  The first
    CDS exon in 5'→3' order is assumed to start at phase 0,
    which holds for canonical protein-coding transcripts but may
    misclassify selenocysteine or non-AUG-start cases.

    Args:
        exons_0 (list[tuple[int, int]]): Exons in genomic
            ascending order, 0-based half-open ``(start, end)``.
        cds_start_0 (int): UCSC ``cdsStart`` (0-based).
        cds_end (int): UCSC ``cdsEnd`` (0-based half-open).
        strand (str): Strand direction, ``'+'`` or ``'-'``.

    Returns:
        tuple[list[tuple[int, int]], list[int]]:
            (cds_spans_1based, cds_frames) -- cds_spans_1based
            holds ``(start, end)`` pairs in genomic ascending
            order (1-based closed); cds_frames is aligned with
            cds_spans_1based.
    """
    intersections: list[tuple[int, int]] = []
    for s0, e in exons_0:
        cs = max(s0, cds_start_0)
        ce = min(e, cds_end)
        if cs < ce:
            intersections.append((cs, ce))

    if strand == "-":
        order_53 = list(range(len(intersections) - 1, -1, -1))
    else:
        order_53 = list(range(len(intersections)))

    frames_genomic = [0] * len(intersections)
    cum_len = 0
    for idx in order_53:
        s0, e = intersections[idx]
        frames_genomic[idx] = (3 - (cum_len % 3)) % 3
        cum_len += e - s0

    spans_1based = [(s0 + 1, e) for s0, e in intersections]

    return spans_1based, frames_genomic


def parse_ucsc_tables(
    known_canonical_path: str,
    gencode_basic_path: str,
    tsl_map: dict[str, int] | None = None,
    tsl_max: int | None = None,
    level_map: dict[str, int] | None = None,
    level_max: int | None = None,
) -> list[dict]:
    """Build canonical transcripts from UCSC tables.

    Joins ``knownCanonical`` (one transcript per gene
    cluster) with ``gencodeBasic`` (exon structure) on
    transcript ID.  Only protein-coding transcripts
    (``cdsStart != cdsEnd``) are retained.  Converts
    UCSC 0-based half-open coordinates to the 1-based
    format expected by SpliceAI.

    When *tsl_max* is set, transcripts whose TSL exceeds
    the threshold are excluded.  TSL values are looked
    up from *tsl_map* (typically loaded via
    :func:`load_tsl_from_gtf`).  Transcripts absent from
    the map are treated as TSL N/A and kept.

    When *level_max* is set, transcripts whose level
    exceeds the threshold are excluded.  Level values
    are looked up from *level_map* (typically loaded via
    :func:`load_level_from_gtf`).  Transcripts absent
    from the map (N/A) are also excluded.

    Coordinate conversion:
        - TX_START = txStart + 1  (0-based → 1-based)
        - TX_END   = txEnd        (unchanged)
        - JN_START = exonEnds[:-1]  (already 1-based)
        - JN_END   = exonStarts[1:] + 1  (0-based → 1-based)

    Args:
        known_canonical_path (str): Path to UCSC
            knownCanonical table (6-col TSV with header).
        gencode_basic_path (str): Path to UCSC
            gencodeBasic table (16-col TSV with header).
        tsl_map (dict[str, int] | None): transcript_id
            to TSL mapping for filtering.
        tsl_max (int | None): Maximum allowed TSL
            (1-5). None disables the filter.
        level_map (dict[str, int] | None): transcript_id
            to level mapping for filtering.
        level_max (int | None): Maximum allowed GENCODE
            annotation level (1-3). None disables the
            filter.

    Returns:
        list[dict]: Canonical transcripts sorted by
            chromosome and position. Each dict has keys:
            gene_name, chrom, strand, tx_start, tx_end,
            exons, jn_start_str, jn_end_str.
    """
    logger.info(f"Reading knownCanonical: {known_canonical_path}")
    canonical_tx_ids: set[str] = set()
    with open(known_canonical_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if len(cols) < 5:
                continue
            canonical_tx_ids.add(cols[4])

    logger.info(f"  {len(canonical_tx_ids)} canonical transcript IDs loaded")

    logger.info(f"Reading gencodeBasic: {gencode_basic_path}")
    tx_map: dict[str, dict] = {}
    with open(gencode_basic_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            if len(cols) < 16:
                continue
            tx_id = cols[1]
            if tx_id not in canonical_tx_ids:
                continue

            chrom = cols[2]
            if chrom not in STANDARD_CHROMS:
                continue

            gene_name = cols[12]
            strand = cols[3]
            tx_start_0 = int(cols[4])
            tx_end = int(cols[5])
            cds_start = int(cols[6])
            cds_end = int(cols[7])

            if cds_start == cds_end:
                continue

            exon_count = int(cols[8])

            exon_starts_0 = [int(s) for s in cols[9].rstrip(",").split(",") if s]
            exon_ends = [int(e) for e in cols[10].rstrip(",").split(",") if e]

            if len(exon_starts_0) != exon_count:
                continue
            if len(exon_ends) != exon_count:
                continue
            if exon_count < 2:
                continue

            tx_map[tx_id] = {
                "gene_name": gene_name,
                "chrom": chrom,
                "strand": strand,
                "tx_start": tx_start_0 + 1,
                "tx_end": tx_end,
                "exon_starts_0": exon_starts_0,
                "exon_ends": exon_ends,
                "exon_count": exon_count,
                "cds_start_0": cds_start,
                "cds_end": cds_end,
            }

    logger.info(f"  {len(tx_map)} transcripts matched (>= 2 exons, standard chroms)")

    if tsl_max is not None and tsl_map is not None:
        before = len(tx_map)
        filtered: dict[str, dict] = {}
        n_kept = 0
        n_dropped = 0
        n_na = 0
        for tid, tx in tx_map.items():
            tsl = tsl_map.get(tid)
            if tsl is None:
                n_na += 1
                filtered[tid] = tx
            elif tsl <= tsl_max:
                n_kept += 1
                filtered[tid] = tx
            else:
                n_dropped += 1
        tx_map = filtered
        logger.info(
            f"  TSL filter (max={tsl_max}): "
            f"kept={n_kept}, dropped={n_dropped}, "
            f"N/A={n_na}, total={len(tx_map)}/{before}"
        )

    if level_max is not None and level_map is not None:
        before = len(tx_map)
        lvl_filtered: dict[str, dict] = {}
        lvl_n_kept = 0
        lvl_n_dropped = 0
        lvl_n_na = 0
        for tid, tx in tx_map.items():
            lvl = level_map.get(tid)
            if lvl is None:
                lvl_n_na += 1
            elif lvl <= level_max:
                lvl_n_kept += 1
                lvl_filtered[tid] = tx
            else:
                lvl_n_dropped += 1
        tx_map = lvl_filtered
        logger.info(
            f"  Level filter (max={level_max}): "
            f"kept={lvl_n_kept}, "
            f"dropped={lvl_n_dropped}, "
            f"N/A={lvl_n_na}, "
            f"total={len(tx_map)}/{before}"
        )
        if lvl_n_na > 0:
            logger.warning(
                f"  {lvl_n_na} transcripts had no level annotation and were excluded"
            )

    # Deduplicate: one transcript per gene
    # (knownCanonical may map multiple clusters to
    # the same gene; keep the one with most exons,
    # then longest exonic length)
    gene_best: dict[str, dict] = {}
    for tx_id, tx in tx_map.items():
        gene = tx["gene_name"]
        if gene not in gene_best:
            gene_best[gene] = tx
        else:
            prev = gene_best[gene]
            cur_len = sum(
                e - s
                for s, e in zip(
                    tx["exon_starts_0"],
                    tx["exon_ends"],
                )
            )
            prev_len = sum(
                e - s
                for s, e in zip(
                    prev["exon_starts_0"],
                    prev["exon_ends"],
                )
            )
            cur_score = (tx["exon_count"], cur_len)
            prev_score = (
                prev["exon_count"],
                prev_len,
            )
            if cur_score > prev_score:
                gene_best[gene] = tx

    # Build junction strings and output list
    canonical: list[dict] = []
    for gene, tx in gene_best.items():
        es0 = tx["exon_starts_0"]
        ee = tx["exon_ends"]

        # Internal boundaries only (strand-independent)
        jn_start = ee[:-1]
        jn_end = [s + 1 for s in es0[1:]]

        jn_start_str = ",".join(str(v) for v in jn_start) + ","
        jn_end_str = ",".join(str(v) for v in jn_end) + ","

        # Region annotation columns (1-based closed positions).
        exons_0 = list(zip(es0, ee))
        exon_start_str = ",".join(str(s + 1) for s in es0) + ","
        exon_end_str = ",".join(str(e) for e in ee) + ","

        cds_spans, cds_frames = compute_ucsc_cds_spans_and_frames(
            exons_0,
            tx["cds_start_0"],
            tx["cds_end"],
            tx["strand"],
        )
        if cds_spans:
            cds_start_str = ",".join(str(s) for s, _ in cds_spans) + ","
            cds_end_str = ",".join(str(e) for _, e in cds_spans) + ","
            cds_frame_str = ",".join(str(p) for p in cds_frames) + ","
        else:
            cds_start_str = ","
            cds_end_str = ","
            cds_frame_str = ","

        canonical.append(
            {
                "gene_name": gene,
                "chrom": tx["chrom"],
                "strand": tx["strand"],
                "tx_start": tx["tx_start"],
                "tx_end": tx["tx_end"],
                "exons": exons_0,
                "jn_start_str": jn_start_str,
                "jn_end_str": jn_end_str,
                "exon_start_str": exon_start_str,
                "exon_end_str": exon_end_str,
                "cds_start_str": cds_start_str,
                "cds_end_str": cds_end_str,
                "cds_frame_str": cds_frame_str,
            }
        )

    canonical.sort(key=_sort_key)

    logger.info(f"Selected {len(canonical)} canonical genes")

    return canonical


# ── GTF mode ────────────────────────────────────────


def parse_gtf_attributes(
    attr_string: str,
) -> dict[str, str]:
    """Parse GTF attribute string into a dictionary.

    GENCODE repeats some keys, one ``tag`` entry per tag; the
    values of a repeated key are joined with commas.

    Args:
        attr_string (str): GTF 9th-column attribute
            string.

    Returns:
        dict[str, str]: Attribute key-value pairs.
    """
    attributes = {}
    for attr in attr_string.strip().split(";"):
        attr = attr.strip()
        if not attr:
            continue
        if '"' in attr:
            key, value = attr.split(' "', 1)
            value = value.rstrip('"')
        else:
            parts = attr.split(" ", 1)
            if len(parts) == 2:
                key, value = parts
            else:
                continue
        key, value = key.strip(), value.strip()
        if key in attributes:
            attributes[key] = f"{attributes[key]},{value}"
        else:
            attributes[key] = value

    return attributes


def get_canonical_score(
    attributes: dict[str, str],
) -> tuple[int, int, int]:
    """Rank a transcript for representative selection.

    Priority (descending): MANE_Select, MANE_Plus_Clinical,
    Ensembl_canonical. A transcript carrying none of the three
    scores ``(0, 0, 0)`` and is never selected.

    Args:
        attributes (dict[str, str]): GTF attributes dict; repeated
            ``tag`` values are comma-joined by
            :func:`parse_gtf_attributes`.

    Returns:
        tuple[int, int, int]: (mane_select, mane_plus_clinical,
            ensembl_canonical) flags.
    """
    tags = set(attributes.get("tag", "").split(","))

    return (
        int("MANE_Select" in tags),
        int("MANE_Plus_Clinical" in tags),
        int("Ensembl_canonical" in tags),
    )


def parse_gencode_gtf(
    gtf_path: str,
) -> dict[str, dict[str, dict]]:
    """Parse GENCODE GTF and extract transcript info.

    Keeps every gene and transcript type on standard
    chromosomes (chr1-22, chrX, chrY);
    :func:`select_canonical_transcripts` drops genes whose
    selected transcript has no CDS.  ``tx_start`` is 1-based
    like UCSC mode; exon, CDS and stop codon starts are
    0-based.

    Args:
        gtf_path (str): Path to GENCODE GTF file
            (plain text or gzipped).

    Returns:
        dict[str, dict[str, dict]]: Nested dict of
            gene_name -> transcript_id -> info.
    """
    genes: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "chrom": None,
                "strand": None,
                "tx_start": float("inf"),
                "tx_end": 0,
                "exons": [],
                "cds": [],
                "stop_codon": [],
                "score": (0, 0, 0),
                "readthrough": False,
                "transcript_type": None,
                "tsl": None,
                "level": None,
            }
        )
    )

    is_gz = gtf_path.endswith(".gz")
    open_func = gzip.open if is_gz else open
    mode = "rt" if is_gz else "r"

    logger.info(f"Parsing GTF: {gtf_path}")

    line_num = 0
    with open_func(gtf_path, mode) as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < GTF_NUM_COLS:
                continue

            (
                chrom,
                _src,
                feature,
                start,
                end,
                _sc,
                strand,
                frame_raw,
                attrs,
            ) = fields
            start, end = int(start), int(end)

            if not chrom.startswith("chr"):
                chrom = "chr" + chrom
            if chrom not in STANDARD_CHROMS:
                continue

            attributes = parse_gtf_attributes(attrs)

            gn = attributes.get("gene_name", "")
            tx_id = attributes.get("transcript_id", "")
            t_type = attributes.get("transcript_type", "")

            if not gn or not tx_id:
                continue

            tx = genes[gn][tx_id]

            if feature == "transcript":
                tx["chrom"] = chrom
                tx["strand"] = strand
                tx["tx_start"] = start
                tx["tx_end"] = end
                tx["score"] = get_canonical_score(attributes)
                tx["readthrough"] = "readthrough_transcript" in attributes.get(
                    "tag", ""
                ).split(",")
                tx["transcript_type"] = t_type
                tsl_raw = attributes.get("transcript_support_level")
                if tsl_raw and tsl_raw.isdigit():
                    tx["tsl"] = int(tsl_raw)
                level_raw = attributes.get("level")
                if level_raw and level_raw.isdigit():
                    tx["level"] = int(level_raw)

            elif feature == "exon":
                tx["exons"].append((start - 1, end))
                tx["tx_start"] = min(tx["tx_start"], start)
                tx["tx_end"] = max(tx["tx_end"], end)
                if tx["chrom"] is None:
                    tx["chrom"] = chrom
                    tx["strand"] = strand

            elif feature in ("CDS", "stop_codon"):
                # GFF3 phase column ("." for missing).
                if frame_raw.isdigit():
                    cds_frame = int(frame_raw)
                else:
                    cds_frame = 0
                key = "cds" if feature == "CDS" else "stop_codon"
                tx[key].append((start - 1, end, cds_frame))

            if line_num % 500_000 == 0:
                logger.info(f"  Processed {line_num:,} lines...")

    logger.info(f"  Total: {line_num:,} lines")
    logger.info(f"  Genes: {len(genes)}")

    return genes


def merge_stop_codon(
    cds: list[tuple[int, int, int]],
    stop_codon: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Add GTF stop codon segments to the CDS segments.

    GENCODE GTF leaves the stop codon out of ``CDS`` while
    UCSC genePred includes it, so this keeps the GTF mode CDS
    columns on the UCSC mode convention.  A stop codon segment
    that abuts a CDS segment extends it; the phase is kept
    because the stop codon sits at the segment's 3' end.  A
    segment in an exon of its own (the stop codon after an exon
    boundary) is added with its GTF phase.

    Args:
        cds (list[tuple[int, int, int]]): ``(start_0, end,
            phase)`` CDS segments.
        stop_codon (list[tuple[int, int, int]]): ``(start_0,
            end, phase)`` stop codon segments.

    Returns:
        list[tuple[int, int, int]]: Merged segments in genomic
            ascending order.
    """
    merged = sorted(cds)
    for s0, e, phase in sorted(stop_codon):
        for i, (cs0, ce, cphase) in enumerate(merged):
            if ce == s0:  # stop codon follows the segment (+ strand)
                merged[i] = (cs0, e, cphase)
                break
            if e == cs0:  # stop codon precedes the segment (- strand)
                merged[i] = (s0, ce, cphase)
                break
        else:
            merged.append((s0, e, phase))

    return sorted(merged)


def select_canonical_transcripts(
    genes: dict[str, dict[str, dict]],
    tsl_max: int | None = None,
    level_max: int | None = None,
) -> list[dict]:
    """Select one representative transcript per gene.

    Picks MANE_Select, then MANE_Plus_Clinical, then
    Ensembl_canonical (:func:`get_canonical_score`); the
    longest exonic length breaks ties.  A gene is excluded
    when that transcript carries none of the three tags, is a
    readthrough transcript other than MANE_Select, has no CDS,
    or has fewer than 2 exons.  Genes are keyed by gene_name, so the chrY copy of
    a pseudoautosomal gene loses to the chrX copy carrying
    MANE_Select, and a readthrough gene sharing a name with a
    MANE gene loses to it.  Coordinates are written 1-based
    as in UCSC mode, with the stop codon inside the CDS
    (:func:`merge_stop_codon`).

    When *tsl_max* is set, only transcripts with TSL
    <= *tsl_max* are considered.  Transcripts without
    a TSL annotation (None) are kept.

    When *level_max* is set, only transcripts with
    level <= *level_max* are considered.  Transcripts
    without a level annotation (None) are excluded.

    Args:
        genes (dict[str, dict[str, dict]]): Nested dict
            gene_name -> transcript_id -> info.
        tsl_max (int | None): Maximum allowed TSL
            (1-5). None disables the filter.
        level_max (int | None): Maximum allowed GENCODE
            annotation level (1-3). None disables the
            filter.

    Returns:
        list[dict]: Canonical transcripts sorted by
            chromosome and position. Each dict includes
            pre-computed jn_start_str / jn_end_str.
    """
    if tsl_max is not None:
        tsl_n_dropped = 0
        tsl_n_kept = 0
        tsl_n_na = 0

    if level_max is not None:
        lvl_n_dropped = 0
        lvl_n_kept = 0
        lvl_n_na = 0

    canonical = []
    excluded: Counter[str] = Counter()

    for gene_name, transcripts in genes.items():
        best_tx = None
        best_score = (-1, -1, -1, 0)

        for tx_id, tx_info in transcripts.items():
            if not tx_info["exons"]:
                continue

            if tsl_max is not None:
                tsl = tx_info.get("tsl")
                if tsl is not None and tsl > tsl_max:
                    tsl_n_dropped += 1
                    continue
                if tsl is None:
                    tsl_n_na += 1
                else:
                    tsl_n_kept += 1

            if level_max is not None:
                lvl = tx_info.get("level")
                if lvl is None:
                    lvl_n_na += 1
                    continue
                if lvl > level_max:
                    lvl_n_dropped += 1
                    continue
                lvl_n_kept += 1

            tx_len = sum(e[1] - e[0] for e in tx_info["exons"])
            score = tx_info["score"] + (tx_len,)

            if score > best_score:
                best_score = score
                best_tx = {
                    "gene_name": gene_name,
                    "transcript_id": tx_id,
                    **tx_info,
                }

        if best_tx is None:
            continue
        if not any(best_score[:3]):
            excluded["no_mane_or_ensembl_canonical"] += 1
            continue
        if best_tx.get("readthrough") and not best_score[0]:
            excluded["readthrough"] += 1
            continue
        if not best_tx.get("cds"):
            excluded["no_cds"] += 1
            continue
        if len(best_tx["exons"]) < 2:
            excluded["single_exon"] += 1
            continue

        exons = sorted(best_tx["exons"], key=lambda x: x[0])
        jn_start = [str(e[1]) for e in exons[:-1]]
        jn_end = [str(e[0] + 1) for e in exons[1:]]
        best_tx["jn_start_str"] = ",".join(jn_start) + ","
        best_tx["jn_end_str"] = ",".join(jn_end) + ","

        # Region annotation columns (1-based closed positions).
        exon_starts_1 = [str(e[0] + 1) for e in exons]
        exon_ends_1 = [str(e[1]) for e in exons]
        best_tx["exon_start_str"] = ",".join(exon_starts_1) + ","
        best_tx["exon_end_str"] = ",".join(exon_ends_1) + ","

        cds_sorted = merge_stop_codon(best_tx["cds"], best_tx.get("stop_codon", []))
        best_tx["cds_start_str"] = ",".join(str(c[0] + 1) for c in cds_sorted) + ","
        best_tx["cds_end_str"] = ",".join(str(c[1]) for c in cds_sorted) + ","
        best_tx["cds_frame_str"] = ",".join(str(c[2]) for c in cds_sorted) + ","

        canonical.append(best_tx)

    canonical.sort(key=_sort_key)

    logger.info(f"  Excluded genes: {dict(excluded)}")

    if tsl_max is not None:
        logger.info(
            f"  TSL filter (max={tsl_max}): "
            f"kept={tsl_n_kept}, "
            f"dropped={tsl_n_dropped}, "
            f"N/A={tsl_n_na}"
        )

    if level_max is not None:
        logger.info(
            f"  Level filter (max={level_max}): "
            f"kept={lvl_n_kept}, "
            f"dropped={lvl_n_dropped}, "
            f"N/A={lvl_n_na}"
        )
        if lvl_n_na > 0:
            logger.warning(
                f"  {lvl_n_na} transcripts had no level annotation and were excluded"
            )

    logger.info(f"Selected {len(canonical)} canonical transcripts (>= 2 exons)")

    return canonical


# ── Orchestrator ────────────────────────────────────


def build_canonical_dataset(
    cfg: DictConfig,
    output_path: str | None = None,
) -> str:
    """Build canonical_dataset.txt.

    Dispatches to UCSC or GTF mode based on
    ``cfg.canonical_dataset.mode``.

    Args:
        cfg (DictConfig): Configuration with
            canonical_dataset.mode and paths.
        output_path (str | None): Output path override.

    Returns:
        str: Path to the written file.
    """
    start_time = time.time()

    mode = cfg.canonical_dataset.mode
    tsl_max = cfg.canonical_dataset.get("tsl_max")
    if tsl_max is not None:
        tsl_max = int(tsl_max)
        logger.info(f"TSL filter enabled: max={tsl_max}")

    level_max = cfg.canonical_dataset.get("level_max")
    if level_max is not None:
        level_max = int(level_max)
        if not 1 <= level_max <= 3:
            raise ValueError(f"level_max must be 1-3, got {level_max}")
        logger.info(f"Level filter enabled: max={level_max}")

    if mode == "ucsc":
        tsl_map = None
        if tsl_max is not None:
            tsl_map = load_tsl_from_gtf(str(cfg.paths.gtf))
        level_map = None
        if level_max is not None:
            level_map = load_level_from_gtf(str(cfg.paths.gtf))
        canonical = parse_ucsc_tables(
            str(cfg.paths.ucsc_known_canonical),
            str(cfg.paths.ucsc_gencode_basic),
            tsl_map=tsl_map,
            tsl_max=tsl_max,
            level_map=level_map,
            level_max=level_max,
        )
    elif mode == "gtf":
        genes = parse_gencode_gtf(str(cfg.paths.gtf))
        canonical = select_canonical_transcripts(
            genes,
            tsl_max=tsl_max,
            level_max=level_max,
        )
    else:
        raise ValueError(
            f"Unknown canonical_dataset.mode: {mode}. Use 'ucsc' or 'gtf'."
        )

    paralog_flags = load_paralog_flags(str(cfg.paths.paralog_flags))

    if output_path is None:
        output_path = str(cfg.paths.splice_table)

    test_chroms = set(cfg.test_chromosomes)

    write_canonical_dataset(
        canonical,
        output_path,
        paralog_flags=paralog_flags,
        test_chroms=test_chroms,
    )

    elapsed = time.time() - start_time
    logger.info(f"Elapsed: {elapsed:.2f}s")
    logger.info(f"Output: {output_path}")

    return output_path


def main() -> None:
    """Run canonical dataset building from CLI."""
    args = parse_args()

    config_path = Path(args.config_path)
    if not config_path.exists():
        print(
            f"Config not found: {config_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = OmegaConf.load(config_path)
    setup_logging(cfg)

    build_canonical_dataset(cfg, args.output)

    return


if __name__ == "__main__":
    main()
