"""Build SpliceVarDB H5 in gencode_anchored mode with allchroms + indel support.

Extends ``splicevardb_gencode_anchored_builder`` with two changes:

1. Uses an **all-24-chromosome** gencode H5 (e.g. ``dataset_all_all.h5``)
   and ``chrom_filter=None``, so variants on chr1/3/5/7/9 (which the
   legacy paralog-hold-out builder dropped as ``out_of_chrom``) are also
   anchored.

2. Adds **indel handling** via a "ref chunk byte-exact + alt
   length-corrected from outside chunk" strategy (per user instruction):

   * **X_ref**: same as SNV path — fetched byte-exact from the gencode H5
     chunk that the model saw during training.
   * **X_alt**: build by patching the variant region in the chunk with
     ``alt`` (or RC(``alt``) for negative-strand genes), then **adjust
     length to 15000**:

     - *Deletion* (``len(ref) > len(alt)``): chunk is now shorter by
       ``L_ref − L_alt`` bytes. Pad on the **3' end of gene reading
       direction** (i.e. rightmost in the chunk) with additional
       nucleotides fetched from genome FASTA — bases that were *not*
       in the original ref chunk (just beyond its 3' boundary in gene
       direction). For ``+`` strand: append next genomic bases. For
       ``−`` strand: fetch + strand positions before the chunk's first
       + strand position, RC, append.

     - *Insertion* (``len(ref) < len(alt)``): chunk is now longer by
       ``L_alt − L_ref`` bytes. Truncate from the same 3' end of gene
       reading direction (i.e. trim the tail of the array).

     - *MNV / complex same-length*: just patch in place (no length
       adjustment).

Variants that fail ref-allele verification or anchor lookup are skipped
with diagnostic counts. Indel-specific failure modes are tracked as
``indel_ref_mismatch`` and ``indel_pad_fetch_failed``.

Usage::

    python -m process_input.splicevardb_gencode_anchored_allchroms_indel_builder \\
        --input-h5 /NAS/.../splicevardb_test.h5 \\
        --output-h5 /NAS/.../splicevardb_test_gencode_anchored_allchroms_indel.h5 \\
        --gencode-final-h5 /NAS/.../h5_cl10000/dataset_all_all.h5 \\
        --gencode-splice-table /NAS/.../canonical_dataset_v49_regions.txt \\
        --genome-fasta /NAS/.../genome/GRCh38.fa \\
        --no-chrom-filter \\
        --diagnostics-json /NAS/.../diagnostics_test_allchroms_indel.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from pyfaidx import Fasta

from process_input.gencode_anchor import (
    AnchorMatch,
    build_genome_index,
    encode_alt_allele,
    encode_ref_allele,
    find_anchor,
    load_canonical_table,
    verify_chunk_sizes,
)

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_CHROMS: tuple[str, ...] = (
    "chr2", "chr4", "chr6", "chr8",
    "chr10", "chr11", "chr12", "chr13", "chr14", "chr15",
    "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22",
    "chrX", "chrY",
)
DEFAULT_SL: int = 5000
TOTAL_LENGTH: int = 15000

_NT_TO_INT: dict[str, int] = {
    "N": 0, "A": 1, "C": 2, "G": 3, "T": 4,
    "n": 0, "a": 1, "c": 2, "g": 3, "t": 4,
}
_COMPLEMENT: dict[str, str] = {
    "A": "T", "T": "A", "C": "G", "G": "C", "N": "N",
    "a": "t", "t": "a", "c": "g", "g": "c", "n": "n",
}


def _decode(value: object) -> str:
    """Coerce an HDF5 string element to a Python str."""
    return value.decode() if isinstance(value, bytes) else str(value)


def _reverse_complement(seq: str) -> str:
    """Return the reverse-complement of ``seq`` (Ns preserved)."""
    return "".join(_COMPLEMENT.get(b, "N") for b in seq[::-1])


def _encode_to_int8(seq: str) -> np.ndarray:
    """Encode a DNA string into int8 ``{N=0, A=1, C=2, G=3, T=4}``."""
    return np.array([_NT_TO_INT.get(b, 0) for b in seq], dtype=np.int8)


def _resolve_fasta_chrom(fasta: Fasta, chrom: str) -> str | None:
    """Return the FASTA's chromosome key matching ``chrom``."""
    if chrom in fasta:
        return chrom
    bare = chrom[3:] if chrom.startswith("chr") else chrom
    if bare in fasta:
        return bare
    if f"chr{bare}" in fasta:
        return f"chr{bare}"
    return None


def _slice_window(
    fasta: Fasta,
    chrom: str,
    start_0based: int,
    end_0based: int,
) -> str:
    """Return a padded slice of ``chrom`` from FASTA (Ns at OOB edges)."""
    chrom_len = len(fasta[chrom])
    left_pad = max(0, -start_0based)
    right_pad = max(0, end_0based - chrom_len)
    slice_start = max(0, start_0based)
    slice_end = min(chrom_len, end_0based)
    body = str(fasta[chrom][slice_start:slice_end]).upper()
    return "N" * left_pad + body + "N" * right_pad


def _read_input(h5_path: str) -> dict[str, np.ndarray]:
    """Read the variant-centric input H5 into a dict of arrays."""
    out: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        for key in (
            "label", "pos", "chrom", "ref", "alt",
            "gene_symbol", "gene_location", "gene_region", "strand",
        ):
            ds = f[key]
            out[key] = np.array(
                [_decode(x) for x in ds[:]] if ds.dtype == object else ds[:]
            )
        out["__attrs__"] = dict(f.attrs)
    return out


def _open_chunks(h5_path: str) -> tuple[h5py.File, dict[int, np.ndarray]]:
    """Open the gencode H5 and return ``(handle, cache)``."""
    return h5py.File(h5_path, "r"), {}


def _fetch_row(
    handle: h5py.File,
    chunk_cache: dict[int, np.ndarray],
    chunk_idx: int,
    local_idx: int,
) -> np.ndarray:
    """Fetch one row of a gencode H5 chunk, caching the chunk array."""
    arr = chunk_cache.get(chunk_idx)
    if arr is None:
        arr = handle[f"X{chunk_idx}"][:]
        chunk_cache[chunk_idx] = arr
    return arr[local_idx]


def _patch_indel(
    x_row: np.ndarray,
    match: AnchorMatch,
    ref: str,
    alt: str,
    pos_hg38: int,
    fasta: Fasta,
) -> tuple[np.ndarray, str] | tuple[None, str]:
    """Build X_alt for an indel by patching the chunk + length adjustment.

    Args:
        x_row (np.ndarray): The ref chunk fetched from gencode_final_h5
            (length 15000, in gene reading orientation).
        match (AnchorMatch): Anchor info — gives strand, chunk_idx,
            variant_offset.
        ref (str): Reference allele (coding orientation).
        alt (str): Alternate allele (coding orientation).
        pos_hg38 (int): 1-based variant position on + strand.
        fasta (Fasta): pyfaidx handle for fetching extension bases.

    Returns:
        tuple[np.ndarray, str] | tuple[None, str]:
            On success: ``(x_alt, "ok")`` where ``x_alt`` has length 15000.
            On failure: ``(None, reason)`` where ``reason`` is one of
            ``ref_mismatch``, ``chrom_not_found``, ``pad_fetch_failed``.
    """
    strand = match.gene.strand
    voff = int(match.variant_offset)
    l_ref = len(ref)
    len(alt)
    fa_chrom = _resolve_fasta_chrom(fasta, match.gene.chrom)
    if fa_chrom is None:
        return None, "chrom_not_found"

    if strand == "+":
        expected_codes = [_NT_TO_INT[b] for b in ref.upper()]
        observed = x_row[voff : voff + l_ref].tolist()
        if observed != expected_codes:
            return None, "ref_mismatch"
        alt_codes = np.array(
            [_NT_TO_INT[b] for b in alt.upper()], dtype=np.int8
        )
        new_arr = np.concatenate([x_row[:voff], alt_codes, x_row[voff + l_ref:]])
    else:
        rc_ref = _reverse_complement(ref.upper())
        expected_codes = [_NT_TO_INT[b] for b in rc_ref]
        observed = x_row[voff - l_ref + 1 : voff + 1].tolist()
        if observed != expected_codes:
            return None, "ref_mismatch"
        rc_alt = _reverse_complement(alt.upper())
        alt_codes = np.array(
            [_NT_TO_INT[b] for b in rc_alt], dtype=np.int8
        )
        new_arr = np.concatenate(
            [x_row[: voff - l_ref + 1], alt_codes, x_row[voff + 1 :]]
        )

    cur_len = len(new_arr)
    if cur_len == TOTAL_LENGTH:
        return new_arr.astype(np.int8), "ok"

    if cur_len > TOTAL_LENGTH:
        return new_arr[:TOTAL_LENGTH].astype(np.int8), "ok"

    extra_needed = TOTAL_LENGTH - cur_len
    if strand == "+":
        chunk_last_pos_1b = pos_hg38 + (TOTAL_LENGTH - 1 - voff)
        start_0b = chunk_last_pos_1b
        end_0b = start_0b + extra_needed
        extra_seq = _slice_window(fasta, fa_chrom, start_0b, end_0b)
        if len(extra_seq) != extra_needed:
            return None, "pad_fetch_failed"
        extra_codes = _encode_to_int8(extra_seq)
        new_arr = np.concatenate([new_arr, extra_codes])
    else:
        chunk_first_pos_1b = pos_hg38 + voff - (TOTAL_LENGTH - 1)
        end_0b = chunk_first_pos_1b - 1
        start_0b = end_0b - extra_needed
        extra_plus = _slice_window(fasta, fa_chrom, start_0b, end_0b)
        if len(extra_plus) != extra_needed:
            return None, "pad_fetch_failed"
        extra_rc = _reverse_complement(extra_plus)
        extra_codes = _encode_to_int8(extra_rc)
        new_arr = np.concatenate([new_arr, extra_codes])

    assert len(new_arr) == TOTAL_LENGTH, f"length {len(new_arr)} != {TOTAL_LENGTH}"
    return new_arr.astype(np.int8), "ok"


def build(
    *,
    input_h5: str,
    output_h5: str,
    gencode_final_h5: str,
    gencode_splice_table: str,
    genome_fasta: str,
    sl: int,
    chrom_filter: set[str] | None,
    diagnostics_json: str | None,
) -> Path:
    """Run the allchroms + indel-aware gencode_anchored build.

    Args:
        input_h5 (str): Path to the variant-centric SpliceVarDB H5.
        output_h5 (str): Output H5 path.
        gencode_final_h5 (str): Gencode training H5 path (must match
            the chrom_filter at chunk-layout level).
        gencode_splice_table (str): Canonical splice table.
        genome_fasta (str): GRCh38 FASTA for fetching extension bases.
        sl (int): Sequence length per target block.
        chrom_filter (set[str] | None): Chromosomes to include. ``None``
            keeps every chromosome.
        diagnostics_json (str | None): JSON output path for build counts.

    Returns:
        Path: Output H5 path.

    Raises:
        ValueError: When no records survive the build.
    """
    logger.info("=== gencode_anchored_allchroms_indel build ===")
    logger.info("Input SpliceVarDB H5: %s", input_h5)
    logger.info("Gencode final H5:     %s", gencode_final_h5)
    logger.info("Splice table:         %s", gencode_splice_table)
    logger.info("Genome FASTA:         %s", genome_fasta)
    logger.info("Chrom filter:         %s", "ALL 24" if chrom_filter is None else sorted(chrom_filter))

    entries = load_canonical_table(
        gencode_splice_table, chrom_filter=chrom_filter, sl=sl
    )
    verify_chunk_sizes(entries, gencode_final_h5)
    genome_index = build_genome_index(entries)

    with h5py.File(gencode_final_h5, "r") as gh:
        x0 = gh["X0"]
        total_length = int(x0.shape[1])
        cl_max = total_length - sl

    if total_length != TOTAL_LENGTH:
        raise ValueError(
            f"Unexpected chunk length {total_length}; this builder "
            f"assumes {TOTAL_LENGTH}."
        )

    payload = _read_input(input_h5)
    n = int(payload["label"].shape[0])
    logger.info("Loaded %d input records", n)

    fasta = Fasta(genome_fasta, sequence_always_upper=True, as_raw=True)
    handle, chunk_cache = _open_chunks(gencode_final_h5)

    keep_idx: list[int] = []
    out_x_ref: list[np.ndarray] = []
    out_x_alt: list[np.ndarray] = []
    out_anchor: list[AnchorMatch] = []
    out_kind: list[str] = []

    counts: dict[str, int] = {
        "out_of_chrom": 0,
        "no_anchor": 0,
        "snv_ref_mismatch": 0,
        "indel_ref_mismatch": 0,
        "indel_chrom_not_found": 0,
        "indel_pad_fetch_failed": 0,
    }

    chroms = payload["chrom"]
    poses = payload["pos"]
    refs = payload["ref"]
    alts = payload["alt"]
    gsyms = payload["gene_symbol"]

    for i in range(n):
        chrom = str(chroms[i])
        if chrom_filter is not None and chrom not in chrom_filter:
            counts["out_of_chrom"] += 1
            continue
        ref_str = str(refs[i]).upper()
        alt_str = str(alts[i]).upper()
        pos_hg38 = int(poses[i])
        match = find_anchor(
            chrom=chrom,
            pos=pos_hg38,
            sl=sl,
            cl_max=cl_max,
            genome_index=genome_index,
            gene_symbol_hint=str(gsyms[i]),
        )
        if match is None:
            counts["no_anchor"] += 1
            continue

        x_row = _fetch_row(
            handle, chunk_cache, match.chunk_idx, match.local_idx
        ).copy()

        if len(ref_str) == 1 and len(alt_str) == 1:
            observed = int(x_row[match.variant_offset])
            expected_ref = encode_ref_allele(ref_str, match.gene.strand)
            if observed != expected_ref:
                counts["snv_ref_mismatch"] += 1
                continue
            alt_code = encode_alt_allele(alt_str, match.gene.strand)
            x_alt = x_row.copy()
            x_alt[match.variant_offset] = np.int8(alt_code)
            kind = "snv"
        else:
            x_alt, reason = _patch_indel(
                x_row=x_row,
                match=match,
                ref=ref_str,
                alt=alt_str,
                pos_hg38=pos_hg38,
                fasta=fasta,
            )
            if x_alt is None:
                if reason == "ref_mismatch":
                    counts["indel_ref_mismatch"] += 1
                elif reason == "chrom_not_found":
                    counts["indel_chrom_not_found"] += 1
                else:
                    counts["indel_pad_fetch_failed"] += 1
                continue
            kind = "indel"

        keep_idx.append(i)
        out_x_ref.append(x_row.astype(np.int8))
        out_x_alt.append(x_alt.astype(np.int8))
        out_anchor.append(match)
        out_kind.append(kind)

    handle.close()

    n_kept = len(keep_idx)
    n_snv_kept = sum(1 for k in out_kind if k == "snv")
    n_indel_kept = n_kept - n_snv_kept
    logger.info(
        "Anchor mapping done: kept=%d (snv=%d, indel=%d). counts=%s",
        n_kept, n_snv_kept, n_indel_kept, counts,
    )
    if n_kept == 0:
        raise ValueError("No records survived; nothing to write")

    keep_arr = np.array(keep_idx, dtype=np.int64)
    x_ref_arr = np.stack(out_x_ref).astype(np.int8)
    x_alt_arr = np.stack(out_x_alt).astype(np.int8)
    label_arr = payload["label"][keep_arr].astype(np.int8)
    pos_arr = payload["pos"][keep_arr].astype(np.uint64)

    str_keys = ("chrom", "ref", "alt", "gene_symbol", "gene_location", "gene_region", "strand")
    str_arrays = {k: payload[k][keep_arr].astype(object) for k in str_keys}
    kind_arr = np.array(out_kind, dtype=object)
    source_gene_name = np.array([m.gene.gene_name for m in out_anchor], dtype=object)
    source_chunk_idx = np.array([m.chunk_idx for m in out_anchor], dtype=np.uint32)
    source_local_idx = np.array([m.local_idx for m in out_anchor], dtype=np.uint32)
    source_strand = np.array([m.gene.strand for m in out_anchor], dtype=object)
    variant_offset = np.array([m.variant_offset for m in out_anchor], dtype=np.uint32)

    flank_length = cl_max // 2
    payload.get("__attrs__", {})
    attrs: dict[str, object] = {
        "build_mode": "gencode_anchored_allchroms_indel",
        "target_length": int(sl),
        "flank_length": int(flank_length),
        "total_length": int(total_length),
        "genome_build": "GRCh38",
        "strand_aware": True,
        "source": "SpliceVarDB",
        "build_timestamp_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "gencode_final_h5_path": str(gencode_final_h5),
        "gencode_splice_table_path": str(gencode_splice_table),
        "input_variant_centric_h5": str(input_h5),
        "genome_fasta_path": str(genome_fasta),
    }

    out_path = Path(output_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("X_ref", data=x_ref_arr, dtype="int8",
                          compression="gzip", compression_opts=4,
                          chunks=(1, total_length))
        h5.create_dataset("X_alt", data=x_alt_arr, dtype="int8",
                          compression="gzip", compression_opts=4,
                          chunks=(1, total_length))
        h5.create_dataset("label", data=label_arr, dtype="int8")
        h5.create_dataset("pos", data=pos_arr, dtype="uint64")
        for key, arr in str_arrays.items():
            h5.create_dataset(key, data=arr, dtype=string_dtype)
        h5.create_dataset("variant_kind", data=kind_arr, dtype=string_dtype)
        h5.create_dataset("source_gene_name", data=source_gene_name, dtype=string_dtype)
        h5.create_dataset("source_chunk_idx", data=source_chunk_idx, dtype="uint32")
        h5.create_dataset("source_local_idx", data=source_local_idx, dtype="uint32")
        h5.create_dataset("source_strand", data=source_strand, dtype=string_dtype)
        h5.create_dataset("variant_offset", data=variant_offset, dtype="uint32")
        for key, value in attrs.items():
            h5.attrs[key] = value

    logger.info("Wrote H5 to %s (rows=%d, snv=%d, indel=%d)",
                out_path, n_kept, n_snv_kept, n_indel_kept)

    if diagnostics_json:
        diag = {
            "input_h5": input_h5,
            "output_h5": str(out_path),
            "gencode_final_h5": gencode_final_h5,
            "gencode_splice_table": gencode_splice_table,
            "genome_fasta": genome_fasta,
            "sl": sl,
            "cl_max": cl_max,
            "total_length": total_length,
            "n_input": n,
            "n_kept": n_kept,
            "n_snv_kept": n_snv_kept,
            "n_indel_kept": n_indel_kept,
            **counts,
        }
        Path(diagnostics_json).parent.mkdir(parents=True, exist_ok=True)
        Path(diagnostics_json).write_text(json.dumps(diag, indent=2))
        logger.info("Saved diagnostics: %s", diagnostics_json)
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(prog="splicevardb_gencode_anchored_allchroms_indel_builder")
    p.add_argument("--input-h5", required=True)
    p.add_argument("--output-h5", required=True)
    p.add_argument("--gencode-final-h5",
                   default="./data/h5_cl10000/dataset_all_all.h5")
    p.add_argument("--gencode-splice-table",
                   default="./data/canonical_dataset_v49_regions.txt")
    p.add_argument("--genome-fasta",
                   default="./genome/GRCh38.fa")
    p.add_argument("--sl", type=int, default=DEFAULT_SL)
    p.add_argument("--no-chrom-filter", action="store_true",
                   help="Disable chrom_filter (include all 24 chroms).")
    p.add_argument("--train-chroms", nargs="+", default=list(DEFAULT_TRAIN_CHROMS),
                   help="Chrom_filter when --no-chrom-filter is not set.")
    p.add_argument("--diagnostics-json", default=None)
    p.add_argument("--logging-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.logging_level.upper(), logging.INFO),
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    chrom_filter = None if args.no_chrom_filter else set(args.train_chroms)
    try:
        build(
            input_h5=args.input_h5,
            output_h5=args.output_h5,
            gencode_final_h5=args.gencode_final_h5,
            gencode_splice_table=args.gencode_splice_table,
            genome_fasta=args.genome_fasta,
            sl=int(args.sl),
            chrom_filter=chrom_filter,
            diagnostics_json=args.diagnostics_json,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.exception("Build failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
