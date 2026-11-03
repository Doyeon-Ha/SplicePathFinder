"""Canonical acceptor / donor positions from the GENCODE v49 annotation.

The annotation table is the one the official SpliceAI runs consume
(``spliceai_annotation_v49.txt``), so a score built on these sites and a
SpliceAI score built with ``-M 1`` are gated on the same set of sites.

Position convention is settled empirically against
``coord_splice_fields_seed42.npz`` rather than assumed: that file supplies the
``intron_nt`` / ``exon_nt`` / ``akind`` distances the Table-9 strata are cut
from, so a track that disagrees with it would mask on one set of coordinates
while the strata report on another.
"""

from __future__ import annotations

import numpy as np

ANNOTATION = ("./spliceai_train/reference/"
              "spliceai_annotation_v49.txt")


def load_canonical_sites(path: str = ANNOTATION) -> dict[str, dict]:
    """Read the annotation table into per-gene splice-site coordinates.

    Exon boundaries are converted to the two candidate anchor bases for each
    site: the boundary exonic base and the first intronic base. Which one the
    caller wants is a convention choice resolved by ``validate_convention``.

    Args:
        path (str): annotation TSV with the SpliceAI 7-column layout.

    Returns:
        dict[str, dict]: gene name -> ``{chrom, strand, tx_start, tx_end,
            acc_exonic, acc_intronic, don_exonic, don_intronic}``. ``tx_start``
            is the 0-based boundary immediately before the first transcript
            base; ``tx_end`` is the 1-based inclusive final base. Site arrays
            are 1-based genomic positions sorted ascending.
    """
    genes: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            (name, chrom, strand, tx_s, tx_e,
             ex_s, ex_e) = line.rstrip("\n").split("\t")
            starts = np.array([int(v) for v in ex_s.rstrip(",").split(",")])
            ends = np.array([int(v) for v in ex_e.rstrip(",").split(",")])
            if len(starts) < 2:
                continue
            order = np.argsort(starts)
            starts, ends = starts[order], ends[order]
            # Genomic-left boundary of each intron: last exonic base = ends[k],
            # first intronic base = ends[k] + 1. Genomic-right boundary: last
            # intronic base = starts[k+1], first exonic base = starts[k+1] + 1.
            left_exonic, left_intronic = ends[:-1], ends[:-1] + 1
            right_intronic, right_exonic = starts[1:], starts[1:] + 1
            if strand == "+":
                don_ex, don_in = left_exonic, left_intronic
                acc_in, acc_ex = right_intronic, right_exonic
            else:
                # Transcribed right-to-left: the genomic-right boundary is the
                # donor and the genomic-left boundary is the acceptor.
                don_ex, don_in = right_exonic, right_intronic
                acc_in, acc_ex = left_intronic, left_exonic
            genes[name] = {
                "chrom": chrom, "strand": strand,
                "tx_start": int(tx_s), "tx_end": int(tx_e),
                "acc_exonic": np.sort(acc_ex), "acc_intronic": np.sort(acc_in),
                "don_exonic": np.sort(don_ex), "don_intronic": np.sort(don_in),
            }
    return genes


def load_sites_by_accession(path: str) -> dict[str, dict]:
    """Read a transcript-keyed annotation into per-accession splice sites.

    The gene-keyed table carries one representative transcript per gene, so a
    clinical report written against a different RefSeq transcript can name a
    junction the table does not have -- masking then deletes that junction's
    loss with no error. Keying on the accession lets the mask follow the
    transcript the variant was actually interpreted against.

    Args:
        path (str): annotation TSV with the SpliceAI 7-column layout, whose
            NAME column holds a versioned accession (``NM_000014.6``).

    Returns:
        dict[str, dict]: unversioned accession -> ``{chrom, strand,
            acc_exonic, don_exonic}``, each site array 1-based and ascending.
    """
    out: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            name, chrom, strand, _ts, _te, ex_s, ex_e = line.rstrip("\n").split("\t")
            st = np.array([int(v) for v in ex_s.rstrip(",").split(",") if v])
            en = np.array([int(v) for v in ex_e.rstrip(",").split(",") if v])
            if len(st) < 2:
                continue
            order = np.argsort(st)
            st, en = st[order], en[order]
            # ⚠ 이 파일의 EXON_START는 **첫 엑손 염기**(1-based)라 +1을 더하지
            # 않는다. 유전자 키 `spliceai_annotation_v49.txt`는 반대로 마지막
            # 인트론 염기라 +1이 필요하다 -- 두 파일의 규약이 다르다. 실측으로
            # 확정했다(사내 canonical SNV 2,044건: exon_start 유래인 + acceptor
            # 398건과 - donor 652건만 정확히 +1 어긋났다).
            if strand == "+":
                don_ex, acc_ex = en[:-1], st[1:]
            else:
                don_ex, acc_ex = st[1:], en[:-1]
            out[name.split(".")[0]] = {
                "chrom": chrom, "strand": strand,
                "acc_exonic": np.sort(acc_ex), "don_exonic": np.sort(don_ex)}
    return out


# The model puts its probability peak on the boundary EXONIC base, not the
# adjacent intronic one -- measured 48/48 for both channels, with the intronic
# candidate off by exactly one (scripts/probe_site_convention.py, job 23857).
# Note this is one base away from the strata coordinates, where
# ``intron_nt = 1`` is the last intronic base.
ANCHOR = {"acc": "acc_exonic", "don": "don_exonic"}


def site_indices(
    gene: dict, gstart: int, length: int, minus: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a gene's canonical sites into chunk-local model coordinates.

    Args:
        gene (dict): one entry from :func:`load_canonical_sites`.
        gstart (int): 1-based genomic position of chunk-local index 0 on the
            + strand (``variant_pos - chunk_local_variant_pos``).
        length (int): chunk length.
        minus (bool): True when the chunk was reverse-complemented, i.e. the
            anchoring gene is on the - strand.

    Returns:
        tuple[np.ndarray, np.ndarray]: acceptor and donor indices inside
            ``[0, length)``, sorted.
    """
    out = []
    for kind in ("acc", "don"):
        off = gene[ANCHOR[kind]].astype(np.int64) - int(gstart)
        idx = (length - 1 - off) if minus else off
        out.append(np.sort(idx[(idx >= 0) & (idx < length)]))
    return out[0], out[1]


def annotation_probs(
    acc_idx: np.ndarray, don_idx: np.ndarray, length: int,
) -> np.ndarray:
    """Build the 0/1 reference readout that stands in for the model's ref pass.

    Args:
        acc_idx (np.ndarray): acceptor indices from :func:`site_indices`.
        don_idx (np.ndarray): donor indices.
        length (int): chunk length.

    Returns:
        np.ndarray: ``(length, 3)`` float32 in the scorer's ``(acc, don, nei)``
            channel order, 1.0 at an annotated site and ``nei`` taking the
            remainder so the three channels still sum to one.
    """
    probs = np.zeros((length, 3), dtype=np.float32)
    probs[acc_idx, 0] = 1.0
    probs[don_idx, 1] = 1.0
    probs[:, 2] = 1.0 - probs[:, 0] - probs[:, 1]
    return probs


def apply_canonical_mask(
    tracks: np.ndarray, acc_idx: np.ndarray, don_idx: np.ndarray,
) -> np.ndarray:
    """Zero the directions a canonical site cannot produce, per channel.

    A loss can only happen where the reference already has a site of that
    type, and a gain only where it does not. The rule is per channel, so an
    annotated acceptor position may still carry a donor gain (and vice versa).

    Args:
        tracks (np.ndarray): ``(L, 4)`` in order
            ``(acc_gain, acc_loss, don_gain, don_loss)``.
        acc_idx (np.ndarray): annotated acceptor indices.
        don_idx (np.ndarray): annotated donor indices.

    Returns:
        np.ndarray: masked copy of ``tracks``.
    """
    out = tracks.copy()
    is_acc = np.zeros(len(out), dtype=bool)
    is_don = np.zeros(len(out), dtype=bool)
    is_acc[acc_idx] = True
    is_don[don_idx] = True
    out[is_acc, 0] = 0.0        # acceptor gain at an annotated acceptor
    out[~is_acc, 1] = 0.0       # acceptor loss where no acceptor is annotated
    out[is_don, 2] = 0.0        # donor gain at an annotated donor
    out[~is_don, 3] = 0.0       # donor loss where no donor is annotated
    return out


def load_variant_transcripts(path: str) -> dict[str, str]:
    """Read a ``cpra_grch38,transcript`` table into a lookup.

    Args:
        path (str): CSV with at least those two columns.

    Returns:
        dict[str, str]: ``chrom-pos-ref-alt`` (no ``chr`` prefix) -> unversioned
            accession.
    """
    import csv

    out: dict[str, str] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["cpra_grch38"]] = row["transcript"].split(".")[0]
    return out


def resolve_sites(
    genes: dict[str, dict],
    by_tx: dict[str, dict],
    var_tx: dict[str, str],
    *,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    gene: str,
) -> tuple[dict, bool]:
    """Pick the splice sites to mask against for one variant.

    The gene-keyed table carries one representative transcript per gene, so a
    variant interpreted against a different RefSeq transcript can name a
    junction that table does not have -- masking then deletes that junction's
    loss with no error. Prefer the variant's own transcript and fall back to
    the gene representative only when it is unavailable.

    Args:
        genes (dict[str, dict]): gene-keyed sites from
            :func:`load_canonical_sites`.
        by_tx (dict[str, dict]): accession-keyed sites from
            :func:`load_sites_by_accession`; empty disables the preference.
        var_tx (dict[str, str]): variant key -> accession.
        chrom (str): contig, with or without the ``chr`` prefix.
        pos (int): 1-based position.
        ref (str): reference allele.
        alt (str): alternate allele.
        gene (str): anchoring gene name, used for the fallback.

    Returns:
        tuple[dict, bool]: the site record, and True when it came from the
            variant's own transcript.
    """
    c = chrom[3:] if chrom.startswith("chr") else chrom
    rec = by_tx.get(var_tx.get(f"{c}-{pos}-{ref}-{alt}", ""))
    if rec is None:
        return genes[gene], False
    return rec, True
