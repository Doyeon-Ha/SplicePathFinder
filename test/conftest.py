"""Shared pytest fixtures for the splicing test suite.

Fixtures defined here are auto-discovered by pytest and available to all
test modules under ``test/``. Feature-specific fixtures should live in
module-local conftest files when they do not benefit the whole suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from process_input.splicevardb_loader import EXCEL_SKIPROWS

if TYPE_CHECKING:
    from pyfaidx import Fasta


@pytest.fixture
def toy_splicevardb_records() -> list[dict[str, Any]]:
    """Return a small handcrafted variant record set for SpliceVarDB tests.

    Records use the **real** Excel column names (``CHROM``, ``POS``,
    ``REF``, ``ALT``, ``Gene_Symbol``, ``Gene_Location``, ``Gene_Region``,
    ``Outcome``) so that fixtures exercise the production alias path.

    Returns:
        list[dict[str, Any]]: Raw record dicts keyed by the expected
        SpliceVarDB column names.
    """
    return [
        {
            "Dataset": "Train",
            "CHROM": "1",
            "POS": 11_000,
            "ID": ".",
            "REF": "A",
            "ALT": "G",
            "DOI": "10.0/toy",
            "Gene_Symbol": "GENE1",
            "Outcome": "SAV",
            "Gene_Location": "intronic",
            "Gene_Region": "acceptor_region",
        },
        {
            "Dataset": "Train",
            "CHROM": "chr1",
            "POS": 12_050,
            "ID": ".",
            "REF": "C",
            "ALT": "T",
            "DOI": "10.0/toy",
            "Gene_Symbol": "GENE1",
            "Outcome": "nonSAV",
            "Gene_Location": "exonic",
            "Gene_Region": ".",
        },
        {
            "Dataset": "Test",
            "CHROM": "2",
            "POS": 5_000,
            "ID": ".",
            "REF": "G",
            "ALT": "GA",
            "DOI": "10.0/toy",
            "Gene_Symbol": "GENE2",
            "Outcome": 1,
            "Gene_Location": "intronic",
            "Gene_Region": "branchpoint",
        },
        {
            "Dataset": "Test",
            "CHROM": "2",
            "POS": 5_100,
            "ID": ".",
            "REF": "TT",
            "ALT": "AA",
            "DOI": "10.0/toy",
            "Gene_Symbol": "GENE2",
            "Outcome": 0,
            "Gene_Location": "exonic",
            "Gene_Region": "donor_region",
        },
        {
            "Dataset": "Train",
            "CHROM": "X",
            "POS": 200,
            "ID": ".",
            "REF": "T",
            "ALT": "C",
            "DOI": "10.0/toy",
            "Gene_Symbol": "GENEX",
            "Outcome": "SAV",
            "Gene_Location": "intronic",
            "Gene_Region": "acceptor_region",
        },
    ]


@pytest.fixture
def toy_gene_strand() -> dict[str, str]:
    """Return a gene-strand lookup matching the toy records.

    ``GENE1`` and ``GENEX`` are plus strand; ``GENE2`` is minus strand.
    """
    return {"GENE1": "+", "GENE2": "-", "GENEX": "+"}


def _write_toy_excel(records: list[dict[str, Any]], destination: Path) -> Path:
    """Serialize records to a SpliceVarDB-shaped Excel at ``destination``.

    The real supplementary Excel prefixes the data with three title/meta
    rows. This helper mimics that layout so that ``skiprows=3`` aligns.

    Args:
        records (list[dict[str, Any]]): Records using real column names.
        destination (Path): Target ``.xlsx`` path.

    Returns:
        Path: ``destination``.
    """
    df = pd.DataFrame(records)
    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        # Emit EXCEL_SKIPROWS leading blank/meta rows then the header+data.
        pd.DataFrame([[""]]).to_excel(writer, index=False, header=False, startrow=0)
        df.to_excel(writer, index=False, startrow=EXCEL_SKIPROWS)
    return destination


@pytest.fixture
def toy_splicevardb_excel(
    tmp_path: Path,
    toy_splicevardb_records: list[dict[str, Any]],
) -> Path:
    """Write the toy SpliceVarDB record set to a temporary .xlsx file.

    Args:
        tmp_path (Path): Pytest-provided temporary directory fixture.
        toy_splicevardb_records (list[dict[str, Any]]): Records generated
            by :func:`toy_splicevardb_records`.

    Returns:
        Path: Absolute path to the generated Excel file.
    """
    path = tmp_path / "splicevardb_toy.xlsx"
    return _write_toy_excel(toy_splicevardb_records, path)


@pytest.fixture
def toy_fasta(tmp_path: Path) -> "Fasta":
    """Create a tiny GRCh38-shaped FASTA and return a :class:`Fasta` handle.

    Args:
        tmp_path (Path): Pytest-provided temporary directory fixture.

    Returns:
        Fasta: A ``pyfaidx.Fasta`` handle pointing at the generated file.
    """
    from pyfaidx import Fasta

    chromosomes: dict[str, str] = {
        "chr1": ("ACGT" * 2_500)[:10_000],
        "chr2": ("CAGT" * 2_500)[:10_000],
        "chrX": ("GCTA" * 2_500)[:10_000],
    }

    fasta_path = tmp_path / "toy_grch38.fa"
    with fasta_path.open("w", encoding="utf-8") as handle:
        for chrom, sequence in chromosomes.items():
            handle.write(f">{chrom}\n")
            handle.write(sequence + "\n")

    return Fasta(str(fasta_path), sequence_always_upper=True)


@pytest.fixture
def mock_chain_lookup() -> MagicMock:
    """Return a MagicMock standing in for :class:`liftover.ChainFile`."""
    chain = MagicMock(name="ChainFile")
    table: dict[tuple[str, int], list[tuple[str, int, str]]] = {}

    def _query(chrom: str, pos: int) -> list[tuple[str, int, str]]:
        return list(table.get((chrom, pos), []))

    chain.query.side_effect = _query
    chain.register_hit = lambda chrom, pos, hits: table.update({(chrom, pos): hits})
    chain.table = table
    return chain
