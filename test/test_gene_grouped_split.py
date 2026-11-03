"""gene_grouped_split 단위 테스트 (spec 015 US2)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from process_input.gene_grouped_split import (
    assert_no_gene_leakage,
    gene_grouped_split,
)


def _synthetic(n_genes: int = 60, per_gene: int = 5):
    genes = np.repeat([f"GENE{g}" for g in range(n_genes)], per_gene)
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=genes.shape[0])
    return genes, labels


def test_no_gene_leakage() -> None:
    genes, labels = _synthetic()
    tr, va, te = gene_grouped_split(genes, labels, seed=42)
    assert_no_gene_leakage(genes, tr, va, te)
    assert len(tr) + len(va) + len(te) == len(genes)


def test_split_reproducible() -> None:
    genes, labels = _synthetic()
    a = gene_grouped_split(genes, labels, seed=42)
    b = gene_grouped_split(genes, labels, seed=42)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_different_seed_differs() -> None:
    genes, labels = _synthetic()
    a = gene_grouped_split(genes, labels, seed=1)
    b = gene_grouped_split(genes, labels, seed=2)
    assert not np.array_equal(a[2], b[2])


def test_missing_gene_excluded() -> None:
    genes, labels = _synthetic(n_genes=40)
    genes = genes.astype(object)
    genes[:5] = ""
    tr, va, te = gene_grouped_split(genes, labels, seed=42)
    assert len(tr) + len(va) + len(te) == len(genes) - 5


def test_bad_ratios_raise() -> None:
    genes, labels = _synthetic(n_genes=10)
    with pytest.raises(ValueError):
        gene_grouped_split(genes, labels, seed=0, ratios=(0.5, 0.4, 0.4))
