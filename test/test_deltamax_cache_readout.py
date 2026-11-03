"""The default cache readout scores the full 8,750-position head frame."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.deltamax_from_multichunk_cache import main


def _inputs(tmp_path: Path, *, masked: bool = False) -> tuple[Path, Path]:
    cache = tmp_path / "cache.h5"
    source = tmp_path / "variants.h5"
    ref = np.zeros((1, 6, 3), np.float16)
    alt = ref.copy()
    alt[0, 3, 0] = 0.1  # variant position
    alt[0, 5, 0] = 0.7  # outside the old +/-1 crop
    with h5py.File(cache, "w") as f:
        f.create_dataset("ref_probs", data=ref)
        f.create_dataset("alt_probs", data=alt)
        f.attrs["window_left"] = 3
        f.attrs["window_right"] = 3
        f.attrs["mask_canonical"] = masked
        f.attrs["mask_application"] = "premerge" if masked else "none"
    with h5py.File(source, "w") as f:
        for key, value in (("chrom", b"1"), ("ref", b"A"),
                           ("alt", b"G"), ("gene_symbol", b"GENE")):
            f.create_dataset(key, data=np.array([value]))
        f.create_dataset("pos", data=np.array([1000]))
    return cache, source


def _score(monkeypatch: pytest.MonkeyPatch, cache: Path, source: Path,
           out: Path, *flags: str) -> np.lib.npyio.NpzFile:
    monkeypatch.setattr(sys, "argv", ["deltamax_from_multichunk_cache.py",
                                      "--cache", str(cache), "--source-h5",
                                      str(source), "--out", str(out), *flags])
    main()
    return np.load(out)


def test_default_reads_full_even_length_window(monkeypatch, tmp_path) -> None:
    """The rightmost cached position has offset +2 in a [-3,+3) frame."""
    cache, source = _inputs(tmp_path)
    with _score(monkeypatch, cache, source, tmp_path / "full.npz") as full:
        assert full["delta_max"][0] == pytest.approx(0.7, abs=1e-3)
        assert int(full["offset"][0]) == 2
        assert int(full["window_left"]) == 3
        assert int(full["window_right"]) == 3
    with _score(monkeypatch, cache, source, tmp_path / "old.npz",
                "--window-bp", "1") as old:
        assert old["delta_max"][0] == pytest.approx(0.1, abs=1e-3)


def test_mask_requires_premerge_cache(monkeypatch, tmp_path) -> None:
    """A raw cache cannot claim exact masked delta max after merging."""
    cache, source = _inputs(tmp_path)
    with pytest.raises(SystemExit, match="build a separate cache"):
        _score(monkeypatch, cache, source, tmp_path / "bad.npz",
               "--mask-canonical")
    masked_cache, source = _inputs(tmp_path, masked=True)
    with _score(monkeypatch, masked_cache, source, tmp_path / "ok.npz",
                "--mask-canonical") as result:
        assert bool(result["masked"])
        assert str(result["mask_application"]) == "premerge"
