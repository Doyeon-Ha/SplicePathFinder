"""Write the SpliceVarDB early-stopping table: the full set minus ClinVar.

Training selects its stopping epoch on this table, so no variant in it may also
sit in the ClinVar test set -- otherwise the selection leaks into the test
through the shared variants. Rows whose ``(chrom, pos, ref, alt)`` appears in
any of the ClinVar tables are dropped; every other dataset row is copied as is,
so the output keeps the gencode-anchored schema the cache builder reads.

    python -m scripts.build_svdb_validation_h5 --out-h5 .../splicevardb_val.h5

``--head N`` keeps only the first N retained rows, for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

_NAS = Path("./data/eval")
SVDB_H5 = _NAS / "splicevardb_full_sa_normal_gencode_anchored.h5"
# The full 2026 release and the splice-region universe scored from it.
CLINVAR_H5S = [
    _NAS
    / "clinvar_20260628_gencode_anchored_indel"
    / "clinvar_20260628_gencode_anchored_indel.h5",
    _NAS / "clinvar_splice_region_table" / "clinvar_splice_region_106504.h5",
]


def _text(values: np.ndarray) -> list[str]:
    """Decode an h5py string column."""
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def variant_keys(path: Path) -> list[tuple[str, int, str, str]]:
    """Return ``(chrom, pos, ref, alt)`` per row of a variant table."""
    with h5py.File(path, "r") as f:
        chrom, ref, alt = (_text(f[c][:]) for c in ("chrom", "ref", "alt"))
        pos = f["pos"][:].astype(np.int64).tolist()
    return list(zip(chrom, pos, ref, alt))


def build(
    svdb_h5: Path, clinvar_h5s: list[Path], out_h5: Path, head: int | None
) -> dict:
    """Copy the SpliceVarDB rows absent from every ClinVar table."""
    clinvar = set()
    for path in clinvar_h5s:
        clinvar.update(variant_keys(path))
    keys = variant_keys(svdb_h5)
    keep = np.array([k not in clinvar for k in keys])
    n_in_clinvar = int(np.count_nonzero(~keep))
    if head is not None:
        keep &= np.cumsum(keep) <= head
    rows = np.flatnonzero(keep)

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(svdb_h5, "r") as src, h5py.File(out_h5, "w") as dst:
        for name, value in src.attrs.items():
            dst.attrs[name] = value
        for name, ds in src.items():
            if not isinstance(ds, h5py.Dataset) or ds.shape[:1] != (len(keys),):
                raise ValueError(f"{svdb_h5}:{name} is not a per-variant dataset")
            dst.create_dataset(name, data=ds[:][rows], dtype=ds.dtype)
        labels = dst["label"][:]
        dst.attrs["source_h5"] = str(svdb_h5)
        dst.attrs["excluded_if_in"] = json.dumps([str(p) for p in clinvar_h5s])

    summary = {
        "source_rows": len(keys),
        "in_clinvar": n_in_clinvar,
        "rows_written": int(len(rows)),
        "positives": int(labels.sum()),
        "head": head,
        "out_h5": str(out_h5),
    }
    logger.info("%s", summary)
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--svdb-h5", type=Path, default=SVDB_H5)
    parser.add_argument("--clinvar-h5", type=Path, nargs="+", default=CLINVAR_H5S)
    parser.add_argument("--out-h5", type=Path, required=True)
    parser.add_argument("--head", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s"
    )
    summary = build(args.svdb_h5, args.clinvar_h5, args.out_h5, args.head)
    args.out_h5.with_suffix(".json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
