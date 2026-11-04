"""Select the manuscript cohort from a raw ClinVar variant table.

The supplied manifest freezes membership, order and annotation anchor metadata.
It contains public ClinVar/GENCODE metadata, not sequences, predictions or weights.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import h5py
import numpy as np
from process_input.variant_identity import decode, variant_keys, row_digest


def prepare(input_h5, manifest, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(manifest, 'rt') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    wanted = np.asarray([r['key'] for r in rows], dtype=object)
    if len(set(wanted)) != len(wanted):
        raise ValueError('Cohort manifest contains duplicate identities')
    with h5py.File(input_h5) as f:
        raw_keys = variant_keys(f)
        if len(set(raw_keys)) != len(raw_keys):
            raise ValueError('Raw variant table contains duplicate identities')
        where = {k: i for i, k in enumerate(raw_keys)}
        missing = [k for k in wanted if k not in where]
        if missing:
            raise ValueError(f'Raw ClinVar input is missing {len(missing)} cohort variants')
        take = np.asarray([where[k] for k in wanted])
        fields = {k: f[k][:][take] for k in
                  ['chrom', 'pos', 'ref', 'alt', 'label', 'gene_symbol']}
        for name, expected in [('label', np.asarray([int(r['label']) for r in rows])),
                               ('gene_symbol', np.asarray([r['gene_symbol'] for r in rows], dtype=object))]:
            actual = decode(fields[name]) if name == 'gene_symbol' else fields[name]
            if not np.array_equal(actual, expected):
                raise ValueError(f'Raw ClinVar {name} differs from the frozen cohort')
    string = h5py.string_dtype('utf-8')

    def write(path, indices, extra):
        with h5py.File(path, 'x') as f:
            for k, v in fields.items():
                v = v[indices]
                f.create_dataset(k, data=v, dtype=string if v.dtype.kind in 'OSU' else v.dtype)
            f.create_dataset('key', data=wanted[indices], dtype=string)
            ref, alt = decode(fields['ref'][indices]), decode(fields['alt'][indices])
            f.create_dataset('variant_kind', data=np.asarray([
                'snv' if len(r) == len(a) == 1 else 'indel' for r, a in zip(ref, alt)
            ], dtype=object), dtype=string)
            for k, v in extra.items():
                f.create_dataset(k, data=v, dtype=string if v.dtype.kind in 'OSU' else v.dtype)
            f.attrs['row_key_sha256'] = row_digest(wanted[indices])
            f.attrs['cohort_manifest'] = str(manifest)

    release = out_dir / 'release.h5'
    write(release, np.arange(len(rows)), {
        'legacy_splice': np.asarray([bool(int(r['legacy_splice'])) for r in rows])})
    vrow = np.asarray([int(r['v50_row']) for r in rows])
    keep = np.flatnonzero(vrow >= 0)
    keep = keep[np.argsort(vrow[keep])]
    if not np.array_equal(vrow[keep], np.arange(len(keep))):
        raise ValueError('V50 anchor row indices are not a complete contiguous order')
    extra = {k: np.asarray([rows[i][field] for i in keep], dtype=object)
             for k, field in [('source_gene_name', 'v50_gene'),
                              ('source_strand', 'v50_strand'), ('strand', 'v50_strand')]}
    for k, field in [('variant_offset', 'v50_offset'), ('source_chunk_idx', 'v50_chunk_idx'),
                     ('source_local_idx', 'v50_local_idx')]:
        extra[k] = np.asarray([int(rows[i][field]) for i in keep], dtype=np.int64)
    write(out_dir / 'v50_anchors.h5', keep, extra)
    report = dict(rows=len(rows), v50_rows=len(keep), row_key_sha256=row_digest(wanted),
                  raw_variant_h5=str(input_h5), cohort_manifest=str(manifest))
    (out_dir / 'cohort.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-h5', required=True)
    p.add_argument('--cohort-manifest', required=True)
    p.add_argument('--out-dir', required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.input_h5, a.cohort_manifest, a.out_dir), indent=2))


if __name__ == '__main__':
    main()
