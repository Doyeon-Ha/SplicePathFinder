"""Freeze the v50 nonparalog annotation split and audit reusable native scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

NAS = Path('.')
V50 = NAS / 'data/h5_cl10000_sl5000_s1250_regions_realflank_v50mane'
REFERENCE = NAS / 'data/h5_cl10000_sl5000_s1250_regions_realflank/dataset_test_0.h5'
CHROMS = {'chr1', 'chr3', 'chr5', 'chr7', 'chr9'}


def fingerprint(x, y):
    h = hashlib.sha256(x.tobytes())
    h.update(y.tobytes())
    return h.digest()


def suffixes(f):
    return sorted((int(k[1:]) for k in f if k.startswith('X') and k[1:].isdigit()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    dataset = out / 'dataset_test_0.h5'
    assert not dataset.exists(), dataset
    flags = dict(line.rstrip().split('\t') for line in
                 (NAS / 'paralog_info/grch38_paralog_flags.tsv').read_text().splitlines()[1:])
    xs, ys, genes, source_rows, row_genes, strands = [], [], [], [], [], []
    with h5py.File(V50 / 'datafile_test_all.h5') as meta, h5py.File(V50 / 'dataset_test_all.h5') as src:
        names = meta['NAME'][:].astype(str)
        chroms = meta['CHROM'][:].astype(str)
        paralogs = meta['PARALOG'][:]
        starts = meta['TX_START'][:].astype(np.int64)
        ends = meta['TX_END'][:].astype(np.int64)
        gene_strands = meta['STRAND'][:].astype(str)
        assert set(chroms) == CHROMS
        assert len(set(names)) == len(names)
        # The canonical builder conservatively flags unlisted gene names as 1.
        assert all(int(flags.get(n, 1)) == int(p) for n, p in zip(names, paralogs))
        missing_flags = [n for n in names if n not in flags]
        sizes = (np.maximum(ends - starts + 1 - 5000, 0) + 1249) // 1250 + 1
        assert set(np.unique(paralogs)) == {0, 1}
        offset = 0
        for i in suffixes(src):
            lo, hi = i * 100, min((i + 1) * 100, len(names))
            if i == max(suffixes(src)):
                hi = len(names)
            assert src[f'X{i}'].shape == (int(sizes[lo:hi].sum()), 15000)
            assert src[f'Y{i}'].shape == (int(sizes[lo:hi].sum()), 5000)
            local = 0
            for g in range(lo, hi):
                n = int(sizes[g])
                if paralogs[g] == 0:
                    x = src[f'X{i}'][local:local+n]
                    y = src[f'Y{i}'][local:local+n]
                    assert x.dtype == y.dtype == np.int8
                    assert np.isin(x, [0, 1, 2, 3, 4]).all()
                    assert np.isin(y, [-1, 0, 1, 2]).all()
                    # Check exon-boundary labels directly in transcript coordinates.
                    def positions(key):
                        return np.asarray([int(v) for v in meta[key][g, 0].decode().split(',') if v])
                    left, right = positions('JN_START'), positions('JN_END')
                    if gene_strands[g] == '+':
                        acc, don = right - starts[g], left - starts[g]
                    else:
                        acc, don = ends[g] - left, ends[g] - right
                    for w in range(n):
                        valid_n = min(5000, int(ends[g] - starts[g] + 1) - w*1250)
                        expected = np.full(5000, -1, np.int8)
                        expected[:valid_n] = 0
                        for sites, label in [(don, 2), (acc, 1)]:
                            rel = sites - w*1250
                            expected[rel[(rel >= 0) & (rel < valid_n)]] = label
                        np.testing.assert_array_equal(y[w], expected)
                    genes.append({'gene': names[g], 'chrom': chroms[g], 'strand': gene_strands[g],
                                  'paralog': 0, 'windows': n, 'source_gene_index': g})
                    xs.append(x); ys.append(y)
                    source_rows.extend(range(offset+local, offset+local+n))
                    row_genes.extend([names[g]]*n); strands.extend([gene_strands[g]]*n)
                local += n
            offset += local
        assert offset == int(sizes.sum())
    x = np.concatenate(xs); y = np.concatenate(ys)
    del xs, ys
    with h5py.File(dataset, 'w') as f:
        for i, start in enumerate(range(0, len(x), 1024)):
            f.create_dataset(f'X{i}', data=x[start:start+1024])
            f.create_dataset(f'Y{i}', data=y[start:start+1024])
        f.attrs.update(annotation='GENCODE v50', transcript_policy='MANE Select > MANE Plus Clinical > Ensembl canonical',
                       paralog_filter=0, source_h5=str(V50/'dataset_test_all.h5'), stride=1250,
                       source_datafile=str(V50/'datafile_test_all.h5'), flank_fill='real')
    old_rows = {}; old_counts = []; old_digest = hashlib.sha256()
    with h5py.File(REFERENCE) as f:
        base = 0
        for i in suffixes(f):
            ox, oy = f[f'X{i}'][:], f[f'Y{i}'][:]
            old_counts.extend((oy >= 0).sum(axis=1).tolist())
            for j in range(len(ox)):
                key = fingerprint(ox[j], oy[j])
                old_rows.setdefault(key, base+j)
                old_digest.update(key)
            base += len(ox)
    matched = np.asarray([old_rows.get(fingerprint(a, b), -1) for a, b in zip(x, y)], np.int64)
    missing = np.flatnonzero(matched < 0)
    pool = np.flatnonzero(matched >= 0)
    controls = pool[np.linspace(0, len(pool)-1, min(32, len(pool)), dtype=int)]
    infer_rows = np.concatenate([missing, controls])
    with h5py.File(out/'native_inference.h5', 'w') as f:
        f.create_dataset('X0', data=x[infer_rows])
        f.create_dataset('Y0', data=y[infer_rows])
        f.create_dataset('row_index', data=infer_rows)
    with h5py.File(out/'splicepath_preflight.h5', 'w') as f:
        f.create_dataset('X0', data=x[:32])
        f.create_dataset('Y0', data=y[:32])
    np.savez(out/'mapping.npz', source_rows=source_rows, reference_rows=matched,
             reference_valid_counts=old_counts, inference_rows=infer_rows, control_rows=controls,
             valid_counts=(y >= 0).sum(axis=1), gene=np.asarray(row_genes), strand=np.asarray(strands))
    report = {'annotation':'GENCODE v50', 'source_h5':str(V50/'dataset_test_all.h5'),
              'source_datafile':str(V50/'datafile_test_all.h5'), 'reference_h5':str(REFERENCE),
              'dataset':str(dataset), 'chromosomes':sorted(CHROMS), 'paralog_filter':0,
              'genes':len(genes), 'unlisted_paralog_genes_excluded':missing_flags,
              'windows':len(x), 'valid_positions':int((y >= 0).sum()),
              'acceptors':int((y == 1).sum()), 'donors':int((y == 2).sum()),
              'exact_reference_matches':int((matched >= 0).sum()), 'native_inference_windows':len(missing),
              'native_control_windows':len(controls), 'labels_checked_against_gene_junctions':True,
              'input_and_label_sha256':hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),
              'reference_row_fingerprint_sha256':old_digest.hexdigest()}
    (out/'genes.json').write_text(json.dumps(genes,indent=2)+'\n')
    (out/'dataset.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2), flush=True)


if __name__ == '__main__':
    main()
