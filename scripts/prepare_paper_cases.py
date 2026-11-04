"""Prepare and plot the two manuscript cases using frozen ClinVar coordinates."""
from __future__ import annotations
import argparse
import gzip
import json
import h5py
import numpy as np

from pathlib import Path
import argparse
NAS = Path('.')
OUT = Path('outputs/cases')
TABLE = Path('data/canonical_dataset_v50_mane_regions.txt')
ARCHIVE = Path('data/clinvar_archive/clinvar_20260628.vcf.gz')
P = {'tag': 'v50mane_ep10'}


CASES = {'13096': ('RB1', 'c.2490-1398A>G'), '2854936': ('BARD1', 'c.1314+3469G>T')}


def prepare():
    from process_input.gencode_anchor import load_canonical_table, build_genome_index, find_anchor
    archive = ARCHIVE
    found = {}
    with gzip.open(archive, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            col = line.rstrip().split('\t')
            if col[2] in CASES:
                found[col[2]] = col
            if len(found) == len(CASES):
                break
    assert set(found) == set(CASES), found.keys()
    annotation = str(TABLE)
    idx = build_genome_index(load_canonical_table(annotation, chrom_filter=None, sl=5000))
    rows = []
    for vid, (gene, hgvs) in CASES.items():
        col = found[vid]
        chrom, pos, ref, alt = 'chr' + col[0].removeprefix('chr'), int(col[1]), col[3], col[4]
        assert len(ref) == len(alt) == 1
        anchor = find_anchor(chrom=chrom, pos=pos, sl=5000, cl_max=10000,
                             genome_index=idx, gene_symbol_hint=gene)
        assert anchor is not None and anchor.gene.gene_name == gene
        rows.append(dict(variation_id=vid, gene=gene, hgvs=hgvs, chrom=chrom,
            pos=pos, ref=ref, alt=alt, strand=anchor.gene.strand,
            variant_offset=anchor.variant_offset, vcf_info=col[7]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'identity.json').write_text(json.dumps(dict(source=str(archive), rows=rows), indent=2)+'\n')
    with h5py.File(OUT/'variants.h5', 'w') as f:
        for key in ['chrom', 'ref', 'alt', 'strand']:
            f.create_dataset(key, data=[r[key] for r in rows], dtype=h5py.string_dtype())
        for key in ['pos', 'variant_offset']:
            f.create_dataset(key, data=[r[key] for r in rows])
        for key in ['gene_symbol', 'source_gene_name']:
            f.create_dataset(key, data=[r['gene'] for r in rows], dtype=h5py.string_dtype())
        f.create_dataset('variant_kind', data=['snv']*len(rows), dtype=h5py.string_dtype())
        f.create_dataset('label', data=np.full(len(rows), -1, dtype=np.int8))
        f.attrs['label_note'] = 'Unlabelled case illustrations; never use for benchmark metrics.'
    print(json.dumps(rows, indent=2), flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = json.loads((OUT/'identity.json').read_text())['rows']
    sp = np.load(OUT/'deltamax.npz', allow_pickle=True)
    sa = np.load(OUT/'spliceai/scores.npz')
    sai = np.full(len(rows), np.nan)
    sai[sa['idx']] = np.where(sa['scored'], sa['delta_max'], 0.)
    assert np.isfinite(sai).all()
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    metrics = []
    with h5py.File(OUT/'cache.h5') as f:
        for i, r in enumerate(rows):
            ref, alt = f['ref_probs'][i].astype(float), f['alt_probs'][i].astype(float)
            cov = f['coverage'][i]
            # Cache is transcript oriented. Offset signs follow transcript direction.
            x = np.arange(len(ref)) - 4375
            lo, hi = (-180, 70) if r['gene'] == 'RB1' else (-50, 150)
            m = (x >= lo) & (x <= hi) & (cov > 0)
            fig, ax = plt.subplots(figsize=(6.8, 2.65), constrained_layout=True)
            points = [-103, -1] if r['gene'] == 'RB1' else [11, 76]
            ax.axvspan(*points, color='#e9e9e9', zorder=0)
            ax.text(np.mean(points), .9, f'Reported {points[1]-points[0]+1}-nt pseudoexon',
                    ha='center', fontsize=9)
            for j, sign, name, color in [(0, 1, 'Acceptor', '#2a78d6'), (1, -1, 'Donor', '#eb6834')]:
                ax.plot(x[m], sign*ref[m, j], color=color, ls='--', lw=1., label=f'{name}, ref')
                ax.plot(x[m], sign*alt[m, j], color=color, lw=1.4, label=f'{name}, alt')
            ax.axvline(0, color='black', ls=':', lw=.8)
            ax.axhline(0, color='grey', lw=.6)
            ax.set(xlabel='Position relative to variant (transcript direction, nt)',
                   ylabel='Acceptor (+) / donor (−)', xlim=(lo, hi), ylim=(-1.02, 1.02))
            ax.set_title(f"{r['gene']} {r['hgvs']}  |  SpliceAI {sai[i]:.3f}; SplicePath {sp['delta_max'][i]:.3f}")
            ax.legend(ncol=4, fontsize=8.5, loc='upper center', bbox_to_anchor=(.5, -.25))
            fig.savefig(OUT/f"{r['gene'].lower()}_{P['tag']}_raw.pdf", bbox_inches='tight',
                        metadata={'Creator': None, 'Producer': None, 'CreationDate': None})
            plt.close(fig)
            metrics.append(r | dict(spliceai=float(sai[i]), splicepath=float(sp['delta_max'][i]),
                local_largest_gain={name: dict(offset=int(x[m][np.argmax((alt-ref)[m, j])]),
                    delta=float(np.max((alt-ref)[m, j]))) for j, name in [(0, 'acceptor'), (1, 'donor')]},
                expected_site_tracks={str(p): {'ref': ref[p+4375].tolist(), 'alt': alt[p+4375].tolist()}
                                      for p in points}))
    (OUT/'case_scores.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'plot'])
    parser.add_argument('--out-dir', type=Path, default=OUT)
    parser.add_argument('--canonical-table', type=Path, default=TABLE)
    parser.add_argument('--clinvar-vcf', type=Path, default=ARCHIVE)
    args = parser.parse_args()
    OUT, TABLE, ARCHIVE = args.out_dir, args.canonical_table, args.clinvar_vcf
    globals()[args.stage]()
