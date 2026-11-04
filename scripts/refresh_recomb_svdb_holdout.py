"""Evaluate saved native scores on the complement of actual epoch-selection rows.

Run via Slurm. Uses the fixed H-Net native no-stop-exon epoch-8 backbone;
no model fitting or inference. Emits auditable masks, scores, metrics and registry
records. The complement overlaps ClinVar and is not a gene-disjoint benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts import dissect_svdb_region_table1 as native
from scripts.build_svdb_validation_h5 import CLINVAR_H5S
from scripts.refresh_recomb2027_results import Ranker, components, dec, keys, metric, table

NAS = Path('.')
EVAL = NAS / 'data/eval'
RUN = NAS / 'checkpoints/splicepath-allgenes-realflank-nostopexon-hnet-20ep_20261021_011600'
CKPT = RUN / 'epoch_008_model.pt'
VARIANTS = Path(native.ANCHORED_H5)
SELECTION = EVAL / 'splicevardb_val_excl_clinvar.h5'
CACHE = EVAL / 'sav_pernt_cache_multichunk/SVDB_sp_allgenes_realflank_nostopexon_hnet_ep8'
EXTERNAL = EVAL / 'recomb2027/ep12_refresh_20261023/sav_predictions.npz'
DATASET = 'splicevardb_excl_epoch_selection_clinvar_overlap_20261026'
MODELS = ['Pangolin', 'SpliceAI', 'ST', 'SSNet', 'SplicePath']
META = {
    'Pangolin': ('pangolin_nocall0', 'pangolin', 'official_tool', 2500, 'none'),
    'SpliceAI': ('spliceai_illumina_nocall0', 'spliceai', 'official_tool', 2500, 'none'),
    'ST': ('splicetransformer_nocall0', 'splicetransformer', 'official_tool', 2500, 'none'),
    'SSNet': ('ssnet_pangolin_nocall0', 'ssnet', 'official_tool', 100000, 'none'),
    'SplicePath': ('sp_allgenes_realflank_nostopexon_hnet_ep8', 'ours',
                   'v4_aligned_k4_cache', 4375, str(CKPT)),
}


def npz_keys(z):
    return keys(z['chrom'], z['pos'], z['ref'], z['alt'])


def svdb_components(symbols):
    """SpliceVarDB composite symbols use commas as well as pipes."""
    normalized = np.array(['|'.join(t.strip() for t in g.replace(',', '|').split('|'))
                           for g in dec(symbols)])
    result = components(normalized)
    token_group = {}
    for name, group in zip(normalized, result):
        for token in name.split('|'):
            if token in {'', '.', 'NA', 'nan', 'None'}:
                continue
            if token in token_group:
                assert token_group[token] == group, token
            token_group[token] = group
    return result


def bootstrap(y, scores, groups):
    """Paired gene-component bootstrap; fixed fitted models, seed 42."""
    _, inv = np.unique(groups, return_inverse=True)
    n_groups = int(inv.max()) + 1
    rankers = {name: Ranker(y, s) for name, s in scores.items()}
    for name, ranker in rankers.items():
        weights = (np.arange(len(y)) % 5).astype(float)
        np.testing.assert_allclose(ranker.score(weights), [
            roc_auc_score(y, scores[name], sample_weight=weights),
            average_precision_score(y, scores[name], sample_weight=weights)],
            rtol=1e-10, atol=1e-10)
    rng = np.random.default_rng(42)
    draws = {name: np.empty((2000, 2)) for name in scores}
    for i in range(2000):
        weights = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)[inv]
        for name, ranker in rankers.items():
            draws[name][i] = ranker.score(weights)
    quantile = lambda a: np.nanpercentile(a, [2.5, 97.5], axis=0).T.tolist()
    return dict(n_components=n_groups, n_boot=2000, seed=42,
                valid_replicates={k: int(np.isfinite(v).all(axis=1).sum()) for k, v in draws.items()},
                intervals={k: quantile(v) for k, v in draws.items()},
                differences={k: quantile(draws['SplicePath'] - draws[k])
                             for k in scores if k != 'SplicePath'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--sp-scores', type=Path, default=CACHE / 'deltamax_unmasked.npz')
    parser.add_argument('--sp-model', default=META['SplicePath'][0])
    parser.add_argument('--sp-source', type=Path, action='append', default=[],
                        help='Additional cache/anchor provenance for an alternate readout')
    parser.add_argument('--sp-note', default='')
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    full_keys, y, snv = table(VARIANTS)
    selection_keys, selection_y, _ = table(SELECTION)
    assert len(full_keys) == 24933 and len(selection_keys) == 23473
    selection_mask = np.isin(full_keys, selection_keys)
    np.testing.assert_array_equal(full_keys[selection_mask], selection_keys)
    np.testing.assert_array_equal(y[selection_mask], selection_y)
    test = ~selection_mask
    assert test.sum() == 1460
    assert not (set(full_keys[test]) & set(selection_keys))
    clinvar = set().union(*(set(table(p)[0]) for p in CLINVAR_H5S))
    assert set(full_keys[test]) == set(full_keys) & clinvar
    assert not (set(selection_keys) & clinvar)

    history = json.loads((RUN / 'svdb_validation/history.json').read_text())
    assert history['best_epoch'] == 8
    assert [r['epoch'] for r in history['records']] == list(range(1, 10))
    audited_epochs = []
    for row in history['records']:
        path = RUN / f"svdb_validation/epoch_{row['epoch']:03d}/deltamax.npz"
        with np.load(path, allow_pickle=True) as z:
            np.testing.assert_array_equal(npz_keys(z), selection_keys)
            assert row['svdb_n'] == len(selection_keys)
            np.testing.assert_allclose(roc_auc_score(selection_y, z['delta_max']),
                                       row['svdb_auroc'], rtol=0, atol=1e-12)
        audited_epochs.append(str(path))

    # Native artifact readers retain the chr prefix; table()/keys() remove it.
    key2i = {'chr' + k: i for i, k in enumerate(full_keys)}
    sai, called = native._score_spliceai(len(y))
    st = native._score_st(len(y))
    with np.load(native.ST_NPZ) as z:
        np.testing.assert_array_equal(z['label'], y[z['row_idx']])
        st[z['row_idx'][~z['covered'].astype(bool)]] = np.nan
    raw = dict(SpliceAI=np.where(called, sai, np.nan),
               Pangolin=native._score_pangolin(len(y), key2i), ST=st,
               SSNet=native._score_ssnet(len(y), key2i))
    scores = {k: np.where(np.isfinite(s), s, 0.) for k, s in raw.items()}
    with np.load(EXTERNAL, allow_pickle=True) as z:
        np.testing.assert_array_equal([k.removeprefix('chr') for k in dec(z['key'])], full_keys)
        np.testing.assert_array_equal(z['y'], y)
        for name, s in scores.items():
            np.testing.assert_array_equal(z[name], s)
    with np.load(args.sp_scores, allow_pickle=True) as z:
        np.testing.assert_array_equal(npz_keys(z), full_keys)
        assert not z['masked'].item() and z['window_bp'].item() == 4375
        scores['SplicePath'] = z['delta_max'].astype(float)
        np.testing.assert_array_equal(scores['SplicePath'], z['per_direction'].max(axis=1))
    assert np.isfinite(scores['SplicePath']).all()
    raw['SplicePath'] = scores['SplicePath']
    with h5py.File(CACHE / 'cache.h5') as f:
        np.testing.assert_array_equal(f['labels'][:], y)
        assert np.all(f['n_chunks_used'][:] == 4)
        assert f['ref_probs'].shape == (len(y), 8750, 3)
        assert not f.attrs['mask_canonical']
    with h5py.File(VARIANTS) as f:
        gene = dec(f['gene_symbol'][:])
        group = svdb_components(gene)
    parts = {'overall': test, 'snv': test & snv, 'indel': test & ~snv}
    counts = {p: dict(n=int(m.sum()), positive=int(y[m].sum()),
                     negative=int(m.sum() - y[m].sum()),
                     prevalence=float(y[m].mean()), n_components=len(set(group[m])))
              for p, m in parts.items()}
    rows = {p: {k: metric(y[m], scores[k][m]) for k in MODELS} for p, m in parts.items()}
    ci = {p: bootstrap(y[m], {k: scores[k][m] for k in MODELS}, group[m])
          for p, m in parts.items()}
    no_calls = {p: {k: int((~np.isfinite(raw[k][m])).sum()) for k in MODELS}
                for p, m in parts.items()}
    for p, m in parts.items():
        for name in MODELS:
            assert rows[p][name]['n'] == int(m.sum())
            assert np.all(scores[name][m][~np.isfinite(raw[name][m])] == 0)
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    sources = [VARIANTS, SELECTION, *CLINVAR_H5S, CACHE / 'cache.h5',
               args.sp_scores, *args.sp_source, EXTERNAL, Path(native.PANGOLIN_CSV),
               Path(native.SPLICEAI_DIR), Path(native.ST_NPZ), Path(native.SSNET_NPZ),
               RUN / 'train_config.yaml', RUN / 'svdb_validation/history.json']
    result = dict(dataset=DATASET, checkpoint=str(CKPT), git_commit=git_commit,
                  selection_n=len(selection_keys), evaluation_n=int(test.sum()),
                  selection_overlap=0, clinvar_overlap=int(test.sum()), counts=counts,
                  shared_gene_components=len(set(group[test]) & set(group[selection_mask])),
                  rows=rows, bootstrap=ci, no_calls=no_calls, no_call_value=0,
                  audited_selection_epochs=audited_epochs,
                  source_paths=list(map(str, sources)),
                  test_key_sha256=hashlib.sha256('\n'.join(full_keys[test]).encode()).hexdigest(),
                  interpretation='Variant-disjoint from epoch selection; overlaps ClinVar; '
                  'not gene-disjoint or a prospective holdout for all development choices; '
                  'SplicePath label-free delta max, no ClinVar-trained head. ' + args.sp_note)
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(out / 'split_masks.npz', key=full_keys, selection=selection_mask,
                        test=test, label=y, snv=snv, component=group)
    np.savez_compressed(out / 'predictions.npz', key=full_keys[test], y=y[test],
                        row_index=np.flatnonzero(test), snv=snv[test], component=group[test],
                        **{k: scores[k][test] for k in MODELS},
                        **{f'{k}_called': np.isfinite(raw[k][test]) for k in MODELS})
    records = []
    for part, vals in rows.items():
        for name, metrics in vals.items():
            model, family, scoring, window, checkpoint = META[name]
            if name == 'SplicePath':
                model = args.sp_model
            records.append(dict(model=model, model_family=family, dataset=DATASET,
                dataset_n=int(test.sum()), partition=part, partition_n=metrics['n'],
                scoring=scoring, window_bp=window, auroc=metrics['auroc'], auprc=metrics['auprc'],
                source=str(out / 'predictions.npz'), data_path=list(map(str, sources)),
                script='scripts/refresh_recomb_svdb_holdout.py', checkpoint=checkpoint,
                git_commit=git_commit, notes=result['interpretation'] +
                f" Missing scores=0; {no_calls[part][name]} no-calls; "
                f"positive={counts[part]['positive']}, negative={counts[part]['negative']}. " +
                ('SSNet window_bp is its native 100000-nt input frame.' if name == 'SSNet' else '')))
    (out / 'registry.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    print(json.dumps({k: result[k] for k in ['counts', 'rows', 'no_calls', 'shared_gene_components']}, indent=2))


if __name__ == '__main__':
    main()
