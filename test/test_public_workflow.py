"""Exercise data identities, public metadata joins and portable fold evaluation."""
import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from process_input.variant_identity import row_digest
from scripts.prepare_clinvar_release import prepare
from scripts.build_clinvar_gene_metadata import build, rows as summary_rows
from scripts.verify_clinvar_inputs import validate

STR = h5py.string_dtype('utf-8')


def write_rows(path, header, rows):
    with gzip.open(path, 'wt', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def inputs(tmp_path, n=10):
    keys = np.asarray([f'1:{i+10}:A:G' for i in range(n)], dtype=object)
    y = np.arange(n) % 2
    digest = row_digest(keys)
    variants, cache = tmp_path/'variants.h5', tmp_path/'cache.h5'
    with h5py.File(variants, 'w') as f:
        for k, v in dict(chrom=['chr1']*n, ref=['A']*n, alt=['G']*n,
                         gene_symbol=[f'GENE{i}' for i in range(n)]).items():
            f.create_dataset(k, data=v, dtype=STR)
        f['pos'] = np.arange(n)+10
        f['label'] = y
    rng = np.random.default_rng(42)
    with h5py.File(cache, 'w') as f:
        f['ref_probs'] = rng.random((n, 16, 3)).astype(np.float16)
        f['alt_probs'] = rng.random((n, 16, 3)).astype(np.float16)
        f['coverage'] = np.full((n, 16), 4, dtype=np.int8)
        f['labels'] = y
        f['row_index'] = np.arange(n)
        f.create_dataset('key', data=keys, dtype=STR)
        f.attrs.update(window_left=8, window_right=8, stride=1250)
    coord, scalars = tmp_path/'coord.npz', tmp_path/'scalars.npz'
    np.savez(coord, labels=y, splice=np.ones(n, bool), intron_nt=np.full(n, 10),
             exon_nt=np.full(n, -1), akind=np.zeros(n, int), row_key_sha256=digest)
    np.savez(scalars, scalars=rng.random((n, 6)).astype(np.float32), row_key_sha256=digest)
    for k in range(5):
        fold = np.arange(n) % 5
        np.savez(tmp_path/f'fold{k}_split_cc_seed42.npz',
                 train_idx=np.flatnonzero(~np.isin(fold, [k, (k+1)%5])),
                 val_idx=np.flatnonzero(fold==(k+1)%5), test_idx=np.flatnonzero(fold==k),
                 row_key_sha256=digest)
    return variants, cache, coord, scalars, keys


def test_row_order_validation(tmp_path):
    v, c, coord, scalars, keys = inputs(tmp_path)
    assert validate(c, v, coord, scalars, tmp_path)['rows'] == 10
    with h5py.File(c, 'a') as f:
        f['key'][:] = keys[::-1]
    with pytest.raises(ValueError, match='cache identities'):
        validate(c, v, coord, scalars, tmp_path)


def test_scalar_order_validation(tmp_path):
    v, c, coord, scalars, keys = inputs(tmp_path)
    np.savez(scalars, scalars=np.ones((10, 6)), row_key_sha256=row_digest(keys[::-1]))
    with pytest.raises(ValueError, match='Conservation row identity'):
        validate(c, v, coord, scalars, tmp_path)


def test_long_allele_identity_memory():
    script = '''
import os, resource
from pathlib import Path
from process_input.variant_identity import require_equal
keys = ['1:10:A:G'] * 10000
keys[-1] = '1:20:' + 'A' * 10000 + ':G'
current = int(Path('/proc/self/statm').read_text().split()[0]) * os.sysconf('SC_PAGE_SIZE')
limit = current + 128 * 1024**2
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
require_equal(keys, keys, 'long allele identities')
'''
    subprocess.run([sys.executable, '-c', script], check=True)


def test_summary_quotes_are_literal(tmp_path):
    path=tmp_path/'submission.tsv.gz'
    with gzip.open(path,'wt') as f:
        f.write('#VariationID\tSubmittedGeneSymbol\tSubmitter\n'
                '22\tB\t"Lab\n23\tOTHER\tSecond lab\n')
    assert [r['VariationID'] for r in summary_rows(path)] == ['22','23']


def test_raw_cohort_and_gene_join(tmp_path):
    raw = tmp_path/'raw.h5'
    with h5py.File(raw, 'w') as f:
        for name, values in dict(chrom=['chr1']*3, ref=['A']*3, alt=['G']*3,
                                 gene_symbol=['A', 'B', 'C']).items():
            f.create_dataset(name, data=values, dtype=STR)
        f['pos'] = [10, 20, 30]
        f['label'] = [0, 1, 0]
    cohort = tmp_path/'rows.tsv.gz'
    write_rows(cohort, ['key','label','gene_symbol','legacy_splice','v50_row','v50_gene',
                       'v50_strand','v50_offset','v50_chunk_idx','v50_local_idx'], [
        ['1:20:A:G',1,'B',1,0,'B','+',6000,0,0],
        ['1:10:A:G',0,'A',0,-1,'','',-1,-1,-1]])
    output = tmp_path/'prepared'
    assert prepare(raw, cohort, output)['rows'] == 2
    with h5py.File(output/'release.h5') as f:
        np.testing.assert_array_equal(f['pos'][:], [20,10])
    vs, ss = tmp_path/'variant.gz', tmp_path/'submission.gz'
    write_rows(vs, ['#AlleleID','Name','Assembly','Chromosome','PositionVCF',
                    'ReferenceAlleleVCF','AlternateAlleleVCF','VariationID'], [
        ['1','NM_1(A):c.1A>G','GRCh38','1','10','A','G','11'],
        ['2','NM_2(B):c.1A>G','GRCh38','1','20','A','G','22'],
        ['3','NM_3(OTHER):c.1A>G','GRCh38','1','20','A','G','23']])
    write_rows(ss, ['#VariationID','SubmittedGeneSymbol','ContributesToAggregateClassification'],
               [['11','not provided','yes'], ['22','B;ALIAS','yes'], ['23','OTHER','no']])
    genes = tmp_path/'genes.tsv.gz'
    build(output/'release.h5', vs, ss, genes)
    with gzip.open(genes, 'rt') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    assert rows[0] == dict(key='1:20:A:G', submitted_genes='ALIAS|B|OTHER', preferred_gene='B')
    assert rows[1]['submitted_genes'] == ''


def test_reference_and_alternative_cache_assembly(tmp_path):
    from scripts.assemble_v50mane_reportedgene_cache import GEOMETRY
    variants, _, _, _, keys = inputs(tmp_path, n=3)
    with h5py.File(variants, 'a') as f:
        f.create_dataset('variant_kind', data=['snv']*3, dtype=STR)
    v50, alt = tmp_path/'v50.h5', tmp_path/'alt.h5'
    with h5py.File(variants) as src:
        for path, rows in [(v50, [0,1]), (alt, [1])]:
            with h5py.File(path, 'w') as dst:
                for k in ['chrom','pos','ref','alt']:
                    values=src[k][:][rows]
                    dst.create_dataset(k,data=values,dtype=STR if values.dtype.kind=='O' else values.dtype)
                if path == alt:dst['parent_row']=[1]
    selection = tmp_path/'anchors.npz'
    np.savez(selection,key=keys,needs_alt=[False,True,False],train_mask=[True,True,False],
             anchored=[True,True,False],universe=[True,True,True],gene=np.asarray(['A','B',''],object))
    coord = tmp_path/'mane.npz'
    np.savez(coord,labels=np.arange(3)%2,splice=np.ones(3,bool),row_key_sha256=row_digest(keys))
    for name,n,value in [('base',2,.1),('alt',1,.8)]:
        folder=tmp_path/name;folder.mkdir()
        with h5py.File(folder/'part_00000.h5','w') as f:
            f['ref_probs']=np.full((n,8750,3),value,dtype=np.float16)
            f['alt_probs']=np.full((n,8750,3),value,dtype=np.float16)
            f['coverage']=np.full((n,8750),4,dtype=np.int8)
            f['row_index']=np.arange(n)
            f.attrs.update(GEOMETRY)
    out=tmp_path/'assembled.h5'
    subprocess.run([sys.executable,'-m','scripts.assemble_v50mane_reportedgene_cache',
                    '--release-h5',str(variants),'--v50-h5',str(v50),'--anchor-selection',str(selection),
                    '--alt-h5',str(alt),'--coord',str(coord),'--parts-dir',str(tmp_path/'base'),
                    '--alt-dir',str(tmp_path/'alt'),'--out-h5',str(out)],check=True)
    with h5py.File(out) as f:
        assert f['ref_probs'][0,0,0] == np.float16(.1)
        assert f['ref_probs'][1,0,0] == np.float16(.8)
        assert not f['ref_probs'][2].any() and not f['coverage'][2].any()
        assert f.attrs['row_key_sha256']==row_digest(keys)


def test_scalar_generation_cli(tmp_path):
    import pyBigWig
    v,c,_,_,keys=inputs(tmp_path,n=3)
    cons=tmp_path/'cons';cons.mkdir()
    for name,value in [('phastCons',.2),('phyloP',5.)]:
        with pyBigWig.open(str(cons/f'hg38.{name}100way.bw'),'w') as f:
            f.addHeader([('chr1',50)])
            f.addEntries('chr1',0,values=[value]*50,span=1,step=1)
    output=tmp_path/'generated.npz'
    subprocess.run([sys.executable,'-m','process_input.build_conservation_scalar_sidecar',
                    '--cache',str(c),'--variant-h5',str(v),'--cons-dir',str(cons),
                    '--workers','1','--out',str(output)],check=True)
    z=np.load(output)
    np.testing.assert_allclose(z['scalars'],np.tile([.2,.2,.2,.5,.5,.5],(3,1)),rtol=1e-6)
    assert str(z['row_key_sha256'])==row_digest(keys)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Mamba-2 forward requires CUDA')
def test_current_oof_command(tmp_path):
    from model.sav_classifier import RbfMamba2SavClassifier
    v, c, coord, scalars, _ = inputs(tmp_path, n=330)
    cv = tmp_path/'heads'
    kwargs = dict(num_prob_channels=6, num_feature_channels=1, num_scalar_features=6,
                  d_model=32, n_layers=1, d_state=16, headdim=16, use_marker=False)
    for k in range(5):
        torch.manual_seed(k)
        model = RbfMamba2SavClassifier(**kwargs)
        path = cv/f'fold{k}'
        path.mkdir(parents=True)
        torch.save(dict(model_state=model.state_dict(), model_kwargs=kwargs,
                        config=dict(train=dict(batch_size=64), window=dict(half_bp=8,left=8,right=8,coverage=True),
                                    feature=dict(scalar_npz='unused/scalars.npz', scalar_key='scalars'))), path/'best.pt')
    out = tmp_path/'oof.json'
    subprocess.run([sys.executable, '-m', 'scripts.eval_drcons_cv5_oof',
                    '--cache',str(c),'--cvdir',str(cv),'--split-dir',str(tmp_path),
                    '--split-tag','cc_seed42','--coord',str(coord),'--variants-h5',str(v),
                    '--scalars',str(scalars),'--out',str(out),'--batch-size','64','--num-workers','0'], check=True)
    pred = np.load(tmp_path/'oof_preds.npz')
    assert np.isfinite(pred['p']).all() and pred['p'].shape == (330,)
    assert json.loads(out.read_text())['validation']['every_row_tested_once']
