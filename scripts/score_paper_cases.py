"""Score two reported-gene MANE Select cases with their held-out SPF head."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

RUN = Path('outputs/paper_case_scores')
NAS = Path('.')
EVAL = NAS / 'data/eval'
CASES = EVAL / 'recomb2027/v50mane_ep10_canonical_20261030/cases'
HEAD = EVAL / 'spf_multichunk/cv5_sp_v50mane_hnet_ep10_reportedgene_cc'
ANCHORS = EVAL / 'recomb2027/v50mane_reportedgene_anchors_v2_20261030/anchor_selection_v50mane.npz'
TABLE = NAS / 'data/canonical_dataset_v50_mane_regions.txt'
SCALARS = EVAL / 'clinvar2026_phastcons_probe/clinvar_20260628_conservation_scalars.npz'


def prepare():
    from process_input.gencode_anchor import load_canonical_table, build_genome_index, find_anchor
    index = build_genome_index(load_canonical_table(str(TABLE), chrom_filter=None, sl=5000))
    RUN.mkdir(parents=True, exist_ok=True)
    rows = json.loads((CASES / 'case_scores.json').read_text())
    with (EVAL / 'training_label_audit_20261023/selected_transcripts_v50.tsv').open() as f:
        transcripts = {r['gene_name']: r for r in csv.DictReader(f, delimiter='\t')}
    anchors = np.load(ANCHORS, allow_pickle=True)
    for r in rows:
        a = find_anchor(chrom=r['chrom'], pos=r['pos'], sl=5000, cl_max=10000,
                        genome_index=index, gene_symbol_hint=r['gene'])
        tx = transcripts[r['gene']]
        assert tx['tier'] == 'MANE_Select'
        assert a.gene.gene_name == r['gene'] and a.gene.strand == r['strand']
        assert a.variant_offset == r['variant_offset']
        assert (a.gene.tx_start, a.gene.tx_end) == (int(tx['tx_start_1based']), int(tx['tx_end']))
        gene_rows = np.flatnonzero(anchors['gene'] == r['gene'])
        assert len(gene_rows) > 0 and set(anchors['fold_id'][gene_rows]) == {3}
        split = np.load(EVAL / 'clinvar_pb_2026_split_cc/fold3_split_cc_seed42.npz')
        assert not np.intersect1d(gene_rows, split['train_idx']).size
        assert not np.intersect1d(gene_rows, split['val_idx']).size
        r.update(transcript_id=tx['transcript_id'], transcript_tier=tx['tier'],
                 transcript_start=a.gene.tx_start, transcript_end=a.gene.tx_end,
                 fold=3, n_gene_rows_excluded_from_fit=int(len(gene_rows)))
    cons_script = Path('process_input/build_conservation_scalar_sidecar.py')
    spec = importlib.util.spec_from_file_location('scalar_builder', cons_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._init(str(NAS / 'data/conservation'))
    scalars = mod._work((np.asarray([r['chrom'] for r in rows]), np.asarray([r['pos'] for r in rows])))
    rb1 = np.flatnonzero(anchors['key'] == '13:48471962:A:G')
    assert len(rb1) == 1 and anchors['gene'][rb1[0]] == 'RB1'
    rb1 = int(rb1[0])
    np.testing.assert_allclose(scalars[0], np.load(SCALARS)['scalars'][rb1], atol=1e-7, rtol=0)
    with h5py.File(CASES / 'cache.h5') as src, h5py.File(RUN / 'cases.h5', 'x') as dst:
        for name in ['ref_probs', 'alt_probs', 'coverage']:
            dst.create_dataset(name, data=src[name][:])
        dst.create_dataset('labels', data=np.full(2, -1, dtype=np.int8))
        dst.attrs.update(window_left=4375, window_right=4375, stride=1250,
                         mask_canonical=False, source_cache=str(CASES / 'cache.h5'))
    for r in rows:
        r.pop('vcf_info', None)
        r.pop('expected_site_tracks', None)
        r.pop('local_largest_gain', None)
    np.savez(RUN / 'scalars.npz', scalars=scalars)
    report = dict(rows=rows, head=str(HEAD / 'fold3/best.pt'),
                  backbone=str(NAS / 'checkpoints/splicepath-v50mane-allgenes-realflank-hnet-20ep_20261023_144538/epoch_010_model.pt'),
                  cache_source=str(CASES / 'cache.h5'), anchor_table=str(TABLE),
                  anchor_selection=str(ANCHORS), rb1_release_row=rb1,
                  scalar_builder=str(cons_script),
                  scalar_builder_sha256=hashlib.sha256(cons_script.read_bytes()).hexdigest(),
                  scalars=scalars.tolist(), input_policy='reported gene, GENCODE v50 MANE Select, transcript direction; K=4 select-pair merged 8750, unmasked',
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    (RUN / 'inputs.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def score():
    import torch
    from model.sav_classifier import RbfMamba2SavClassifier
    from train.dataset_sav import SavFeatureDataset
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    assert torch.cuda.is_available()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    info = json.loads((RUN / 'inputs.json').read_text())
    ck = torch.load(info['head'], map_location='cpu', weights_only=False)
    assert ck['epoch'] == json.loads((HEAD / 'fold3/complete.json').read_text())['best_epoch']
    assert ck['config']['train']['batch_size'] == 64
    assert ck['config']['cache_h5'].endswith('REPORTEDGENE_sp_v50mane_hnet_ep10/cache.h5')
    assert Path(ck['config']['feature']['scalar_npz']).name == SCALARS.name
    split = np.load(EVAL / 'clinvar_pb_2026_split_cc/fold3_split_cc_seed42.npz')
    old = np.load(HEAD / 'drcons_cv5_oof_result_v50mane_reportedgene_preds.npz')['p']
    candidates = split['test_idx'][np.isfinite(old[split['test_idx']])]
    controls = np.r_[info['rb1_release_row'], candidates[candidates != info['rb1_release_row']][:61]]
    assert len(controls) == 62
    control_cache = EVAL / 'sav_pernt_cache_multichunk/REPORTEDGENE_sp_v50mane_hnet_ep10/cache.h5'
    control_ds = SavFeatureDataset(str(control_cache), controls, scalar_features_npz=str(SCALARS))
    case_ds = SavFeatureDataset(str(RUN / 'cases.h5'), np.arange(2), scalar_features_npz=str(RUN / 'scalars.npz'))
    items = [case_ds[i] for i in range(2)] + [control_ds[i] for i in range(62)]
    np.testing.assert_array_equal(items[0]['channels'].numpy(), items[2]['channels'].numpy())
    np.testing.assert_array_equal(items[0]['scalars'].numpy(), items[2]['scalars'].numpy())
    channels = torch.stack([b['channels'] for b in items]).cuda()
    scalars = torch.stack([b['scalars'] for b in items]).cuda()
    model = RbfMamba2SavClassifier(**ck['model_kwargs']).cuda().eval()
    model.load_state_dict(ck['model_state'])
    with torch.no_grad():
        probabilities = torch.sigmoid(model(channels, None, scalars)).float().cpu().numpy()
    errors = np.abs(probabilities[2:] - old[controls])
    np.savez(RUN / 'predictions.npz', case_scores=probabilities[:2], control_rows=controls,
             control_scores=probabilities[2:], control_reference=old[controls])
    np.testing.assert_allclose(probabilities[2:], old[controls], rtol=0, atol=1e-5)
    report = dict(inputs=info, results=[dict(r, spf=float(p)) for r, p in zip(info['rows'], probabilities[:2])],
                  checkpoint_epoch=int(ck['epoch']), batch_size=64, precision='float32, no autocast',
                  device=torch.cuda.get_device_name(0),
                  torch_version=torch.__version__, job_id=os.environ.get('SLURM_JOB_ID'),
                  control_n=62, control_max_abs_error=float(errors.max()),
                  rb1_identical_to_reportedgene_training_cache=True,
                  git_commit=commit, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  note='Case scores only, not a clinical calibration or aggregate performance evaluation.')
    (RUN / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'score': score}[sys.argv[1]]()
