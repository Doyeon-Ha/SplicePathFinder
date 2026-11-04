"""Measure actual K4 backbone forwards and the existing 8750-position SPF head.

Slurm/A100 only. Hooks time network execution in the production cache builder;
encoding, alignment, merging and disk I/O are excluded. No model is fitted.
"""
import functools
import json
import os
import subprocess
import sys
import h5py
import numpy as np
import torch
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant-h5', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--train-config', type=Path, required=True)
    parser.add_argument('--ref-fasta', type=Path, required=True)
    parser.add_argument('--head', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from evaluate import evaluate_splicevardb_sav as ev
    from scripts import evaluate_splicevardb_sav_multichunk as mc
    from process_input import build_multichunk_pernt_cache as builder
    from model.sav_classifier import RbfMamba2SavClassifier
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    calls, groups = [], []
    original_load, original_forward = ev._load_hyenadna_model, mc._batched_forward

    def load(*args, **kwargs):
        model, cfg = original_load(*args, **kwargs)
        original = model.forward
        @functools.wraps(original)
        def forward(*a, **kw):
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record(); result = original(*a, **kw); end.record()
            calls.append((start, end))
            return result
        model.forward = forward
        return model, cfg

    def batched(*args, **kwargs):
        first = len(calls)
        result = original_forward(*args, **kwargs)
        groups.append((first, len(calls)))
        return result

    ev._load_hyenadna_model, mc._batched_forward = load, batched
    cache = args.out_dir/'speed_cache.h5'
    sys.argv = ['build_multichunk_pernt_cache', '--variant-h5', str(args.variant_h5),
        '--checkpoint', str(args.checkpoint), '--out-h5', str(cache), '--start', '0', '--end', '68', '--batch-size', '16', '--train-config', str(args.train_config), '--ref-fasta', str(args.ref_fasta)]
    builder.main()
    torch.cuda.synchronize()
    assert len(calls) == 34, len(calls)
    backbone_ms = [start.elapsed_time(end)/2 for start, end in calls[2:]]
    with h5py.File(cache) as f:
        assert (f['n_chunks_used'][:] == 4).all()
        tracks = np.concatenate([f['ref_probs'][4:], f['alt_probs'][4:]], axis=-1).astype(np.float32)
        coverage = f['coverage'][4:].astype(np.float32)[..., None] / 4.
    assert tracks.shape == (64, 8750, 6) and np.isfinite(tracks).all()
    torch.cuda.empty_cache()
    head_path = args.head
    z = torch.load(head_path, map_location='cpu', weights_only=False)
    kwargs = z['model_kwargs'].copy()
    if 'num_input_channels' in kwargs:
        kwargs['num_prob_channels'] = kwargs.pop('num_input_channels')
    head = RbfMamba2SavClassifier(**kwargs).cuda().eval()
    head.load_state_dict(z['model_state'])
    assert not head.use_marker and head.num_prob_channels == 6 and head.num_feature_channels == 1
    x = torch.from_numpy(np.concatenate([tracks, coverage], axis=-1)).cuda()
    scalars = torch.zeros((16, head.num_scalar_features), device='cuda')
    with torch.inference_mode():
        for _ in range(3):
            head(x[:16], scalars=scalars)
        torch.cuda.synchronize()
        head_ms = []
        for _ in range(5):
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for i in range(0, 64, 16):
                pred = head(x[i:i+16], scalars=scalars)
            end.record();torch.cuda.synchronize()
            assert torch.isfinite(pred).all()
            head_ms.append(start.elapsed_time(end)/64)
    report = dict(backbone_ms_per_variant=float(np.mean(backbone_ms)),
        head_ms_per_variant=float(np.mean(head_ms)), backbone_n=64, head_repeats=5,
        backbone_forward_batch=16, head_batch=16, chunks_per_variant=4, merged_positions=8750,
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        checkpoint=str(args.checkpoint), head_checkpoint=str(head_path), data_path=str(args.variant_h5),
        head_kwargs=kwargs, head_parameters=sum(p.numel() for p in head.parameters()),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        method='CUDA-event forward-only timing in production K4 builder; two warm-up batches (four variants) excluded. Head uses real paired cached tracks and six constant scalars; no score evaluation. Excludes encoding, softmax outside model, alignment, merge, disk I/O, and model loading.')
    (args.out_dir/'speed_metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
