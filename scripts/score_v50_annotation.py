"""Save native annotation probabilities on the frozen v50 benchmark inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_v50_annotation import NAS, suffixes


def native_predictor(tool, native):
    if tool in ('spliceai', 'ssnet'):
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
        assert tf.config.list_physical_devices('GPU'), 'GPU required'
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        if tool == 'spliceai':
            from scripts.eval_spliceai_official_annotation import IN_MAP
            paths = [NAS/f'spliceai_env/lib/python3.11/site-packages/spliceai/models/spliceai{i}.h5'
                     for i in range(1, 6)]
            models = [tf.keras.models.load_model(p) for p in paths]
            def predict(x):
                # Native five-model mean, channels neither/acceptor/donor.
                p = np.mean([m(IN_MAP[x], training=False).numpy() for m in models], axis=0)
                return p[..., [1, 2]]
        else:
            from scripts.eval_ssnet_annotation import _load_models, _encode_batch, _forward
            models = _load_models(str(native/'ssnet'), str(native/'ssnet_weights'))
            def predict(x):
                return np.stack(_forward(models, _encode_batch(x)), axis=-1)
        return predict, {'tensorflow':tf.__version__}
    import torch
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    assert torch.cuda.is_available(), 'GPU required'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if tool == 'st':
        from scripts.eval_st_annotation import _load_st_model, _forward
        model, device = _load_st_model(str(native/'st'))
        predict = lambda x: np.stack(_forward(model, device, x), axis=-1)
    else:
        sys.path.insert(0, str(NAS/'pangolin_pglibs'))
        from scripts.eval_pangolin_annotation import _load_pangolin_models, _forward_splice4
        models, _ = _load_pangolin_models(str(NAS/'pangolin_pglibs/pangolin/models'), torch.device('cuda'))
        predict = lambda x: _forward_splice4(models, x, 5000).max(axis=-1, keepdims=True)
    return predict, {'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(0)}


def splicepath_batches(args, info):
    import torch
    from omegaconf import OmegaConf
    from evaluate.evaluate import _load_train_config, _build_merged_config
    from evaluate.evaluator import Evaluator
    from train.dataset import SplicingH5Dataset
    from torch.utils.data import DataLoader, Subset
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
    assert torch.cuda.is_available(), 'GPU required'
    assert args.batch_size == 16
    cfg = OmegaConf.load('config/evaluate/evaluate_annotation_mig.yaml')
    cfg.evaluation.batch_size = 16
    cfg.evaluation.num_workers = 0
    cfg.mlflow.enabled = False
    train_cfg = _load_train_config(str(args.checkpoint), use_h5=True, model_type='hyenadna',
                                   train_config_path=None, ignore_embed_metadata=False)
    cfg = _build_merged_config(cfg, train_cfg, checkpoint_path=str(args.checkpoint))
    assert cfg.evaluation.probability_mode == 'softmax'
    evaluator = Evaluator(cfg, OmegaConf.load('config/issasplice/model.yaml'), local_rank=0, world_size=1)
    evaluator.model.eval()
    data = Subset(SplicingH5Dataset(str(args.h5)), range(args.row_start, args.row_end))
    loader = DataLoader(data, batch_size=16, shuffle=False, num_workers=0,
                        pin_memory=True, drop_last=False)
    info.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), precision='native float32',
                probability_mode='softmax', checkpoint=str(args.checkpoint), drop_last=False)
    with torch.no_grad():
        for batch in loader:
            result = evaluator._process_batch(batch)
            assert result.probs is not None and result.labels is not None
            # H5 padding has an all-zero three-class target, not a negative label.
            # Native comparators exclude Y=-1; use exactly the same positions here.
            assert result.labels.shape[:2] == batch['labels'].transpose(1, 2).shape[:2]
            valid = result.labels.sum(dim=-1) > 0
            p = result.probs[..., :2][valid].cpu().numpy()
            lab = result.labels[..., :2][valid].cpu().numpy()
            y = np.zeros(len(lab), np.int8)
            y[lab[:, 0] > .5] = 1
            y[lab[:, 1] > .5] = 2
            yield p, y
    info['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', choices=['splicepath', 'spliceai', 'pangolin', 'st', 'ssnet'], required=True)
    parser.add_argument('--h5', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    args = parser.parse_args()
    assert not args.out.exists(), args.out
    with h5py.File(args.h5) as f:
        sufs = suffixes(f)
        n_rows = sum(len(f[f'X{i}']) for i in sufs)
        labels = np.concatenate([f[f'Y{i}'][:] for i in sufs])
        assert 0 <= args.shard_index < args.num_shards
        # Partition complete batches, keeping the full-run batch composition.
        n_batches = (n_rows + args.batch_size - 1) // args.batch_size
        per_shard = (n_batches + args.num_shards - 1) // args.num_shards
        args.row_start = args.shard_index * per_shard * args.batch_size
        args.row_end = min(n_rows, (args.shard_index+1) * per_shard * args.batch_size)
        assert args.row_start < args.row_end
        labels = labels[args.row_start:args.row_end]
        expected = labels[labels >= 0]
        del labels
    info = {'tool':args.tool, 'h5':str(args.h5), 'batch_size':args.batch_size,
            'windows':args.row_end-args.row_start, 'total_windows':n_rows,
            'row_start':args.row_start, 'row_end':args.row_end,
            'shard_index':args.shard_index, 'num_shards':args.num_shards,
            'valid_positions':len(expected), 'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'job_id':os.environ['SLURM_JOB_ID'], 'host':os.uname().nodename}
    started = time.monotonic()
    if args.tool == 'splicepath':
        assert args.checkpoint is not None
        batches = splicepath_batches(args, info)
    else:
        predict, runtime = native_predictor(args.tool, args.native)
        info.update(runtime)
        def native_batches():
            with h5py.File(args.h5) as f:
                base = 0
                for i in sufs:
                    size = len(f[f'X{i}'])
                    lo = max(0, args.row_start-base)
                    hi = min(size, args.row_end-base)
                    for start in range(lo, hi, args.batch_size):
                        end = min(hi, start+args.batch_size)
                        x = f[f'X{i}'][start:end].astype(np.int64)
                        y = f[f'Y{i}'][start:end]
                        p = predict(x)
                        yield p[y >= 0], y[y >= 0]
                    base += size
        batches = native_batches()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix('.partial.h5')
    with h5py.File(partial, 'w') as f:
        channels = 1 if args.tool == 'pangolin' else 2
        dtype = 'float32' if args.tool == 'splicepath' else 'float16'
        scores = f.create_dataset('scores', (len(expected), channels), dtype=dtype,
                                  chunks=(min(65536, len(expected)), channels))
        f.create_dataset('labels', data=expected)
        offset = 0
        for b, (prob, label) in enumerate(batches):
            assert prob.shape == (len(label), channels)
            assert np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1
            np.testing.assert_array_equal(label, expected[offset:offset+len(label)])
            scores[offset:offset+len(label)] = prob
            offset += len(label)
            if b < 3 or (b + 1) % 100 == 0:
                print(f'{args.tool} batch={b+1} valid={offset}/{len(expected)} elapsed={time.monotonic()-started:.1f}s', flush=True)
                f.flush()
        assert offset == len(expected)
        info['seconds'] = time.monotonic()-started
        info['score_dtype'] = dtype
        f.attrs['metadata'] = json.dumps(info)
        f.attrs['complete'] = True
    partial.rename(args.out)
    args.out.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2), flush=True)


if __name__ == '__main__':
    main()
