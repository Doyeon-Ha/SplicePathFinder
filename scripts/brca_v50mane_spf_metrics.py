#!/usr/bin/env python3
"""BRCA Exchange 벤치마크를 v50 MANE 보고유전자 SPF head로 채점한 결과를 집계한다.

``score_spf_external``이 fold 3 head로 쓴 두 코호트 예측(``all``/``extra``)을
모아 각 코호트 변이표에 join하고, overall/SNV/indel 지표와 registry 레코드를
쓴다. BRCA1·BRCA2는 둘 다 fold 3의 test fold에 있으므로 fold 3 head만이 두
유전자를 **학습에도 early-stopping 검증에도** 쓰지 않았다. 그 사실을 여기서
ClinVar 릴리스의 gene token과 동결 split으로 직접 다시 확인한다.

    python -m scripts.brca_v50mane_spf_metrics --out-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

NAS = Path(".")
EVAL = NAS / "data/eval"
FULL = (EVAL / "clinvar_20260628_gencode_anchored_indel"
        / "clinvar_20260628_gencode_anchored_indel.h5")
SPLIT = EVAL / "clinvar_pb_2026_split_cc/fold3_split_cc_seed42.npz"
CVDIR = EVAL / "spf_multichunk/cv5_sp_v50mane_hnet_ep10_reportedgene_cc"
COHORTS = {
    "all": EVAL / "brca_all_gencode_anchored.h5",
    "expert": EVAL / "brca_expert_gencode_anchored.h5",
}
GENES = ("BRCA1", "BRCA2")
FOLD = 3
MODEL = "learned_mamba2_sp_v50mane_hnet_ep10_reportedgene_fold3"


def dec(a) -> np.ndarray:
    """바이트 또는 문자열 배열을 str 객체 배열로 바꾼다."""
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in a],
                      dtype=object)


def keys(c, p, r, a) -> np.ndarray:
    """(chrom, pos, ref, alt)를 하나의 변이 키 문자열로 만든다."""
    return np.asarray([f'{str(ch).removeprefix("chr")}:{int(po)}:{re}:{al}'
                       for ch, po, re, al in zip(dec(c), p, dec(r), dec(a))],
                      dtype=object)


def table(path: Path):
    """변이표에서 키·라벨·SNV 마스크를 읽는다."""
    with h5py.File(path) as f:
        k = keys(f["chrom"][:], f["pos"][:], f["ref"][:], f["alt"][:])
        y = f["label"][:].astype(np.int8)
        snv = dec(f["variant_kind"][:]) == "snv"
    assert len(set(k)) == len(k), f"duplicate variant in {path}"
    return k, y, snv


def metric(y, s) -> dict:
    """한 부분집합의 AUROC/AUPRC를 낸다. 라벨이 한 종류면 None."""
    assert len(y) == len(s) and np.isfinite(s).all()
    r = dict(n=int(len(y)), n_pos=int(y.sum()))
    if len(np.unique(y)) < 2:
        return r | dict(auroc=None, auprc=None)
    return r | dict(auroc=float(roc_auc_score(y, s)),
                    auprc=float(average_precision_score(y, s)))


def digest(path) -> str:
    """파일의 sha256."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_fold_is_clean() -> dict:
    """fold 3의 train·val에 BRCA1/BRCA2가 한 행도 없음을 확인한다."""
    with h5py.File(FULL) as f:
        gene = dec(f["gene_symbol"][:])
        full_key = keys(f["chrom"][:], f["pos"][:], f["ref"][:], f["alt"][:])
    split = np.load(SPLIT)
    exposed = np.concatenate([split["train_idx"], split["val_idx"]])
    hits = {g: int(sum(g in str(s).split("|") for s in gene[exposed])) for g in GENES}
    assert not any(hits.values()), f"BRCA rows leaked into fold{FOLD} train/val: {hits}"
    test_hits = {g: int(sum(g in str(s).split("|") for s in gene[split["test_idx"]]))
                 for g in GENES}
    return dict(exposed_rows=int(exposed.size), train_val_hits=hits,
                test_hits=test_hits, exposed_keys=set(full_key[exposed].tolist()))


def main() -> None:
    """두 코호트의 지표와 registry 레코드를 쓴다."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    leak = assert_fold_is_clean()
    exposed_keys = leak.pop("exposed_keys")
    print(f"fold{FOLD} clean: {leak}", flush=True)

    ckpt = CVDIR / f"fold{FOLD}/best.pt"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    done = json.loads((ckpt.with_name("complete.json")).read_text())
    assert done["training_completed"] and done["best_epoch"] == ck["epoch"]
    assert ck["config"]["split"]["freeze_npz"] == str(SPLIT), ck["config"]["split"]
    assert ck["config"]["cache_h5"].endswith(
        "REPORTEDGENE_sp_v50mane_hnet_ep10/cache.h5"), ck["config"]["cache_h5"]

    pred_key, pred_score = [], []
    for name in ("all", "extra"):
        z = np.load(out / f"{name}_fold{FOLD}.npz", allow_pickle=True)
        np.testing.assert_array_equal(z["fold_ids"], [FOLD])
        np.testing.assert_array_equal(z["p_folds"][0], z["p_mean"])
        pred_key.extend(keys(z["chrom"], z["pos"], z["ref"], z["alt"]))
        pred_score.extend(z["p_mean"])
    assert len(set(pred_key)) == len(pred_key), "duplicate scored variant"
    lookup = dict(zip(pred_key, pred_score))

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report, records = {}, []
    for cohort, source in COHORTS.items():
        key, y, snv = table(source)
        missing = [k for k in key if k not in lookup]
        assert not missing, f"{cohort}: {len(missing)} variants unscored, e.g. {missing[:3]}"
        assert not exposed_keys.intersection(key), f"{cohort} overlaps fold{FOLD} train/val"
        p = np.asarray([lookup[k] for k in key], dtype=np.float32)
        npz = out / f"brca_{cohort}_spf_v50mane.npz"
        np.savez_compressed(npz, key=np.asarray(key, dtype=object), y=y, snv=snv, SPF=p)
        report[cohort] = dict(n=int(len(y)), n_pos=int(y.sum()), source=str(source),
                              fold=FOLD, epoch=int(ck["epoch"]), rows={})
        for part, mask in dict(overall=np.ones(len(y), bool), snv=snv, indel=~snv).items():
            m = metric(y[mask], p[mask])
            report[cohort]["rows"][part] = m
            print(f"  {cohort:7s} {part:7s} n={m['n']:6d} pos={m['n_pos']:5d} "
                  f"AUROC={m['auroc']} AUPRC={m['auprc']}", flush=True)
            if m["auroc"] is None:
                continue
            records.append(dict(
                model=MODEL, model_family="ours", dataset=f"brca_exchange_{cohort}",
                dataset_n=int(len(y)), partition=part, partition_n=m["n"],
                scoring="learned_mamba2", window_bp=4375,
                auroc=m["auroc"], auprc=m["auprc"], source=str(npz),
                script="scripts/brca_v50mane_spf_metrics.py", checkpoint=str(ckpt),
                git_commit=commit,
                data_path=[str(source), str(FULL), str(SPLIT),
                           str(out / f"inputs_fold{FOLD}.json")],
                notes=("v50 MANE reported-gene SPF head, fold 3 only; BRCA1 and BRCA2 "
                       "are both in fold 3's test split, so this head used neither "
                       "gene for training nor for early-stopping validation. The "
                       "expert set repeats ClinVar held-out variants and is not an "
                       "independent external cohort.")))
    report["leakage_check"] = leak
    report["checkpoint_sha256"] = digest(ckpt)
    report["git_commit"] = commit
    (out / "brca_metrics_v50mane.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (out / "registry.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records))
    (out / "complete.json").write_text(json.dumps(dict(
        scoring_complete=True, fold=FOLD, head_epoch=int(ck["epoch"]),
        git_commit=commit, n_records=len(records)), indent=2) + "\n")
    print(f"wrote {out / 'brca_metrics_v50mane.json'} and {out / 'registry.jsonl'}",
          flush=True)


if __name__ == "__main__":
    main()
