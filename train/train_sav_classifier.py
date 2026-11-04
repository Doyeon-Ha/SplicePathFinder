"""학습형 SAV 분류기 학습 루프 (plain PyTorch, spec 015 US2).

per-nt feature 캐시를 gene-grouped로 분할하고, 변이 중심 윈도 → RBF →
단방향 Mamba2 → mean pool → MLP를 BCEWithLogits로 학습한다. val AUROC로
조기 종료와 체크포인트 선택을 수행한다
(기본값 참). frozen base model은 이 루프에 포함되지 않는다(캐시가 경계).
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from model.sav_classifier import RbfMamba2SavClassifier  # noqa: E402
from process_input.gene_grouped_split import (  # noqa: E402
    assert_no_gene_leakage,
    gene_grouped_split,
)
from train.dataset_sav import SavFeatureDataset  # noqa: E402

logger = logging.getLogger(__name__)


def _build_model(
    cfg: DictConfig, num_feature_channels: int = 0, num_prob_channels: int = 6,
    num_scalar_features: int = 0,
) -> tuple[RbfMamba2SavClassifier, dict]:
    """config로부터 분류기와 재구성용 kwargs를 만든다.

    Args:
        cfg (DictConfig): 학습 설정.
        num_feature_channels (int): 데이터셋이 concat할 피처 채널 수
            (spec 017; 0이면 확률-only).
        num_prob_channels (int): 확률 채널 수 = ``2 * len(channel_indices)``
            (dataset이 동적 산출; med_ac/SpliceAI=6, Pangolin 메인=8; spec 019).
        num_scalar_features (int): per-variant 스칼라 피처 수(pool 뒤 concat).

    Returns:
        tuple[RbfMamba2SavClassifier, dict]: 모델과 생성 kwargs.
    """
    feat_cfg = cfg.get("feature", {})
    kwargs = {
        "num_prob_channels": num_prob_channels,
        "num_feature_channels": num_feature_channels,
        "num_scalar_features": num_scalar_features,
        "feature_rbf_expand": bool(feat_cfg.get("rbf_expand", False)),
        "rbf_num_centers": int(cfg.rbf.num_centers),
        "rbf_gamma": cfg.rbf.gamma,
        "rbf_trainable": bool(cfg.rbf.trainable),
        "rbf_trainable_gamma": bool(cfg.rbf.trainable_gamma),
        "use_marker": bool(cfg.marker.enabled),
        "d_model": int(cfg.model.d_model),
        "n_layers": int(cfg.model.n_layers),
        "d_state": int(cfg.model.d_state),
        "d_conv": int(cfg.model.d_conv),
        "expand": int(cfg.model.expand),
        "headdim": int(cfg.model.headdim),
        "dropout": float(cfg.model.dropout),
        "pool": str(cfg.model.pool),
        "bidirectional": bool(cfg.model.get("bidirectional", False)),
        "pool_coverage_mask": bool(cfg.model.get("pool_coverage_mask", False)),
        "delta_channels": bool(cfg.model.get("delta_channels", False)),
    }
    return RbfMamba2SavClassifier(**kwargs), kwargs


def restrict_to_mask(indices: np.ndarray, mask_npz: str, key: str) -> np.ndarray:
    """분할 인덱스를 boolean mask가 True인 행으로 좁힌다.

    splice-region universe(``coord_splice_fields_seed42.npz['splice']``)처럼
    캐시 행 순서로 정렬된 mask를 fold 분할에 적용할 때 쓴다.

    Args:
        indices (np.ndarray): 좁힐 행 인덱스.
        mask_npz (str): boolean mask를 담은 npz 경로.
        key (str): npz 내 mask 배열 키.

    Returns:
        np.ndarray: mask가 True인 인덱스만 남긴 배열.
    """
    mask = np.load(mask_npz)[key].astype(bool)
    return np.asarray(indices)[mask[indices]]


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_marker: bool,
) -> tuple[float, float]:
    """loader에 대해 AUROC/AUPRC를 계산한다.

    Args:
        model (torch.nn.Module): 분류기(eval 모드).
        loader (DataLoader): 평가 데이터 로더.
        device (torch.device): 연산 디바이스.
        use_marker (bool): 마커 채널 사용 여부.

    Returns:
        tuple[float, float]: (AUROC, AUPRC). 단일 클래스면 (nan, nan).
    """
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        channels = batch["channels"].to(device, non_blocking=True)
        marker = batch["marker"].to(device, non_blocking=True) if use_marker else None
        scalars = (
            batch["scalars"].to(device, non_blocking=True)
            if "scalars" in batch
            else None
        )
        logits = model(channels, marker, scalars)
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(batch["label"].numpy())
    y_score = np.concatenate(scores)
    y_true = np.concatenate(labels)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(y_true, y_score)),
        float(average_precision_score(y_true, y_score)),
    )




def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_kwargs: dict,
    cfg: DictConfig,
    epoch: int,
    val_auroc: float | None,
) -> None:
    """재구성에 필요한 kwargs·config와 함께 체크포인트를 저장한다.

    Args:
        path (Path): 저장 경로.
        model (torch.nn.Module): 저장할 모델.
        model_kwargs (dict): 모델 재생성 kwargs.
        cfg (DictConfig): 학습 설정.
        epoch (int): 저장 시점 epoch.
        val_auroc (float | None): val AUROC. 검증 없는 arm이면 None.
    """
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_kwargs": model_kwargs,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "epoch": epoch,
            "val_auroc": val_auroc,
        },
        path,
    )


@hydra.main(
    version_base=None,
    config_path="../config/train",
    config_name="splicepathfinder_v50",
)
def main(cfg: DictConfig) -> None:
    """gene-grouped split으로 분류기를 학습한다.

    Args:
        cfg (DictConfig): Hydra 설정.
    """
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(int(cfg.split.seed))
    device = torch.device(cfg.train.device)

    with h5py.File(cfg.cache_h5, "r") as f:
        gene_symbols = f["gene_symbol"][:]
        labels = f["labels"][:].astype(np.int64)
    freeze_npz = cfg.split.get("freeze_npz", None)
    if freeze_npz:
        # 동결 split(예: ClinVar present_2022 gene-grouped) 사용 —
        # 내부 재계산 대신 정본 npz의 train/val/test 인덱스를 읽는다.
        split_npz = np.load(str(freeze_npz))
        train_idx = split_npz["train_idx"]
        val_idx = split_npz["val_idx"]
        test_idx = split_npz["test_idx"]
        logger.info(
            "동결 split 로드: %s (train=%d val=%d test=%d)",
            freeze_npz, len(train_idx), len(val_idx), len(test_idx),
        )
    else:
        train_idx, val_idx, test_idx = gene_grouped_split(
            gene_symbols,
            labels,
            seed=int(cfg.split.seed),
            ratios=tuple(cfg.split.ratios),
        )
    mask_npz = cfg.split.get("splice_mask_npz", None)
    if mask_npz:
        # splice-region universe로 좁힌 twin (end-to-end arm과 같은 학습 집합).
        key = str(cfg.split.get("splice_mask_key", "splice"))
        train_idx = restrict_to_mask(train_idx, str(mask_npz), key)
        val_idx = restrict_to_mask(val_idx, str(mask_npz), key)
        test_idx = restrict_to_mask(test_idx, str(mask_npz), key)
        logger.info(
            "splice mask 적용: %s[%s] (train=%d val=%d test=%d)",
            mask_npz, key, len(train_idx), len(val_idx), len(test_idx),
        )
    assert_no_gene_leakage(gene_symbols, train_idx, val_idx, test_idx)

    use_marker = bool(cfg.marker.enabled)
    half = int(cfg.window.half_bp)
    preload = bool(cfg.train.get("preload", False))
    feat_cfg = cfg.get("feature", {})
    feat_enabled = bool(feat_cfg.get("enabled", False))
    sidecar = (
        str(feat_cfg.get("sidecar_h5"))
        if feat_enabled and feat_cfg.get("sidecar_h5")
        else None
    )
    families = tuple(feat_cfg.get("families", [])) if feat_enabled else ()
    ci_cfg = cfg.get("channel_indices", None)
    channel_indices = tuple(ci_cfg) if ci_cfg is not None else None
    # per-variant scalar features (spec 019 conservation scalar-at-pool); enabled
    # only when feature.scalar_npz is set (independent of per-nt sidecar).
    scalar_npz = (
        str(feat_cfg.get("scalar_npz"))
        if feat_enabled and feat_cfg.get("scalar_npz")
        else None
    )
    scalar_key = str(feat_cfg.get("scalar_key", "scalars"))
    # window.left/right override half_bp for a cache whose window cannot be
    # symmetric about the variant (the multichunk cache's 8,750 is even).
    # Left as None the dataset takes them from the cache's own attributes.
    win_cfg = cfg.get("window", {})
    # ONBOARDING 4.54 masking, moved from the delta_max readout onto the head
    # input: alt is clamped to ref wherever a direction is impossible.
    mask_cfg = cfg.get("mask", None)
    canonical_mask = (
        str(mask_cfg["canonical_h5"])
        if mask_cfg is not None and mask_cfg.get("canonical_h5", None)
        else None
    )
    mask_snv_only = bool(mask_cfg.get("snv_only", False)) if mask_cfg else False
    if canonical_mask:
        logger.info("canonical masking ON: %s (snv_only=%s)",
                    canonical_mask, mask_snv_only)
    ds_kwargs = dict(
        window_half_bp=half,
        window_left=(int(win_cfg["left"]) if win_cfg.get("left") is not None
                     else None),
        window_right=(int(win_cfg["right"]) if win_cfg.get("right") is not None
                      else None),
        use_coverage=win_cfg.get("coverage", None),
        use_marker=use_marker,
        preload=preload,
        feature_sidecar_h5=sidecar,
        feature_families=families,
        channel_indices=channel_indices,
        scalar_features_npz=scalar_npz,
        scalar_features_key=scalar_key,
        canonical_mask_h5=canonical_mask,
        canonical_mask_snv_only=mask_snv_only,
    )
    train_ds = SavFeatureDataset(cfg.cache_h5, train_idx, **ds_kwargs)
    # val_idx가 비면 검증·조기종료 없이 고정 epoch 수만큼 학습하고 final.pt를 쓴다
    # (fold 분할 없이 전량 학습하는 배포용 arm).
    val_ds = (
        SavFeatureDataset(cfg.cache_h5, val_idx, **ds_kwargs)
        if len(val_idx) > 0
        else None
    )
    logger.info(
        "feature=%s families=%s num_feature_channels=%d "
        "channel_indices=%s num_prob_channels=%d",
        feat_enabled, families, train_ds.num_feature_channels,
        channel_indices, train_ds.num_prob_channels,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=int(cfg.train.batch_size),
            shuffle=False,
            num_workers=int(cfg.train.num_workers),
            pin_memory=True,
        )
        if val_ds is not None
        else None
    )

    model, model_kwargs = _build_model(
        cfg, train_ds.num_feature_channels, train_ds.num_prob_channels,
        train_ds.num_scalar_features,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    pos_weight_val = float(cfg.train.get("pos_weight", 1.0))
    pos_weight = (
        torch.tensor(pos_weight_val, device=device)
        if pos_weight_val != 1.0
        else None
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_auroc = float("-inf")
    best_epoch = -1
    patience = int(cfg.train.patience)

    for epoch in range(int(cfg.train.max_epochs)):
        model.train()
        running = 0.0
        for batch in train_loader:
            channels = batch["channels"].to(device, non_blocking=True)
            marker = (
                batch["marker"].to(device, non_blocking=True)
                if use_marker
                else None
            )
            scalars = (
                batch["scalars"].to(device, non_blocking=True)
                if "scalars" in batch
                else None
            )
            target = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(channels, marker, scalars)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        train_loss = running / max(1, len(train_loader))
        if val_loader is None:
            logger.info("epoch %d train_loss=%.4f (검증 없음)", epoch, train_loss)
            continue
        val_auroc, val_auprc = _evaluate(model, val_loader, device, use_marker)
        logger.info(
            "epoch %d train_loss=%.4f val_auroc=%.4f val_auprc=%.4f",
            epoch,
            train_loss,
            val_auroc,
            val_auprc,
        )
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            ckpt_path = out_dir / "best.pt"
            _save_checkpoint(
                ckpt_path, model, model_kwargs, cfg, epoch, val_auroc
            )
        if epoch - best_epoch >= patience:
            logger.info("early stop (epoch %d, best %d)", epoch, best_epoch)
            break
    if val_loader is None:
        last_epoch = int(cfg.train.max_epochs) - 1
        ckpt_path = out_dir / "final.pt"
        _save_checkpoint(ckpt_path, model, model_kwargs, cfg, last_epoch, None)
        logger.info(
            "검증 없는 arm — 마지막 epoch %d 저장: %s", last_epoch, ckpt_path
        )
    else:
        logger.info("best val_auroc=%.4f @epoch %d", best_auroc, best_epoch)


if __name__ == "__main__":
    main()
