"""Standalone model evaluator for IssaSplice.

Provides :class:`Evaluator` which handles model loading,
forward pass, metric computation, and DDP aggregation
without depending on :class:`train.train.SplicingTrainer`.
"""

import heapq
import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import IssaSplice
from train.online_metrics_calculator import (
    OnlineMetricsCalculator,
)
from train.trainer import IssaSpliceTrainer
from utils.distributed_utils import is_master_process
from utils.utils import (
    center_indices,
    compute_keep_mask_and_stats,
    log_only_master_process,
    safe_tensor_to_numpy,
)

logger = logging.getLogger(__name__)


@dataclass
class _ScoreLabelPairs:
    """Container for (score, label) pairs on CPU."""

    pairs: torch.Tensor

    def to(self, device: torch.device) -> "_ScoreLabelPairs":
        """Move pairs tensor to *device*.

        Args:
            device (torch.device): Target device.

        Returns:
            _ScoreLabelPairs: New container on *device*.
        """
        return _ScoreLabelPairs(self.pairs.to(device))


@dataclass
class _FilteredBatch:
    """Batch data after keep-mask filtering."""

    ref_seqs: list[str]
    labels: torch.Tensor
    keep: torch.Tensor


@dataclass
class _RegressionErrors:
    """Accumulated PSI regression error statistics."""

    acc_sse: float = 0.0
    acc_n: int = 0
    don_sse: float = 0.0
    don_n: int = 0
    acc_mae: float = 0.0
    don_mae: float = 0.0

    def __iadd__(self, other: "_RegressionErrors") -> "_RegressionErrors":
        """Accumulate regression errors in-place.

        Args:
            other (_RegressionErrors): Errors to add.

        Returns:
            _RegressionErrors: Self after accumulation.
        """
        self.acc_sse += other.acc_sse
        self.acc_n += other.acc_n
        self.don_sse += other.don_sse
        self.don_n += other.don_n
        self.acc_mae += other.acc_mae
        self.don_mae += other.don_mae
        return self

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Serialise to a float64 tensor for DDP reduction.

        Args:
            device (torch.device): Target device.

        Returns:
            torch.Tensor: Shape [6] float64 tensor.
        """
        return torch.tensor(
            [
                self.acc_sse,
                float(self.acc_n),
                self.don_sse,
                float(self.don_n),
                self.acc_mae,
                self.don_mae,
            ],
            device=device,
            dtype=torch.float64,
        )

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "_RegressionErrors":
        """Deserialise from a float64 tensor.

        Args:
            t (torch.Tensor): Shape [6] tensor produced
                by :meth:`to_tensor`.

        Returns:
            _RegressionErrors: Restored instance.
        """
        return cls(
            acc_sse=t[0].item(),
            acc_n=int(t[1].item()),
            don_sse=t[2].item(),
            don_n=int(t[3].item()),
            acc_mae=t[4].item(),
            don_mae=t[5].item(),
        )


@dataclass
class _BatchResult:
    """Result of a single evaluation batch."""

    probs: torch.Tensor | None
    labels: torch.Tensor | None
    soft_labels: torch.Tensor | None
    reg_errors: _RegressionErrors


class Evaluator:
    """Standalone evaluator for IssaSplice checkpoints.

    Handles model initialisation, checkpoint loading, DDP
    wrapping, tokenisation, forward pass, label alignment,
    and distributed metric aggregation.
    """

    _EMPTY_PAIRS = _ScoreLabelPairs(torch.empty((0, 2), dtype=torch.float32))

    def __init__(
        self,
        cfg: DictConfig,
        cfg_model: DictConfig,
        *,
        local_rank: int,
        world_size: int,
    ) -> None:
        """Initialise the evaluator.

        Args:
            cfg (DictConfig): Full evaluation config
                (model, training, data, logging, mlflow).
            cfg_model (DictConfig): Legacy model config
                (config/issasplice/model.yaml).
            local_rank (int): Local GPU rank.
            world_size (int): Total number of processes.
        """
        self.cfg = cfg
        self.cfg_model = cfg_model
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_distributed = world_size > 1

        self.logger = logging.getLogger(self.__class__.__name__)
        if not is_master_process():
            self.logger.setLevel(logging.WARNING)

        self._init_device()
        self._init_model()
        self._init_flags()

        return

    # ── Device ──────────────────────────────────────────

    def _init_device(self) -> None:
        """Resolve the compute device.

        Uses *local_rank* under DDP; otherwise the first GPU visible to the
        process. GPU allocation is owned by the launcher (Slurm ``srun`` /
        torchrun), which sets ``CUDA_VISIBLE_DEVICES`` — never read from config.
        """
        if self.is_distributed:
            # % device_count keeps the index valid under Slurm per-task GPU
            # isolation, where a worker may see only its single allocated GPU.
            local_device = self.local_rank % torch.cuda.device_count()
            self.device = torch.device(f"cuda:{local_device}")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        log_only_master_process(
            self.logger,
            "info",
            f"Device: {self.device}",
        )

        return

    # ── Model ───────────────────────────────────────────

    def _init_model(self) -> None:
        """Create, load weights, and optionally DDP-wrap.

        The model is created through
        :meth:`IssaSpliceTrainer.init_model`, then moved to
        device, checkpoint loaded, and wrapped with DDP when
        running distributed.
        """
        base_trainer = IssaSpliceTrainer(self.cfg)
        self.model: IssaSplice | DDP = base_trainer.init_model(
            verbose=is_master_process()
        ).to(self.device)

        ckpt = self.cfg.model.initial_weights_path
        if ckpt:
            log_only_master_process(
                self.logger,
                "info",
                f"Loading checkpoint: {ckpt}",
            )
            actual = self._unwrap_model()
            actual.load_model_from_checkpoint(ckpt)
            log_only_master_process(
                self.logger,
                "info",
                "Checkpoint loaded successfully",
            )

        self.model_type: str = self.cfg.model.model_type.lower()

        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=True,
            )
            log_only_master_process(
                self.logger,
                "info",
                "Model wrapped with DDP",
            )

        return

    def _unwrap_model(self) -> IssaSplice:
        """Get the actual model, unwrapping DDP if needed.

        Returns:
            IssaSplice: Unwrapped model.
        """
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    # ── Flags ───────────────────────────────────────────

    def _init_flags(self) -> None:
        """Read evaluation-relevant flags from config.

        Soft labels and regression were removed in the v2 config revamp;
        their flags are pinned to safe-off values here so the legacy
        evaluation paths short-circuit.
        """
        # Soft labels and regression were removed in the v2 revamp.
        self.use_soft_labels: bool = False
        self.is_regression_enabled: bool = False
        self.uses_hard_labels_cls: bool = bool(
            OmegaConf.select(
                self.cfg,
                "training.classification.use_hard_labels",
                default=False,
            )
        )
        self.uses_hard_labels_reg: bool = False
        raw_prob = OmegaConf.select(
            self.cfg,
            "evaluation.probability_mode",
            default=None,
        )
        if raw_prob is None:
            raise KeyError(
                "evaluation.probability_mode must be set (sigmoid or softmax)"
            )
        self.probability_mode: str = str(raw_prob).lower()

        return

    # ── Encoding ────────────────────────────────────────

    def _encode_sequences(
        self, sequences: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Tokenise DNA sequences via model tokenizer.

        Args:
            sequences (list[str]): DNA strings.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]:
                (input_ids, attention_mask).
        """
        actual = self._unwrap_model()
        max_len = self.cfg.model.tokenizer_max_length

        if self.model_type == "hyenadna":
            enc = actual.tokenizer(
                sequences,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_offsets_mapping=False,
                add_special_tokens=False,
            )
            return (
                enc["input_ids"].to(self.device),
                None,
            )

        if self.model_type == "nucleotide-transformer":
            enc = actual.tokenizer(
                sequences,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_len,
            )
            return (
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
            )

        if self.model_type == "neobert":
            enc = actual.tokenizer(
                sequences,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_len,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(self.device)
            from omegaconf import OmegaConf

            skip_pad_mask = OmegaConf.select(
                self.cfg, "model.neobert.skip_padding_mask", default=False
            )
            if skip_pad_mask:
                return input_ids, None
            return input_ids, enc["attention_mask"].to(self.device)

        if self.model_type == "modernbert":
            # ModernBERT (feature 011) mirrors NeoBERT's tokenisation path. The
            # fixed-length splicing input has no padding so the attention mask
            # is omitted, letting the SDPA backend pick FlashAttention-2.
            enc = actual.tokenizer(
                sequences,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_len,
                add_special_tokens=False,
            )
            return enc["input_ids"].to(self.device), None

        raise NotImplementedError(f"Tokenisation not implemented for {self.model_type}")

    # ── Forward ─────────────────────────────────────────

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run model forward pass.

        Args:
            input_ids (torch.Tensor): Token IDs [B, L].
            attention_mask (torch.Tensor | None): Mask.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]:
                ``(cls_logits [B, L, C], None)``. The trailing ``None`` is
                kept for ABI compatibility with the legacy regression-aware
                call sites; the regression head was removed in the v2
                config revamp.
        """
        model = (
            self.model
            if self.model.training or not self.is_distributed
            else self._unwrap_model()
        )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Model.forward returns (cls, aux_dict); aux heads are not consumed
        # by evaluation, so drop them here. The previous interface returned
        # (cls, reg, aux); preserve a 2-tuple ``(cls, None)`` for callers.
        if isinstance(outputs, tuple) and len(outputs) == 2:
            cls_logits, _ = outputs
            return cls_logits, None
        if isinstance(outputs, tuple) and len(outputs) == 3:
            cls_logits, reg_out, _ = outputs
            return cls_logits, reg_out
        return outputs

    # ── Label helpers ───────────────────────────────────

    @staticmethod
    def _normalize_label_shape(
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Ensure labels are [B, 3, L].

        Args:
            labels (torch.Tensor): Labels tensor.

        Returns:
            torch.Tensor: [B, 3, L].

        Raises:
            ValueError: On unexpected shape.
        """
        if labels.dim() == 3 and labels.shape[1] == 3:
            return labels
        if labels.dim() == 2 and labels.shape[0] == 3:
            return labels.unsqueeze(0)
        raise ValueError(f"Unexpected labels shape: {labels.shape}")

    @staticmethod
    def _align(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Center-crop logits [B,L,C] and labels [B,3,L].

        Args:
            logits (torch.Tensor): [B, L, C].
            labels (torch.Tensor): [B, 3, L].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Cropped
                (logits, labels).
        """
        lg_len = logits.shape[1]
        lb_len = labels.shape[2]
        if lg_len == lb_len:
            return logits, labels

        tgt = min(lg_len, lb_len)
        s, e = center_indices(lg_len, tgt)
        logits = logits[:, s:e, :]
        s, e = center_indices(lb_len, tgt)
        labels = labels[:, :, s:e]
        return logits, labels

    @staticmethod
    def _align_regression(
        reg: torch.Tensor,
        tgt: torch.Tensor,
        ref_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Center-crop regression outputs & targets.

        Args:
            reg (torch.Tensor): [B, L, 2].
            tgt (torch.Tensor): [B, L, 2].
            ref_len (int): Reference length.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Cropped
                (reg, tgt).
        """
        out_l = reg.shape[1]
        tgt_l = tgt.shape[1]
        t = min(ref_len, out_l, tgt_l)
        if out_l == t and tgt_l == t:
            return reg, tgt
        s, e = center_indices(out_l, t)
        reg = reg[:, s:e, :]
        s, e = center_indices(tgt_l, t)
        tgt = tgt[:, s:e, :]
        return reg, tgt

    def _to_hard_labels_cls(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert soft labels → hard (classification).

        Keeps only the center position of consecutive
        positive regions per channel.

        Args:
            labels (torch.Tensor): [B, 3, L].

        Returns:
            torch.Tensor: Hard labels [B, 3, L].
        """
        hard = labels.clone()
        a_win = self.cfg.training.classification.acceptor_window_size
        d_win = self.cfg.training.classification.donor_window_size
        hard[:, 0, :] = _find_center_of_nonzero(hard[:, 0, :], a_win, keep_value=False)
        hard[:, 1, :] = _find_center_of_nonzero(hard[:, 1, :], d_win, keep_value=False)
        return hard

    def _to_hard_labels_reg(self, psi: torch.Tensor) -> torch.Tensor:
        """Convert soft PSI targets → hard (regression).

        Args:
            psi (torch.Tensor): [B, L, 2].

        Returns:
            torch.Tensor: Hard PSI targets [B, L, 2].
        """
        hard = torch.zeros_like(psi)
        a_win = self.cfg.training.regression.acceptor_window_size
        d_win = self.cfg.training.regression.donor_window_size
        hard[:, :, 0] = _find_center_of_nonzero(psi[:, :, 0], a_win, keep_value=True)
        hard[:, :, 1] = _find_center_of_nonzero(psi[:, :, 1], d_win, keep_value=True)
        return hard

    # ── DDP gather helpers ──────────────────────────────

    def _gather_to_master(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """Gather variable-size tensors to rank 0.

        Args:
            tensor (torch.Tensor): Local tensor.

        Returns:
            torch.Tensor | None: Concatenated on master,
                None on others.
        """
        if not self.is_distributed:
            return tensor

        tensor = tensor.to(self.device)
        local_sz = torch.tensor(
            [tensor.shape[0]],
            device=self.device,
            dtype=torch.long,
        )
        all_sz = [
            torch.zeros(1, dtype=torch.long, device=self.device)
            for _ in range(self.world_size)
        ]
        dist.all_gather(all_sz, local_sz)
        max_sz = max(s.item() for s in all_sz)

        if tensor.shape[0] < max_sz:
            pad = torch.zeros(
                (max_sz - tensor.shape[0],) + tensor.shape[1:],
                dtype=tensor.dtype,
                device=self.device,
            )
            tensor = torch.cat([tensor, pad], dim=0)

        if is_master_process():
            gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        else:
            gathered = None

        dist.gather(tensor, gather_list=gathered, dst=0)

        if is_master_process() and gathered is not None:
            parts = [
                gathered[i][: int(all_sz[i].item())].cpu()
                for i in range(self.world_size)
            ]
            del gathered
            torch.cuda.empty_cache()
            return torch.cat(parts, dim=0)

        return None

    # ── Top-k helpers ───────────────────────────────────

    @staticmethod
    def _local_topk_pairs(
        scores: torch.Tensor,
        labels: torch.Tensor,
        k: int,
        chunk_size: int = 1_000_000,
    ) -> _ScoreLabelPairs:
        """Select local top-k (score, label) pairs.

        Args:
            scores (torch.Tensor): Score tensor.
            labels (torch.Tensor): Label tensor.
            k (int): Number of top pairs.
            chunk_size (int): Chunk size for streaming.

        Returns:
            _ScoreLabelPairs: Pairs on CPU.
        """
        if k <= 0 or scores.numel() == 0:
            return Evaluator._EMPTY_PAIRS

        sf = scores.reshape(-1)
        lf = labels.reshape(-1)
        n = sf.numel()
        k_eff = min(k, n)

        top_s = torch.empty((0,), dtype=torch.float32, device="cpu")
        top_l = torch.empty((0,), dtype=torch.float32, device="cpu")

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            cs = sf[start:end].detach().float().cpu()
            cl = lf[start:end].detach().cpu().float().gt(0.5).float()
            kc = min(k_eff, cs.numel())
            if kc <= 0:
                continue

            cand_s, cand_i = torch.topk(cs, k=kc, largest=True, sorted=False)
            cand_l = cl[cand_i]

            if top_s.numel() == 0:
                top_s, top_l = cand_s, cand_l
                continue

            ms = torch.cat([top_s, cand_s])
            ml = torch.cat([top_l, cand_l])
            if ms.numel() > k_eff:
                ns, ni = torch.topk(
                    ms,
                    k=k_eff,
                    largest=True,
                    sorted=False,
                )
                top_s, top_l = ns, ml[ni]
            else:
                top_s, top_l = ms, ml

        pairs = torch.stack([top_s, top_l], dim=1)
        return _ScoreLabelPairs(pairs)

    @staticmethod
    def _merge_topk_pairs(
        current: _ScoreLabelPairs,
        scores: torch.Tensor,
        labels: torch.Tensor,
        k: int,
    ) -> _ScoreLabelPairs:
        """Merge one score chunk into current top-k pair set.

        Args:
            current (_ScoreLabelPairs): Current top pairs.
            scores (torch.Tensor): Candidate scores.
            labels (torch.Tensor): Candidate labels.
            k (int): Number of top pairs to keep.

        Returns:
            _ScoreLabelPairs: Updated top score-label pairs.
        """
        if k <= 0:
            return Evaluator._EMPTY_PAIRS

        incoming = Evaluator._local_topk_pairs(scores, labels, k)
        if incoming.pairs.numel() == 0:
            return current
        if current.pairs.numel() == 0:
            return incoming

        merged = torch.cat([current.pairs, incoming.pairs], dim=0)
        if merged.shape[0] <= k:
            return _ScoreLabelPairs(merged)

        top_idx = torch.topk(merged[:, 0], k=k, largest=True, sorted=False).indices
        return _ScoreLabelPairs(merged[top_idx])

    def _gather_pairs(self, local: _ScoreLabelPairs) -> _ScoreLabelPairs | None:
        """Gather pairs to master.

        Args:
            local (_ScoreLabelPairs): Local pairs.

        Returns:
            _ScoreLabelPairs | None: Gathered on master.
        """
        if not self.is_distributed:
            return local.to(torch.device("cpu"))

        gathered = self._gather_to_master(local.pairs.to(self.device))
        if is_master_process() and gathered is not None:
            return _ScoreLabelPairs(gathered.cpu())
        return None

    @staticmethod
    def _topk_accuracy_from_pairs(pairs: _ScoreLabelPairs, k: int) -> float:
        """Compute top-k accuracy from gathered pairs.

        Args:
            pairs (_ScoreLabelPairs): Pairs container.
            k (int): Global k.

        Returns:
            float: Top-k accuracy in [0, 1], or
                ``nan`` when *k* <= 0 or pairs are
                empty (metric is undefined).
        """
        if k <= 0 or pairs.pairs.numel() == 0:
            return float("nan")
        p = pairs.pairs
        if p.device.type != "cpu":
            p = p.cpu()
        heap: list[tuple[float, int]] = []
        for i in range(p.shape[0]):
            s = float(p[i, 0].item())
            lbl = 1 if p[i, 1].item() > 0.5 else 0
            if len(heap) < k:
                heapq.heappush(heap, (s, lbl))
            elif s > heap[0][0]:
                heapq.heapreplace(heap, (s, lbl))
        hits = sum(lbl for _, lbl in heap)
        return float(hits / k)

    # ── Single batch ────────────────────────────────────

    def _filter_batch(
        self,
        batch: dict[str, Any],
    ) -> _FilteredBatch | None:
        """Filter batch samples via keep-mask.

        Extracts labels, computes the keep mask based on
        sequence validity, then returns only the surviving
        samples.

        Args:
            batch (dict[str, Any]): Batch from DataLoader.

        Returns:
            _FilteredBatch | None: Filtered data, or None
                if all samples are filtered out.

        Raises:
            KeyError: If neither label key is present.
        """
        ref_seqs = batch["ref"]
        lbl_key = "soft_labels" if self.use_soft_labels else "labels"
        orig_labels = batch.get(lbl_key, batch.get("labels"))
        if orig_labels is None:
            raise KeyError(f"Missing '{lbl_key}' or 'labels'")

        filter_loss_type = (
            "multi_class_focal_loss"
            if self.probability_mode == "softmax"
            else "sigmoid"
        )
        keep, _ = compute_keep_mask_and_stats(
            ref_seqs=ref_seqs,
            original_labels=orig_labels,
            seq_length=int(self.cfg.model.seq_length),
            loss_type=filter_loss_type,
        )

        if not torch.any(keep):
            return None

        ref_seqs = [s for i, s in enumerate(ref_seqs) if keep[i]]
        labels = orig_labels[keep].to(self.device)

        if not isinstance(ref_seqs, list):
            ref_seqs = list(ref_seqs)

        return _FilteredBatch(
            ref_seqs=ref_seqs,
            labels=labels,
            keep=keep,
        )

    def _compute_classification_outputs(
        self,
        cls_logits: torch.Tensor,
        labels: torch.Tensor,
        batch: dict[str, Any],
        keep: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Compute classification probs and aligned labels.

        Args:
            cls_logits (torch.Tensor): Raw logits [B,L,C].
            labels (torch.Tensor): Labels tensor [B,3,L].
            batch (dict[str, Any]): Original batch dict.
            keep (torch.Tensor): Boolean keep mask.

        Returns:
            tuple[torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None]:
                (probs, labels_transposed,
                soft_labels_transposed).
        """
        labels = self._normalize_label_shape(labels)
        cls_logits, labels = self._align(cls_logits, labels)

        if self.uses_hard_labels_cls:
            labels = self._to_hard_labels_cls(labels)

        soft_m: torch.Tensor | None = None
        try:
            sl_full = batch.get("soft_labels")
            if sl_full is not None:
                sl = sl_full[keep].to(self.device)
                sl = self._normalize_label_shape(sl)
                _, sl = self._align(cls_logits, sl)
                soft_m = sl.transpose(1, 2).contiguous()
        except Exception:
            pass

        if self.probability_mode == "softmax":
            probs = torch.softmax(cls_logits, dim=-1)
        else:
            probs = torch.sigmoid(cls_logits)
        labels_m = labels.transpose(1, 2).contiguous()

        return probs, labels_m, soft_m

    def _compute_regression_errors(
        self,
        batch: dict[str, Any],
        keep: torch.Tensor,
        reg_out: torch.Tensor | None,
        cls_seq_len: int,
    ) -> _RegressionErrors:
        """Compute PSI regression errors for non-zero sites.

        Args:
            batch (dict[str, Any]): Original batch dict.
            keep (torch.Tensor): Boolean keep mask.
            reg_out (torch.Tensor | None): Regression
                model output.
            cls_seq_len (int): Classification sequence
                length used for alignment.

        Returns:
            _RegressionErrors: Per-site-type SSE, MAE,
                and counts for non-zero positions.
        """
        don_psi = batch.get("donor_psi")
        acc_psi = batch.get("acceptor_psi")
        if not self.is_regression_enabled or don_psi is None or acc_psi is None:
            return _RegressionErrors()

        try:
            dp = don_psi[keep].to(self.device).float()
            ap = acc_psi[keep].to(self.device).float()
            psi_tgt = torch.stack([ap, dp], dim=-1)

            if reg_out is None:
                raise RuntimeError("Regression outputs are None")
            reg_out, psi_tgt = self._align_regression(
                reg_out,
                psi_tgt,
                cls_seq_len,
            )
            if self.uses_hard_labels_reg:
                psi_tgt = self._to_hard_labels_reg(psi_tgt)

            err_sq = (reg_out.float() - psi_tgt) ** 2
            err_abs = torch.abs(reg_out.float() - psi_tgt)

            result = _RegressionErrors()

            am = psi_tgt[..., 0] != 0
            dm = psi_tgt[..., 1] != 0
            if am.any():
                result.acc_sse = err_sq[..., 0][am].sum().item()
                result.acc_mae = err_abs[..., 0][am].sum().item()
                result.acc_n = int(am.sum().item())
            if dm.any():
                result.don_sse = err_sq[..., 1][dm].sum().item()
                result.don_mae = err_abs[..., 1][dm].sum().item()
                result.don_n = int(dm.sum().item())

            return result
        except Exception as exc:
            log_only_master_process(
                self.logger,
                "warning",
                f"PSI regression skipped: {exc}",
            )
            return _RegressionErrors()

    def _process_batch(
        self,
        batch: dict[str, Any],
    ) -> _BatchResult:
        """Process one evaluation batch.

        Orchestrates filtering, forward pass,
        classification output computation, and
        regression error calculation.

        Args:
            batch (dict[str, Any]): Batch from DataLoader.

        Returns:
            _BatchResult: Classification outputs and
                regression errors for this batch.
        """
        empty = _BatchResult(
            probs=None,
            labels=None,
            soft_labels=None,
            reg_errors=_RegressionErrors(),
        )

        filtered = self._filter_batch(batch)
        if filtered is None:
            return empty

        ids, attn = self._encode_sequences(filtered.ref_seqs)
        cls_logits, reg_out = self._forward(ids, attn)

        probs, labels_m, soft_m = self._compute_classification_outputs(
            cls_logits,
            filtered.labels,
            batch,
            filtered.keep,
        )

        reg_errors = self._compute_regression_errors(
            batch,
            filtered.keep,
            reg_out,
            cls_logits.shape[1],
        )

        return _BatchResult(
            probs=probs,
            labels=labels_m,
            soft_labels=soft_m,
            reg_errors=reg_errors,
        )

    # ── Batch-level helpers for streaming evaluate ─────

    def _apply_thresholds(
        self,
        probs: torch.Tensor,
        thresholds: tuple[float, float, float] | None,
    ) -> torch.Tensor:
        """Convert probabilities to binary predictions.

        Uses explicit *thresholds* when provided, otherwise
        falls back to argmax (softmax) or 0.5 (sigmoid).

        Args:
            probs (torch.Tensor): Probabilities [B, L, 2].
            thresholds (tuple[float, float, float] | None):
                Per-class thresholds (acceptor, donor, …).

        Returns:
            torch.Tensor: Binary predictions [B, L, 2].
        """
        if thresholds is not None:
            thresholds_tensor = torch.tensor(
                thresholds[:2], device=probs.device, dtype=probs.dtype
            )
            return (probs >= thresholds_tensor).float()

        if self.probability_mode == "softmax":
            pi = torch.argmax(probs, dim=-1)
            return torch.nn.functional.one_hot(pi, num_classes=probs.shape[-1]).float()

        return (probs >= 0.5).float()

    def _compute_topk_accuracy(
        self,
        local_pairs: _ScoreLabelPairs,
        k: int,
    ) -> float:
        """Gather local top-k pairs and compute global
        top-k accuracy.

        Accepts per-rank local top-k candidate pairs,
        gathers them across ranks in distributed mode,
        then selects the global top-k to compute
        precision@k (fraction of true positives among
        the k highest-scoring positions).

        In distributed mode every rank must call this
        method because it contains a collective gather.

        Args:
            local_pairs (_ScoreLabelPairs): Per-rank
                top-k (score, label) candidate pairs.
            k (int): Global number of top positions.

        Returns:
            float: Top-k accuracy in [0, 1], or
                ``nan`` when *k* <= 0 (metric is
                undefined).  In DDP mode, only the
                master rank returns the true value;
                others return ``nan``.
        """
        if k <= 0:
            return float("nan")

        if self.is_distributed:
            gathered = self._gather_pairs(local_pairs)
            if is_master_process() and gathered is not None:
                return self._topk_accuracy_from_pairs(gathered, k)
            return float("nan")

        if local_pairs.pairs.shape[0] < k:
            return float("nan")
        return self._topk_accuracy_from_pairs(local_pairs, k)

    def _compute_topk_pairs_for_loader(
        self,
        loader: DataLoader,
        thresholds: tuple[float, float, float] | None,
        k_acceptor: int,
        k_donor: int,
    ) -> tuple[_ScoreLabelPairs, _ScoreLabelPairs]:
        """Compute local top-k pairs for all evaluation batches.

        This performs a second streaming pass over the
        loader to avoid storing all scores in memory.

        Args:
            loader (DataLoader): Evaluation data loader.
            thresholds (tuple[float, float, float] | None):
                Unused. Kept for call-shape compatibility.
            k_acceptor (int): Top-k for acceptor channel.
            k_donor (int): Top-k for donor channel.

        Returns:
            tuple[_ScoreLabelPairs, _ScoreLabelPairs]:
                Local pairs for (acceptor, donor).
        """
        del thresholds

        pairs_a = Evaluator._EMPTY_PAIRS
        pairs_d = Evaluator._EMPTY_PAIRS

        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Top-k pass",
                disable=False,
            ):
                result = self._process_batch(batch)
                if result.probs is None or result.labels is None:
                    continue

                lbl = (
                    result.soft_labels
                    if (self.use_soft_labels and result.soft_labels is not None)
                    else result.labels
                )
                probs_2 = result.probs[..., :2].detach().cpu()
                labels_2 = lbl[..., :2].detach().cpu()

                s_a = probs_2[..., 0].reshape(-1)
                l_a = (labels_2[..., 0] > 0.5).float().reshape(-1)
                s_d = probs_2[..., 1].reshape(-1)
                l_d = (labels_2[..., 1] > 0.5).float().reshape(-1)

                pairs_a = self._merge_topk_pairs(
                    current=pairs_a,
                    scores=s_a,
                    labels=l_a,
                    k=k_acceptor,
                )
                pairs_d = self._merge_topk_pairs(
                    current=pairs_d,
                    scores=s_d,
                    labels=l_d,
                    k=k_donor,
                )

        return pairs_a, pairs_d

    # ── Main evaluate loop ──────────────────────────────

    def evaluate(
        self,
        loader: DataLoader,
        *,
        thresholds: (tuple[float, float, float] | None) = None,
    ) -> dict[str, float]:
        """Run evaluation on *loader*.

        Metrics are computed in a streaming fashion:
        AUC / confusion-matrix data is fed to
        :class:`OnlineMetricsCalculator` per batch,
        and only compact score-label pairs are kept
        for top-k accuracy.

        Args:
            loader (DataLoader): Test data loader.
            thresholds (tuple[float, float, float] | None):
                Optional per-class thresholds
                (acceptor, donor, neither).

        Returns:
            dict[str, float]: Evaluation metrics.
        """
        self.model.eval()

        mc = OnlineMetricsCalculator(
            logger=self.logger,
            num_classes=OmegaConf.select(
                self.cfg, "model.n_classification_classes", default=3
            ),
            use_histogram=bool(self.cfg.evaluation.use_histogram),
        )

        total_batches = len(loader)
        log_only_master_process(
            self.logger,
            "info",
            f"Evaluation: {total_batches} batches, {len(loader.dataset)} samples",
        )

        reg_acc = _RegressionErrors()
        pos_count_a = 0
        pos_count_d = 0

        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Evaluating",
                disable=False,
            ):
                result = self._process_batch(batch)
                reg_acc += result.reg_errors

                if result.probs is None or result.labels is None:
                    continue

                lbl = (
                    result.soft_labels
                    if (self.use_soft_labels and result.soft_labels is not None)
                    else result.labels
                )

                probs_2 = result.probs[..., :2].detach().cpu()
                labels_2 = lbl[..., :2].detach().cpu()

                probs_np = safe_tensor_to_numpy(probs_2)
                labels_np = safe_tensor_to_numpy(labels_2)
                mc.update_auc_data(labels=labels_np, probs=probs_np)

                preds_2 = self._apply_thresholds(probs_2, thresholds)
                preds_np = safe_tensor_to_numpy(preds_2)
                mc.update(preds=preds_np, labels=labels_np)

                l_a = (labels_2[..., 0] > 0.5).float().reshape(-1)
                l_d = (labels_2[..., 1] > 0.5).float().reshape(-1)

                pos_count_a += int(l_a.sum().item())
                pos_count_d += int(l_d.sum().item())

        # ── DDP reductions ───────────────────────────────
        if self.is_distributed:
            if self.is_regression_enabled:
                buf = reg_acc.to_tensor(self.device)
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                reg_acc = _RegressionErrors.from_tensor(buf)

            mc.distributed_reduce()

            pc = torch.tensor(
                [pos_count_a, pos_count_d],
                device=self.device,
                dtype=torch.long,
            )
            dist.all_reduce(pc, op=dist.ReduceOp.SUM)
            topk_k_a = int(pc[0].item())
            topk_k_d = int(pc[1].item())
        else:
            topk_k_a = pos_count_a
            topk_k_d = pos_count_d

        # ── Top-k accuracy via second streaming pass ─────
        pairs_a, pairs_d = self._compute_topk_pairs_for_loader(
            loader=loader,
            thresholds=thresholds,
            k_acceptor=topk_k_a,
            k_donor=topk_k_d,
        )
        topk_acc_a = self._compute_topk_accuracy(
            local_pairs=pairs_a,
            k=topk_k_a,
        )
        topk_acc_d = self._compute_topk_accuracy(
            local_pairs=pairs_d,
            k=topk_k_d,
        )

        # ── Assemble metrics ────────────────────────────
        metrics: dict[str, float] = {}

        skip = self.is_distributed and not is_master_process()
        if not skip:
            metrics = mc.compute_metrics()
            metrics["topk_accuracy_acceptor"] = topk_acc_a
            metrics["topk_accuracy_donor"] = topk_acc_d
            metrics["topk_k_acceptor"] = float(topk_k_a)
            metrics["topk_k_donor"] = float(topk_k_d)

        if self.is_regression_enabled:
            if reg_acc.acc_n > 0:
                metrics["regression_mse_acceptor_nonzero"] = (
                    reg_acc.acc_sse / reg_acc.acc_n
                )
                metrics["regression_mae_acceptor_nonzero"] = (
                    reg_acc.acc_mae / reg_acc.acc_n
                )
            if reg_acc.don_n > 0:
                metrics["regression_mse_donor_nonzero"] = (
                    reg_acc.don_sse / reg_acc.don_n
                )
                metrics["regression_mae_donor_nonzero"] = (
                    reg_acc.don_mae / reg_acc.don_n
                )

        log_only_master_process(
            self.logger,
            "info",
            "Evaluation complete",
        )

        del mc

        return metrics


# ── Module-level helpers ────────────────────────────────


def _find_center_of_nonzero(
    channel: torch.Tensor,
    window_size: int,
    *,
    keep_value: bool = False,
) -> torch.Tensor:
    """Find centers of consecutive non-zero regions.

    Vectorised implementation.

    Args:
        channel (torch.Tensor): [B, L].
        window_size (int): Expected non-zero width.
        keep_value (bool): Retain original value at center.

    Returns:
        torch.Tensor: [B, L] with only centers marked.
    """
    bs, sl = channel.shape
    dev = channel.device

    nz = (channel != 0.0).to(torch.int8)
    padded = torch.cat(
        [torch.zeros(bs, 1, device=dev, dtype=torch.int8), nz],
        dim=1,
    )
    diff = torch.diff(padded, dim=1)
    is_start = diff == 1

    sb, sp = torch.where(is_start)
    if len(sb) == 0:
        return torch.zeros_like(channel)

    co = window_size // 2
    cp = sp + co

    valid = cp < sl
    vb = sb[valid]
    vc = cp[valid]

    hard = torch.zeros_like(channel)
    if keep_value:
        hard[vb, vc] = channel[vb, vc]
    else:
        hard[vb, vc] = 1.0

    return hard
