"""Trainer helpers for the splicing pipeline (h5-only).

This module owns the dataset/loader plumbing and the model instantiation
glue used by ``train.py``. After the v2 config revamp the trainer assumes
the SpliceAI-style HDF5 format produced by the build pipeline; LMDB / PT
inputs and chromosome-based splits are no longer supported here.
"""

import contextlib
import io
import logging
import warnings
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset

from model.model import IssaSplice
from train.dataset import PangolinSSUDataset, SplicingH5Dataset


class IssaSpliceTrainer:
    """Trainer class for handling configuration loading and data loader creation."""

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize Trainer with configuration.

        Args:
            cfg (DictConfig): Loaded configuration object containing sections for
                data, model, training, and logging.
        """
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)
        # TF32 matmul on Ampere+ tensor cores — free speedup, AUROC-neutral (ONBOARDING §4.21).
        # Single chokepoint: all IssaSplice train + eval backbone loads build a trainer
        # (eval via evaluate_splicevardb_sav._load_hyenadna_model). Ampere-only; no-op elsewhere.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def _get_model_name_from_config(self) -> str:
        """Get model name from config.

        Uses ``model_options[model_index]`` when available,
        falls back to ``model_type`` otherwise (DD-007).

        Returns:
            str: Resolved model name.
        """
        model_options = OmegaConf.select(
            self.cfg, "model.model_options", default=None
        )
        model_index = OmegaConf.select(
            self.cfg, "model.model_index", default=0
        )

        if model_options and len(model_options) > 0:
            if not (0 <= model_index < len(model_options)):
                raise ValueError(
                    f"model_index {model_index} out"
                    f" of range. Available: "
                    f"0-{len(model_options) - 1}"
                )
            model_name = model_options[model_index]
        else:
            model_name = self.cfg.model.model_type
        self.logger.info(f"Selected model: {model_name}")
        return model_name

    def create_datasets(
        self,
    ) -> tuple[
        SplicingH5Dataset | Subset,
        SplicingH5Dataset | Subset,
        SplicingH5Dataset | Subset,
    ]:
        """Build train/val/test datasets from SpliceAI-format H5 files.

        ``data.train_h5_path`` is the canonical training source.

        Validation selection:
            - If ``data.val_h5_path`` is set, that H5 is loaded in full
              as the validation set and the entire ``train_h5_path`` is
              used for training (``train_split``/``val_split`` are
              ignored, with a warning if both are non-default).
            - Otherwise, ``train_h5_path`` is randomly split into
              train/val using ``data.train_split`` and ``data.val_split``.

        ``data.test_h5_path`` is loaded as the test set in full when
        provided (no further splitting).

        Returns:
            tuple[SplicingH5Dataset | Subset, ...]:
                ``(train_dataset, val_dataset, test_dataset)``.
        """
        self.logger.info("🔄 Starting dataset creation (h5)...")

        h5_path = self.cfg.data.train_h5_path
        cl = OmegaConf.select(self.cfg, "data.context_length", default=None)
        feature_h5_path = OmegaConf.select(
            self.cfg, "data.feature_h5_path", default=None
        )
        if feature_h5_path is not None:
            self.logger.info(f"🧬 Loading sidecar feature-H5: {feature_h5_path}")
        phylop_h5_path = OmegaConf.select(
            self.cfg, "data.phylop_h5_path", default=None
        )
        if phylop_h5_path is not None:
            self.logger.info(f"🧬 Loading sidecar phyloP target-H5: {phylop_h5_path}")
        elif bool(
            OmegaConf.select(self.cfg, "model.aux_heads.enable_phylop", default=False)
        ):
            # Without targets the head never receives a loss, so the run would
            # train as plain med_ei_ac while its config claims the phyloP task.
            raise ValueError(
                "model.aux_heads.enable_phylop=true requires data.phylop_h5_path."
            )
        # Multi-species Pangolin SSU: a list of per-species H5 files (spliced +
        # usage labels) trained together via PangolinSSUDataset. Falls back to
        # the single-file SplicingH5Dataset when data.train_h5_paths is unset.
        ms_paths = OmegaConf.select(self.cfg, "data.train_h5_paths", default=None)
        if ms_paths is not None:
            ms_paths = list(ms_paths)
            self.logger.info(
                f"📁 Loading {len(ms_paths)} multi-species SSU H5 files"
            )
            soft_usage = bool(
                OmegaConf.select(self.cfg, "data.soft_usage_labels", default=False)
            )
            if soft_usage:
                # The hard-label pass rebuilds `neither` as the complement of a
                # thresholded acceptor/donor, which rounds every usage weight
                # back to one-hot. That leaves a run that looks soft in the
                # config and trains hard, so refuse it rather than log it.
                if bool(
                    OmegaConf.select(
                        self.cfg,
                        "training.classification.use_hard_labels",
                        default=False,
                    )
                ):
                    raise ValueError(
                        "data.soft_usage_labels=true requires "
                        "training.classification.use_hard_labels=false; "
                        "the hard-label pass would round the usage weights "
                        "back to one-hot."
                    )
                self.logger.info("🎚️  3-class target is usage-weighted (soft)")
            dataset = PangolinSSUDataset(
                h5_paths=ms_paths,
                context_length=cl,
                soft_usage_labels=soft_usage,
            )
        else:
            self.logger.info(f"📁 Loading H5 dataset: {h5_path}")
            site_usage_soft = bool(
                OmegaConf.select(
                    self.cfg, "data.site_usage_soft_labels", default=False
                )
            )
            if site_usage_soft and bool(
                OmegaConf.select(
                    self.cfg,
                    "training.classification.use_hard_labels",
                    default=False,
                )
            ):
                raise ValueError(
                    "data.site_usage_soft_labels=true requires "
                    "training.classification.use_hard_labels=false; "
                    "the hard-label pass would discard the usage values."
                )
            dataset = SplicingH5Dataset(
                h5_path=h5_path,
                context_length=cl,
                feature_h5_path=feature_h5_path,
                site_usage_soft_labels=site_usage_soft,
                phylop_h5_path=phylop_h5_path,
            )

        trl = dataset.target_len
        effective_cl = cl if cl is not None else dataset.cl_max
        seq_len = trl + effective_cl
        self.logger.info(
            f"📐 H5 context: cl_max={dataset.cl_max}, "
            f"context_length={effective_cl}, target_len={trl}, "
            f"seq_length={seq_len}"
        )
        self.cfg.model.seq_length = seq_len
        self.cfg.model.tokenizer_max_length = seq_len

        explicit_val_dataset: SplicingH5Dataset | None = None
        val_h5_path = OmegaConf.select(self.cfg, "data.val_h5_path", default=None)
        if val_h5_path is not None:
            self.logger.info(f"📁 Loading explicit validation H5: {val_h5_path}")
            explicit_val_dataset = SplicingH5Dataset(
                h5_path=val_h5_path, context_length=cl
            )
            self.logger.info(
                f"🎯 Explicit validation set: {len(explicit_val_dataset):,} samples"
            )

        explicit_test_dataset: SplicingH5Dataset | None = None
        test_h5_path = OmegaConf.select(self.cfg, "data.test_h5_path", default=None)
        if test_h5_path is not None:
            self.logger.info(f"📁 Loading explicit test H5: {test_h5_path}")
            explicit_test_dataset = SplicingH5Dataset(
                h5_path=test_h5_path, context_length=cl
            )
            self.logger.info(
                f"🎯 Explicit test set: {len(explicit_test_dataset):,} samples"
            )

        dataset_size = len(dataset)
        self.logger.info(f"📈 Dataset loaded successfully: {dataset_size:,} samples")

        random_seed = self.cfg.data.random_seed
        self.logger.info(f"🎲 Setting random seed: {random_seed}")
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        train_dataset: SplicingH5Dataset | Subset
        val_dataset: SplicingH5Dataset | Subset
        if explicit_val_dataset is not None:
            train_split = float(
                OmegaConf.select(self.cfg, "data.train_split", default=1.0)
            )
            val_split = float(
                OmegaConf.select(self.cfg, "data.val_split", default=0.0)
            )
            if val_split > 0.0 and train_split < 1.0:
                self.logger.warning(
                    "⚠️  data.val_h5_path is set; ignoring data.train_split"
                    f"={train_split} and data.val_split={val_split}. "
                    "Full train_h5_path is used for training and val_h5_path "
                    "is used for validation."
                )
            train_dataset = dataset
            val_dataset = explicit_val_dataset
        else:
            indices = list(range(dataset_size))
            np.random.shuffle(indices)

            train_split = float(self.cfg.data.train_split)
            val_split = float(self.cfg.data.val_split)
            denom = max(train_split + val_split, 1e-8)
            train_size = int(train_split / denom * dataset_size)

            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)

        if explicit_test_dataset is not None:
            test_dataset: SplicingH5Dataset | Subset = explicit_test_dataset
        else:
            test_dataset = Subset(dataset, [])

        self.logger.info("✅ Datasets created successfully!")
        self.logger.info(f"  Train samples: {len(train_dataset):,}")
        self.logger.info(f"  Val samples:   {len(val_dataset):,}")
        self.logger.info(f"  Test samples:  {len(test_dataset):,}")

        return train_dataset, val_dataset, test_dataset

    def init_model(self, *, verbose: bool = True) -> IssaSplice:
        """Initialize the IssaSplice model.

        Args:
            verbose (bool): If ``False``, suppress all stdout/stderr,
                warnings, and non-ERROR log messages during model
                construction. Useful for non-master DDP processes.

        Returns:
            IssaSplice: Initialized model ready for training or inference.
        """
        model_cfg = self.cfg.model
        logging_cfg = self.cfg.logging

        model_name = self._get_model_name_from_config()

        enable_timing_log = logging_cfg.enable_timing_log
        timing_log_interval = logging_cfg.timing_log_interval

        training_cfg = self.cfg.training
        classification_enabled = OmegaConf.select(
            training_cfg, "classification.enabled", default=True
        )

        model_config_obj = OmegaConf.merge(
            OmegaConf.create(
                {
                    "model_name": model_name,
                    "enable_timing_log": enable_timing_log,
                    "timing_log_interval": timing_log_interval,
                    "classification_enabled": classification_enabled,
                }
            ),
            model_cfg,
        )

        with self._quiet_context(verbose):
            model = IssaSplice(model_config_obj)

        if model_cfg.freeze_backbone and verbose:
            model.print_freeze_status()

        if enable_timing_log:
            model.enable_timing_logging(True, timing_log_interval)

        return model

    @staticmethod
    @contextlib.contextmanager
    def _quiet_context(verbose: bool):
        """Optionally suppress stdout/stderr/warnings/logs.

        When *verbose* is ``True`` the block runs normally.
        When ``False``, stdout, stderr, warnings, and all
        log messages below ERROR are silenced for the
        duration of the block.

        Args:
            verbose (bool): Pass through without suppression when ``True``.

        Yields:
            None
        """
        if verbose:
            yield
            return

        root = logging.getLogger()
        orig_level = root.level
        root.setLevel(logging.ERROR)
        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    yield
        finally:
            root.setLevel(orig_level)


def splicing_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Handle variable-length sequences in splicing data with custom collate function.

    Args:
        batch: List of samples from the dataset

    Returns:
        Dictionary with batched data
    """
    refs = []
    alts = []
    afs = []
    labels_list = []

    for sample in batch:
        refs.append(sample["ref"])
        alts.append(sample["alt"])
        afs.append(sample["af"])
        labels_list.append(sample["labels"])

    af_batch = torch.stack(afs)

    max_len = max(len(ref) for ref in refs)

    padded_labels = []
    for labels in labels_list:
        seq_len = labels.shape[1]
        if seq_len < max_len:
            pad_width = max_len - seq_len
            padded = torch.nn.functional.pad(labels, (0, pad_width), value=0)
        else:
            padded = labels
        padded_labels.append(padded)

    labels_batch = torch.stack(padded_labels)

    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, ref in enumerate(refs):
        attention_mask[i, : len(ref)] = True

    return {
        "ref": refs,
        "alt": alts,
        "af": af_batch,
        "labels": labels_batch,
        "attention_mask": attention_mask,
        "lengths": torch.tensor([len(ref) for ref in refs]),
    }
