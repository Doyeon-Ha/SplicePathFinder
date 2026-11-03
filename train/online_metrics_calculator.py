"""
Online metrics calculator class for multi-label classification.

This module supports two accumulation strategies for AUC/threshold tuning:

- Exact sampling (stores per-position labels/probs in lists; memory-heavy)
- Histogram accumulation (stores per-class positive/negative histograms over
  probability bins; memory-light and friendly for distributed reduction)
"""

import logging

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


class OnlineMetricsCalculator:
    """Online metrics calculator class for multi-label classification."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        num_classes: int = 3,
        histogram_bins: int = 256,
        use_histogram: bool = True,
    ) -> None:
        """
        Initialize online metrics calculator.

        Args:
            logger (logging.Logger | None): Logger for status messages.
                Defaults to module logger when None.
            num_classes (int): Number of classes.
            histogram_bins (int): Number of probability bins for histogram mode.
            use_histogram (bool): Whether to store AUC data as histograms instead
                of raw lists to reduce memory and enable distributed reduction.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.num_classes = int(num_classes)
        self.histogram_bins = int(histogram_bins)
        self.use_histogram = bool(use_histogram)
        # Bin edges over [0, 1]
        self._bin_edges: np.ndarray = np.linspace(
            0.0, 1.0, self.histogram_bins + 1, dtype=np.float64
        )
        self.reset()

    def reset(self) -> None:
        """Reset all metrics and storage."""
        # Multi-label specific metrics
        self.true_positives = np.zeros(self.num_classes, dtype=np.int64)
        self.false_positives = np.zeros(self.num_classes, dtype=np.int64)
        self.false_negatives = np.zeros(self.num_classes, dtype=np.int64)
        self.true_negatives = np.zeros(self.num_classes, dtype=np.int64)
        self.exact_matches = 0
        self.total_label_errors = 0

        # Track total samples (batches * seq_len) and total elements (batches * seq_len * channels)
        self.total_samples = 0
        self.total_predictions_count = 0

        # Common metrics storage for PR/AUC or thresholds
        if self.use_histogram:
            # Histogram counts: per-class positive/negative probability bins
            self.hist_pos = np.zeros(
                (self.num_classes, self.histogram_bins), dtype=np.int64
            )
            self.hist_neg = np.zeros(
                (self.num_classes, self.histogram_bins), dtype=np.int64
            )
            # Clear any residual list storage for safety
            self.auc_labels = {i: [] for i in range(self.num_classes)}
            self.auc_probs = {i: [] for i in range(self.num_classes)}
        else:
            # Exact storage (memory-heavy)
            self.auc_labels = {i: [] for i in range(self.num_classes)}
            self.auc_probs = {i: [] for i in range(self.num_classes)}

        return

    def update(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Update metrics with batch predictions.
        """
        self._update_multi_label(preds, labels)

        return

    def update_auc_data(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
    ) -> None:
        """
        Update only AUC data for dynamic threshold calculation.
        - use_histogram=True, update histogram data
        - use_histogram=False, update exact data

        Args:
            labels: Ground truth multi-hot labels of shape
                [batch_size, seq_len, num_classes] or [batch_size, seq_len, 2]
                (when only acceptor/donor classes are provided)
            probs: Class probabilities of shape [batch_size, seq_len, num_classes]
            or [batch_size, seq_len, 2]
        """
        # Determine actual number of classes from input shape
        actual_num_classes = labels.shape[-1]

        # Flatten for easier computation: [B*L, C]
        labels_flat = labels.reshape(-1, actual_num_classes)
        probs_flat = probs.reshape(-1, actual_num_classes)

        if self.use_histogram:
            for class_idx in range(actual_num_classes):
                # Support soft labels by thresholding at 0.5
                binary_labels = labels_flat[:, class_idx] >= 0.5
                class_probs = probs_flat[:, class_idx]

                # Positive class histogram
                if np.any(binary_labels):
                    pos_counts, _ = np.histogram(
                        class_probs[binary_labels], bins=self._bin_edges
                    )
                    self.hist_pos[class_idx] += pos_counts.astype(np.int64)

                # Negative class histogram
                if np.any(~binary_labels):
                    neg_counts, _ = np.histogram(
                        class_probs[~binary_labels], bins=self._bin_edges
                    )
                    self.hist_neg[class_idx] += neg_counts.astype(np.int64)
        else:
            # Exact storage (fallback)
            for class_idx in range(actual_num_classes):
                binary_labels = labels_flat[:, class_idx]
                class_probs = probs_flat[:, class_idx]
                self.auc_labels[class_idx].extend(binary_labels.tolist())
                self.auc_probs[class_idx].extend(class_probs.tolist())

        return

    def _update_multi_label(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Update metrics for multi-label classification.

        Args:
        preds: Binary predictions of shape [batch_size, seq_len, num_classes]
            or [batch_size, seq_len, 2] (when only acceptor/donor classes are provided)
        labels: Ground truth multi-hot labels of shape [batch_size, seq_len,
            num_classes] or [batch_size, seq_len, 2]
        """

        # Ensure preds and labels have the same shape
        assert preds.shape == labels.shape, (
            f"Shape mismatch: preds {preds.shape} vs labels {labels.shape}"
        )
        # Determine actual number of classes from input shape
        actual_num_classes = labels.shape[-1]

        # Flatten for easier computation
        labels_flat = labels.reshape(-1, actual_num_classes)  # [B*seq_len, num_classes]
        preds_flat = preds.reshape(-1, actual_num_classes)  # [B*seq_len, num_classes]

        # Update per-class statistics
        for class_idx in range(actual_num_classes):
            true_mask = labels_flat[:, class_idx] == 1
            pred_mask = preds_flat[:, class_idx] == 1

            self.true_positives[class_idx] += np.sum(true_mask & pred_mask)
            self.false_positives[class_idx] += np.sum(pred_mask & ~true_mask)
            self.false_negatives[class_idx] += np.sum(true_mask & ~pred_mask)
            self.true_negatives[class_idx] += np.sum(~true_mask & ~pred_mask)

        # Update exact match accuracy
        exact_matches = np.all(preds_flat == labels_flat, axis=1)
        self.exact_matches += np.sum(exact_matches)

        # Update hamming loss components
        label_errors = np.sum(preds_flat != labels_flat)
        self.total_label_errors += label_errors

        self.total_samples += len(labels_flat)
        # Add actual number of elements processed to handle subset updates correctly (e.g. only 2 channels)
        self.total_predictions_count += labels_flat.size

        return

    def compute_metrics(self) -> dict[str, float]:
        """
        Compute final metrics from accumulated statistics.
        """
        return self._compute_multi_label_metrics()

    def _compute_multi_label_metrics(self) -> dict[str, float]:
        """Compute metrics for multi-label classification."""
        metrics = {}

        if self.total_samples == 0:
            # Return zero metrics
            metrics["exact_match_accuracy"] = 0.0
            metrics["hamming_loss"] = 0.0
            metrics["macro_precision"] = 0.0
            metrics["macro_recall"] = 0.0

            for i, class_name in enumerate(["acceptor", "donor"]):
                metrics[f"precision_{class_name}"] = 0.0
                metrics[f"recall_{class_name}"] = 0.0
                metrics[f"auc_pr_{class_name}"] = 0.0
            return metrics

        # Exact match accuracy
        metrics["exact_match_accuracy"] = self.exact_matches / self.total_samples

        # Hamming loss: Use total_predictions_count for correct denominator when using subsets of classes
        if self.total_predictions_count > 0:
            metrics["hamming_loss"] = (
                self.total_label_errors / self.total_predictions_count
            )
        else:
            metrics["hamming_loss"] = 0.0

        # Per-class metrics
        class_names = ["acceptor", "donor"]
        precisions = []
        recalls = []

        for i, class_name in enumerate(class_names):
            tp = self.true_positives[i]
            fp = self.false_positives[i]
            fn = self.false_negatives[i]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            metrics[f"precision_{class_name}"] = precision
            metrics[f"recall_{class_name}"] = recall

            tn = self.true_negatives[i]
            mcc = self._compute_mcc(tp, tn, fp, fn)
            metrics[f"mcc_{class_name}"] = mcc

            if (tp + fp + fn + tn) > 0:
                precisions.append(precision)
                recalls.append(recall)

            # AUC-PR
            if self.use_histogram:
                metrics[f"auc_pr_{class_name}"] = self._compute_auc_pr_histogram(i)
            else:
                metrics[f"auc_pr_{class_name}"] = self._compute_auc_pr_exact(i)

            # AUC-ROC (exact storage only; histogram path skipped — lossy)
            if not self.use_histogram:
                metrics[f"auc_roc_{class_name}"] = self._compute_auc_roc_exact(i)

        # Macro metrics
        if precisions:
            metrics["macro_precision"] = np.mean(precisions)
            metrics["macro_recall"] = np.mean(recalls)
        else:
            metrics["macro_precision"] = 0.0
            metrics["macro_recall"] = 0.0

        return metrics

    @staticmethod
    def _compute_mcc(
        tp: int,
        tn: int,
        fp: int,
        fn: int,
    ) -> float:
        """Compute Matthews Correlation Coefficient.

        Args:
            tp (int): True positives.
            tn (int): True negatives.
            fp (int): False positives.
            fn (int): False negatives.

        Returns:
            float: MCC in [-1, 1], or 0.0 when the
                denominator is zero.
        """
        numer = float(tp * tn - fp * fn)
        denom_sq = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
        if denom_sq <= 0.0:
            return 0.0
        return numer / (denom_sq**0.5)

    def _compute_auc_pr_histogram(self, class_idx: int) -> float:
        """Compute AUC-PR for a given class using histogram storage (Reverse CumSum)."""

        pos_counts = self.hist_pos[class_idx].astype(np.float64)
        neg_counts = self.hist_neg[class_idx].astype(np.float64)

        total_pos = pos_counts.sum()
        if total_pos == 0:
            return 0.0

        # Reverse: High probability -> Low probability
        # -> Threshold from 1.0 to 0.0
        pos_rev = pos_counts[::-1]
        neg_rev = neg_counts[::-1]

        # Cumulative Sum = TP, FP at each threshold
        tp = np.cumsum(pos_rev)
        fp = np.cumsum(neg_rev)

        # Precision / Recall
        #    Precision = TP / (TP + FP)
        #    Recall = TP / Total_Pos
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = tp / total_pos

        # Add start point (P=1, R=0) to close the curve nicely
        precision = np.concatenate(([1.0], precision))
        recall = np.concatenate(([0.0], recall))

        return auc(recall, precision)

    def _compute_auc_pr_exact(self, idx: int) -> float:
        """Compute AUC-PR for a given class using exact storage."""
        labels = self.auc_labels[idx]
        probs = self.auc_probs[idx]
        if len(labels) < 10 or len(np.unique(labels)) < 2:
            return 0.0
        try:
            precision, recall, _ = precision_recall_curve(labels, probs)
            return float(auc(recall, precision))

        except (ValueError, Exception) as e:
            self.logger.debug(f"AUC-PR calculation failed for class {idx}: {e}")

            return 0.0

    def _compute_auc_roc_exact(self, idx: int) -> float:
        """Compute AUC-ROC for a given class using exact storage.

        Args:
            idx (int): Class index.

        Returns:
            float: ROC AUC in [0, 1], or 0.0 when insufficient data
                (fewer than 10 samples or only one label class present).
        """
        labels = self.auc_labels[idx]
        probs = self.auc_probs[idx]
        if len(labels) < 10 or len(np.unique(labels)) < 2:
            return 0.0
        try:
            return float(roc_auc_score(labels, probs))
        except ValueError as e:
            self.logger.debug(f"AUC-ROC calculation failed for class {idx}: {e}")
            return 0.0

    def compute_optimal_thresholds(
        self,
        objective: str = "f1",
        target_precision: list[float] | None = None,
        target_recall: list[float] | None = None,
    ) -> list[float]:
        """
        Compute per-class optimal thresholds from stored PR data.

        Args:
            objective: One of {"f1", "target_precision", "target_recall"}
            target_precision: list of desired minimum precision per class
                (used when objective=="target_precision")
            target_recall: list of desired minimum recall per class
                (used when objective=="target_recall")

        Returns:
            list[float]: thresholds per class (length=num_classes)

        Raises:
            ValueError: If target lists don't match num_classes
        """
        # Validate input lists
        if objective == "target_precision" and target_precision is not None:
            if len(target_precision) != self.num_classes:
                raise ValueError(
                    f"target_precision list length ({len(target_precision)}) "
                    f"must match num_classes ({self.num_classes})"
                )

        if objective == "target_recall" and target_recall is not None:
            if len(target_recall) != self.num_classes:
                raise ValueError(
                    f"target_recall list length ({len(target_recall)}) "
                    f"must match num_classes ({self.num_classes})"
                )

        thresholds: list[float] = [0.5 for _ in range(self.num_classes)]

        if (
            self.use_histogram
            and hasattr(self, "hist_pos")
            and hasattr(self, "hist_neg")
        ):
            # Compute from histograms
            for class_idx in range(self.num_classes):
                pos_counts = self.hist_pos[class_idx].astype(np.float64)
                neg_counts = self.hist_neg[class_idx].astype(np.float64)

                total_pos = float(pos_counts.sum())

                if total_pos < 1.0:
                    continue

                # Cumulative sums from high prob -> low prob (descending thresholds)
                pos_cum = np.cumsum(pos_counts[::-1])
                neg_cum = np.cumsum(neg_counts[::-1])

                # Precision/Recall per threshold bin edge (exclude last bin edge)
                tp = pos_cum
                fp = neg_cum
                precision = np.divide(
                    tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0
                )
                recall = tp / total_pos

                best_idx: int | None = None
                if objective == "target_precision" and target_precision is not None:
                    target_p = float(target_precision[class_idx])
                    mask = precision >= target_p
                    idxs = np.where(mask)[0]
                    if idxs.size > 0:
                        best_idx = int(idxs[np.argmax(recall[idxs])])
                elif objective == "target_recall" and target_recall is not None:
                    target_r = float(target_recall[class_idx])
                    mask = recall >= target_r
                    idxs = np.where(mask)[0]
                    if idxs.size > 0:
                        best_idx = int(idxs[np.argmax(precision[idxs])])
                else:
                    # Optimize F1
                    f1_num = 2.0 * precision * recall
                    f1_den = precision + recall
                    f1 = np.divide(
                        f1_num, f1_den, out=np.zeros_like(f1_num), where=f1_den > 0
                    )
                    best_idx = int(np.argmax(f1))

                # Map best_idx back to a numeric threshold in [0,1]
                # best_idx corresponds to bin starting at edge index:
                # self._bin_edges[-(best_idx+2)]
                # Use mid-bin as representative threshold
                left_edge = self._bin_edges[-(best_idx + 2)]
                right_edge = self._bin_edges[-(best_idx + 1)]
                th_value = float((left_edge + right_edge) / 2.0)
                thresholds[class_idx] = th_value
                self.logger.info(
                    "Using histogram-based threshold for class %d: threshold=%.6f",
                    class_idx,
                    th_value,
                )
            return thresholds

        # Fallback to exact PR curve when histograms are not used
        for class_idx in range(self.num_classes):
            labels_list = self.auc_labels[class_idx]
            probs_list = self.auc_probs[class_idx]

            if len(labels_list) < 10 or len(set(labels_list)) < 2:
                continue

            labels_arr = np.asarray(labels_list)
            probs_arr = np.asarray(probs_list)

            precision_curve, recall_curve, thresh = precision_recall_curve(
                labels_arr, probs_arr
            )
            if len(thresh) == 0:
                continue

            p = precision_curve[1:]
            r = recall_curve[1:]
            t = thresh

            best_local = None
            if objective == "target_precision" and target_precision is not None:
                class_target_precision = float(target_precision[class_idx])
                mask = p >= class_target_precision
                idxs = np.where(mask)[0]
                if len(idxs) > 0:
                    best_local = int(idxs[np.argmax(r[idxs])])
            elif objective == "target_recall" and target_recall is not None:
                class_target_recall = float(target_recall[class_idx])
                mask = r >= class_target_recall
                idxs = np.where(mask)[0]
                if len(idxs) > 0:
                    best_local = int(idxs[np.argmax(p[idxs])])
            else:
                f1_numerator = 2.0 * p * r
                f1_denominator = p + r
                f1 = np.divide(
                    f1_numerator,
                    f1_denominator,
                    out=np.zeros_like(f1_numerator),
                    where=f1_denominator != 0,
                )
                best_local = int(np.argmax(f1))

            if best_local is not None:
                thresholds[class_idx] = float(t[best_local])

        return thresholds

    def compute_metrics_at_thresholds(
        self, thresholds: list[float]
    ) -> dict[str, float]:
        """
        Compute per-class precision/recall using given thresholds on stored PR data.

        Note: Uses the sampled AUC store, not the full dataset. Intended for
        validation-time model selection/logging.
        """
        metrics: dict[str, float] = {}

        class_names = ["acceptor", "donor"]

        all_precisions: list[float] = []
        all_recalls: list[float] = []

        if (
            self.use_histogram
            and hasattr(self, "hist_pos")
            and hasattr(self, "hist_neg")
        ):
            # Approximate metrics using histograms
            for class_idx, class_name in enumerate(class_names):
                pos_counts = self.hist_pos[class_idx].astype(np.float64)
                neg_counts = self.hist_neg[class_idx].astype(np.float64)
                total_pos = float(pos_counts.sum())

                th = float(thresholds[class_idx])
                # Identify bin index for threshold
                bin_idx = int(np.searchsorted(self._bin_edges, th, side="right") - 1)
                bin_idx = np.clip(bin_idx, 0, self.histogram_bins - 1)

                # Sum counts for probs >= th
                tp = pos_counts[bin_idx:].sum()
                fp = neg_counts[bin_idx:].sum()
                fn = total_pos - tp

                precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                recall = float(tp / total_pos) if total_pos > 0 else 0.0

                metrics[f"precision_{class_name}_tuned"] = precision
                metrics[f"recall_{class_name}_tuned"] = recall
                metrics[f"best_threshold_{class_name}"] = th

                all_precisions.append(precision)
                all_recalls.append(recall)
        else:
            for class_idx, class_name in enumerate(class_names):
                labels_list = self.auc_labels[class_idx]
                probs_list = self.auc_probs[class_idx]

                if len(labels_list) == 0:
                    metrics[f"precision_{class_name}_tuned"] = 0.0
                    metrics[f"recall_{class_name}_tuned"] = 0.0
                    metrics[f"best_threshold_{class_name}"] = thresholds[class_idx]
                    continue

                labels_arr = np.asarray(labels_list).astype(int)
                probs_arr = np.asarray(probs_list)

                th = float(thresholds[class_idx])
                preds_arr = (probs_arr >= th).astype(int)

                tp = int(np.sum((labels_arr == 1) & (preds_arr == 1)))
                fp = int(np.sum((labels_arr == 0) & (preds_arr == 1)))
                fn = int(np.sum((labels_arr == 1) & (preds_arr == 0)))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                metrics[f"precision_{class_name}_tuned"] = precision
                metrics[f"recall_{class_name}_tuned"] = recall
                metrics[f"best_threshold_{class_name}"] = th

                all_precisions.append(precision)
                all_recalls.append(recall)

        # Macro averages for tuned metrics
        if len(all_precisions) > 0:
            metrics["macro_precision_tuned"] = float(np.mean(all_precisions))
            metrics["macro_recall_tuned"] = float(np.mean(all_recalls))
        else:
            metrics["macro_precision_tuned"] = 0.0
            metrics["macro_recall_tuned"] = 0.0

        return metrics

    # === Distributed helpers ===
    def distributed_reduce(self) -> None:
        """
        Reduce internal accumulators across distributed processes.

        This performs an in-place SUM reduction of histogram counts and
        scalar/array counters. No-op when torch.distributed is not initialized.
        """
        if not dist.is_initialized():
            return

        # Select device for collective ops
        try:
            backend = str(dist.get_backend()).lower()
        except Exception:
            backend = ""
        use_cuda = backend == "nccl" and torch.cuda.is_available()
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if use_cuda
            else torch.device("cpu")
        )

        # Reduce histogram counts when enabled
        if (
            self.use_histogram
            and hasattr(self, "hist_pos")
            and hasattr(self, "hist_neg")
        ):
            for arr in (self.hist_pos, self.hist_neg):
                t = torch.from_numpy(arr).to(device=device, dtype=torch.long)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                arr[:] = t.cpu().numpy()

        # Reduce per-class counters
        for arr in (
            self.true_positives,
            self.false_positives,
            self.false_negatives,
            self.true_negatives,
        ):
            t = torch.from_numpy(arr).to(device=device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            arr[:] = t.cpu().numpy()

        # Reduce scalar counters
        # Added total_predictions_count to reduction list
        scalars = np.asarray(
            [
                self.exact_matches,
                self.total_label_errors,
                self.total_samples,
                self.total_predictions_count,
            ],
            dtype=np.int64,
        )
        t_scalars = torch.from_numpy(scalars).to(device=device, dtype=torch.long)
        dist.all_reduce(t_scalars, op=dist.ReduceOp.SUM)
        reduced = t_scalars.cpu().numpy()
        self.exact_matches = int(reduced[0])
        self.total_label_errors = int(reduced[1])
        self.total_samples = int(reduced[2])
        self.total_predictions_count = int(reduced[3])
