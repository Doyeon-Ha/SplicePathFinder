"""Loss functions for training splicing models"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithClassBalance(nn.Module):
    """Compute focal loss with class balance and label smoothing.

    Note:
        This variant expects hard class indices per sample and applies
        label smoothing before computing a BCE-style focal term.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss value.

        Args:
            inputs (torch.Tensor): Logits of shape [N, C].
            targets (torch.Tensor): Class indices of shape [N].

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        # Apply label smoothing for better generalization
        num_classes = inputs.size(-1)
        smooth_targets = torch.zeros_like(inputs)
        smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0)
        smooth_targets = (
            smooth_targets * (1.0 - self.label_smoothing)
            + self.label_smoothing / num_classes
        )

        # Calculate focal loss
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, smooth_targets, reduction="none"
        )
        p_t = torch.exp(-ce_loss)
        focal_loss = (1.0 - p_t) ** self.gamma * ce_loss

        # Apply class weights
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t.unsqueeze(-1) * focal_loss

        return focal_loss.mean()


class MultiLabelFocalLoss(nn.Module):
    """Compute focal loss for multi-label classification.

    Args:
        alpha (torch.Tensor | None): Per-class weights tensor of shape [C].
        gamma (float | list[float] | tuple[float, ...] | torch.Tensor):
            Focusing parameter for (1 - p_t)^gamma. Can be a single scalar for
            all classes or a per-class list/tensor of shape [C].
        use_negative_sampling (bool): Whether to use negative sampling for no loss
         calculation for 95% neither class samples. Default is False.
        negative_sample_rate (float): The rate of negative samples to keep. Default is 0.05.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float | list[float] | tuple[float, ...] | torch.Tensor = 2.0,
        use_negative_sampling: bool = False,
        negative_sample_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        # Store gamma as either scalar float or 1D tensor; move to device at forward
        if isinstance(gamma, torch.Tensor):
            if gamma.dim() == 0:
                self._gamma_scalar = float(gamma.item())
                self._gamma_tensor = None
            else:
                self._gamma_scalar = None
                self._gamma_tensor = gamma.detach().float().view(-1)
        elif isinstance(gamma, (list, tuple)):
            self._gamma_scalar = None
            self._gamma_tensor = torch.tensor(list(gamma), dtype=torch.float32).view(-1)
        else:
            self._gamma_scalar = float(gamma)
            self._gamma_tensor = None

        self.use_negative_sampling = use_negative_sampling
        self.negative_sample_rate = negative_sample_rate

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss value.

        Args:
            inputs (torch.Tensor): Logits of shape [N, C].
            targets (torch.Tensor): Multi-hot targets of shape [N, C].

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        # Binary focal loss for each class
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="none"
        )

        # Focal term with optional per-class gamma
        pt = torch.exp(-bce_loss)
        one_minus_pt = 1.0 - pt

        if self._gamma_tensor is not None:
            # Move gamma to the same device/dtype and validate shape
            gamma_vec = self._gamma_tensor.to(
                device=inputs.device, dtype=bce_loss.dtype
            )
            num_classes = inputs.size(-1)
            if gamma_vec.numel() not in (1, num_classes):
                raise ValueError(
                    (
                        "Gamma length "
                        f"{gamma_vec.numel()} does not match number of classes "
                        f"{num_classes}"
                    )
                )
            gamma_expanded = gamma_vec.view(1, -1)
            focal_factor = torch.pow(one_minus_pt, gamma_expanded)
        else:
            focal_factor = torch.pow(one_minus_pt, self._gamma_scalar)

        focal_loss = focal_factor * bce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.unsqueeze(0).expand_as(targets)
            focal_loss = alpha_t * focal_loss

        # ==================================================================
        # Negative Subsampling Logic (Neither {negative_sample_rate*100}% 버리기)
        # ==================================================================
        if self.use_negative_sampling:
            # Positive (Correct) is always kept
            pos_mask = targets == 1

            # Randomly keep only {negative_sample_rate*100}% of Negative/Neither
            rand_mask = torch.rand_like(inputs) < self.negative_sample_rate
            neg_mask = (targets == 0) & rand_mask

            # Final mask (Positive + selected Negative)
            valid_mask = pos_mask | neg_mask

            # Apply mask
            focal_loss = focal_loss * valid_mask.float()

            # --------------------------------------------------------------
            # Dynamic Weights (Inverse Frequency in Local Batch)
            # --------------------------------------------------------------
            # Calculate class frequency in sequence targets [B, N, C]
            pos_count = targets.sum(dim=1, keepdim=True).float()  # [B, 1, C]

            total_count = targets.shape[1]  # Sequence length, N

            # Inverse frequency weight
            dynamic_weights = total_count / (pos_count + 1.0)  # [B, 1, C]
            dynamic_weights = torch.log(1.0 + dynamic_weights)

            # Normalize
            dynamic_weights = (dynamic_weights / dynamic_weights.mean()).detach()

            weights_map = torch.where(
                targets == 1, dynamic_weights, torch.ones_like(dynamic_weights)
            )

            focal_loss = focal_loss * weights_map

            num_valid_pixels = valid_mask.sum() + 1e-8
            focal_loss = focal_loss.sum() / num_valid_pixels
        else:
            focal_loss = focal_loss.mean()

        return focal_loss


class MultiClassFocalLoss(nn.Module):
    """Multiclass focal loss supporting hard and soft targets.

    Args:
        alpha (torch.Tensor | None): Per-class weights tensor of shape [C]. If None,
            no class weighting is applied.
        gamma (float): Focusing parameter for modulating factor (1 - p_t)^gamma.
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        # Keep alpha as attribute (already on correct device in caller); may be None
        self.alpha = alpha
        self.gamma = float(gamma)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute multiclass focal loss.

        Args:
            inputs (torch.Tensor): Logits of shape [N, C].
            targets (torch.Tensor): Either class indices of shape [N] (long/int)
                or one-hot/soft labels of shape [N, C] (float).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Log-softmax and probabilities
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = log_probs.exp()

        if targets.dim() == 1 or (targets.dim() == 2 and targets.size(-1) == 1):
            # Hard labels (class indices)
            target_indices = targets.view(-1).long()
            ce_loss_per_sample = F.nll_loss(log_probs, target_indices, reduction="none")
            # Probability of the true class
            p_t = probs.gather(dim=-1, index=target_indices.unsqueeze(-1)).squeeze(-1)

            # Optional class weighting (alpha)
            if self.alpha is not None:
                alpha_factor = self.alpha[target_indices]
                ce_loss_per_sample = ce_loss_per_sample * alpha_factor
        else:
            # Soft or one-hot labels
            targets_float = targets.float()
            # Cross-entropy with soft targets: -sum(y * log p)
            ce_loss_per_sample = -(targets_float * log_probs).sum(dim=-1)
            # p_t as expectation under target distribution: sum(y * p)
            p_t = (targets_float * probs).sum(dim=-1)

            if self.alpha is not None:
                # Expected alpha weight under target distribution
                alpha_factor = (targets_float * self.alpha).sum(dim=-1)
                ce_loss_per_sample = ce_loss_per_sample * alpha_factor

        # Focal modulation
        focal_factor = (1.0 - p_t).clamp(min=1e-6, max=1.0) ** self.gamma
        loss = focal_factor * ce_loss_per_sample
        return loss.mean()


class WeightedBCELoss(nn.Module):
    """Compute weighted BCE loss for multi-label classification."""

    def __init__(self, pos_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        # pos_weights: weight for positive examples per class
        self.pos_weights = pos_weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss value.

        Args:
            inputs (torch.Tensor): Logits of shape [N, C].
            targets (torch.Tensor): Multi-hot targets of shape [N, C].

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        # Ensure targets are float type for binary_cross_entropy_with_logits
        targets = targets.float()

        if self.pos_weights is not None:
            pos_weight = self.pos_weights.to(inputs.device)
        else:
            # Calculate pos_weight based on class imbalance
            pos_count = targets.sum(dim=0)
            neg_count = targets.shape[0] - pos_count
            pos_weight = neg_count / (pos_count + 1e-8)

        loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=pos_weight
        )
        return loss


class WeightedMSELoss(nn.Module):
    """Compute weighted mean squared error for regression targets.

    Args:
        pos_weight (float): Weight multiplier applied where target != 0.
            Use >1.0 to emphasize non-zero targets. Default is 10.0.

    Note:
        This implements element-wise weights w = pos_weight if target != 0 else 1,
        then computes mean((w * (pred - target)^2)). Targets are treated as float.
    """

    def __init__(self, pos_weight: float = 10.0) -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted mean squared error.

        Args:
            inputs (torch.Tensor): Predictions of any shape.
            targets (torch.Tensor): Targets of the same shape as inputs.

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        targets = targets.float()
        weights = torch.where(targets != 0, self.pos_weight, 1.0)
        loss = (weights * (inputs - targets) ** 2).mean()
        return loss


class WeightedCELoss(nn.Module):
    """Compute weighted cross entropy loss for multi-class classification.
    This is a softmax-based loss function for mutually exclusive classes.
    Args:
        class_weights (torch.Tensor | None): Per-class weights tensor of shape [C].
            If None, no class weighting is applied.
        label_smoothing (float): Label smoothing factor. Default is 0.0 (no smoothing).
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self.label_smoothing = float(label_smoothing)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross entropy loss.
        Args:
            inputs (torch.Tensor): Logits of shape [N, C].
            targets (torch.Tensor): Either class indices of shape [N] (long/int)
                or one-hot/soft labels of shape [N, C] (float).
        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        device = inputs.device
        if self.class_weights is not None:
            weight = self.class_weights.to(device)
        else:
            weight = None

        if targets.dim() == 1 or (targets.dim() == 2 and targets.size(-1) == 1):
            # Hard labels (class indices)
            target_indices = targets.view(-1).long()
            loss = F.cross_entropy(
                inputs,
                target_indices,
                weight=weight,
                label_smoothing=self.label_smoothing,
            )
        else:
            # Soft or one-hot labels: manual weighted cross entropy
            targets_float = targets.float()
            log_probs = F.log_softmax(inputs, dim=-1)
            # Cross-entropy with soft targets: -sum(y * log p)
            ce_loss_per_sample = -(targets_float * log_probs).sum(dim=-1)

            if weight is not None:
                # Expected weight under target distribution
                weight_factor = (targets_float * weight).sum(dim=-1)
                ce_loss_per_sample = ce_loss_per_sample * weight_factor

            loss = ce_loss_per_sample.mean()

        return loss
