"""학습형 SAV 분류기: RBF 확장 + 단방향 Mamba2 + mean pool + MLP.

frozen base model이 산출한 변이 중심 윈도의 per-nt acceptor/donor/neither 확률
(ref/alt 6채널)을 입력으로, 변이가 splice-altering(SAV)인지 이진 분류한다.

Mamba2는 CUDA 전용 커널이라 ``_Mamba2Block`` 내부에서 lazy import한다 —
RBF/pooling 등 나머지 모듈은 CPU에서 단위 테스트 가능하다.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

_DEFAULT_NUM_CHANNELS: int = 6


class RbfExpansion(nn.Module):
    """각 스칼라 채널을 ``num_centers``개의 RBF 특징으로 확장한다.

    동일한 위치 불변 kernel을 모든 위치·모든 채널에 적용한다. center는
    ``linspace(0, 1, num_centers)``로 초기화하며, ``trainable``이면 학습한다.
    """

    def __init__(
        self,
        num_centers: int,
        gamma: float | None = None,
        trainable: bool = False,
        trainable_gamma: bool = False,
    ) -> None:
        """RBF 확장 레이어를 초기화한다.

        Args:
            num_centers (int): RBF center 개수 K.
            gamma (float | None): bandwidth. None이면 ``1/(2*sigma^2)``,
                ``sigma = 1/(num_centers-1)``.
            trainable (bool): center를 학습 파라미터로 둘지 여부.
            trainable_gamma (bool): gamma(log 공간)를 학습할지 여부.
        """
        super().__init__()
        if num_centers < 1:
            raise ValueError("num_centers는 1 이상이어야 한다")
        self.num_centers = num_centers
        centers = torch.linspace(0.0, 1.0, num_centers)
        sigma = 1.0 / (num_centers - 1) if num_centers > 1 else 1.0
        gamma_val = gamma if gamma is not None else 1.0 / (2.0 * sigma * sigma)
        log_gamma = torch.log(torch.tensor(float(gamma_val)))
        if trainable:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)
        if trainable_gamma:
            self.log_gamma = nn.Parameter(log_gamma)
        else:
            self.register_buffer("log_gamma", log_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """채널을 RBF로 확장한다.

        Args:
            x (torch.Tensor): shape ``(batch_size, seq_len, num_channels)``,
                값은 ``[0, 1]``.

        Returns:
            torch.Tensor: shape
                ``(batch_size, seq_len, num_channels * num_centers)``.
        """
        gamma = torch.exp(self.log_gamma)
        diff = x.unsqueeze(-1) - self.centers  # (B, W, C, K)
        phi = torch.exp(-gamma * diff * diff)
        return phi.flatten(start_dim=-2)  # (B, W, C*K)


class _Mamba2Block(nn.Module):
    """Pre-norm residual Mamba2 블록 (``x = x + Mamba2(RMSNorm(x))``)."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        headdim: int,
        bidirectional: bool = False,
    ) -> None:
        """Mamba2 블록을 초기화한다.

        Args:
            d_model (int): hidden 차원.
            d_state (int): SSM state 차원.
            d_conv (int): depthwise conv 폭.
            expand (int): inner 확장 배수.
            headdim (int): head 차원 (``expand*d_model``이 나뉘어야 함).
            bidirectional (bool): True면 forward + reversed 스캔을 합산
                (Caduceus/Vim식 bi-Mamba). 각 방향 독립 Mamba2.
        """
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError as exc:
            raise ImportError(
                "Mamba2 requires the 'mamba_ssm' package (and causal-conv1d). "
                "Install with: uv pip install mamba-ssm causal-conv1d"
            ) from exc
        self.norm = nn.RMSNorm(d_model)
        self.bidirectional = bidirectional
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
        self.mamba_rev = (
            Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
            )
            if bidirectional
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """블록을 통과시킨다.

        Args:
            x (torch.Tensor): shape ``(batch_size, seq_len, d_model)``.

        Returns:
            torch.Tensor: 같은 shape.
        """
        normed = self.norm(x)
        out = self.mamba(normed)
        if self.mamba_rev is not None:
            rev = torch.flip(self.mamba_rev(torch.flip(normed, dims=[1])), dims=[1])
            out = out + rev
        return x + out


class RbfMamba2SavClassifier(nn.Module):
    """RBF 확장 → 단방향 Mamba2 × N → mean pool → MLP → SAV 로짓."""

    def __init__(
        self,
        num_prob_channels: int = _DEFAULT_NUM_CHANNELS,
        num_feature_channels: int = 0,
        num_scalar_features: int = 0,
        feature_rbf_expand: bool = False,
        rbf_num_centers: int = 16,
        rbf_gamma: float | None = None,
        rbf_trainable: bool = False,
        rbf_trainable_gamma: bool = False,
        use_marker: bool = False,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.1,
        pool: str = "mean",
        bidirectional: bool = False,
        pool_coverage_mask: bool = False,
        delta_channels: bool = False,
    ) -> None:
        """분류기를 초기화한다.

        Args:
            num_prob_channels (int): RBF로 확장할 확률 채널 수
                (ref/alt × acc/don/nei = 6).
            num_feature_channels (int): 뒤에 concat할 per-nt 피처 채널 수(intron +
                conservation). 0이면 기존 확률-only 동작(spec 017).
            num_scalar_features (int): per-variant 스칼라 피처 수(예: conservation
                @pos/mean/max × phastCons/phyloP = 6). pool 뒤 벡터에 concat 후
                MLP head. 0이면 기존 동작(spec 019 conservation scalar-at-pool).
            feature_rbf_expand (bool): 피처 채널에도 RBF를 적용할지 여부.
                기본 False(정규화 raw를 RBF 출력 뒤에 concat).
            rbf_num_centers (int): RBF center 수 K.
            rbf_gamma (float | None): RBF bandwidth (None이면 자동).
            rbf_trainable (bool): RBF center 학습 여부.
            rbf_trainable_gamma (bool): RBF gamma 학습 여부.
            use_marker (bool): 변이 위치 마커 채널(+1) 사용 여부.
            d_model (int): Mamba2 hidden 차원.
            n_layers (int): Mamba2 블록 수.
            d_state (int): SSM state 차원.
            d_conv (int): depthwise conv 폭.
            expand (int): inner 확장 배수.
            headdim (int): head 차원.
            dropout (float): MLP head dropout 확률.
            pool (str): readout pooling — ``"mean"`` 또는 ``"max"``. SAV
                신호가 sparse하므로 ``"max"``가 신호 보존에 유리.
            bidirectional (bool): True면 각 Mamba2 블록이 forward+reversed
                스캔을 합산(bi-Mamba). 기본 False(단방향).
            pool_coverage_mask (bool): True면 coverage가 0인 위치를 pooling에서
                제외한다. multichunk 캐시는 phase에 따라 최대 625 위치가 구조적으로
                덮이지 않고 그 자리의 확률은 0인데, ``RBF(0)``은 비영 코드이므로
                마스킹하지 않으면 max pooling이 미덮인 위치를 고를 수 있다.
                coverage는 첫 피처 채널(``num_prob_channels`` 인덱스)로 온다.
            delta_channels (bool): True면 확률 채널의 ref/alt 절반을 빼서
                ``alt-ref``와 ``|alt-ref|``를 raw 채널로 덧붙인다. 파이프라인은
                per-position ``|alt-ref|``를 chunk 선택에 쓰고 버리며(
                ``train/train_sav_e2e.py``의 ``collapse_multichunk``), head는
                RBF 확장된 ref/alt에서 뺄셈을 다시 배워야 한다 — RBF는 국소
                비선형 코드라 그 뺄셈이 선형으로 표현되지 않는다.
        """
        super().__init__()
        if pool not in ("mean", "max"):
            raise ValueError(f"pool은 'mean'|'max': {pool}")
        self.pool = pool
        self.pool_coverage_mask = pool_coverage_mask
        self.delta_channels = delta_channels
        self.use_marker = use_marker
        self.num_prob_channels = num_prob_channels
        self.num_feature_channels = num_feature_channels
        self.num_scalar_features = num_scalar_features
        self.feature_rbf_expand = feature_rbf_expand
        self.rbf = RbfExpansion(
            rbf_num_centers,
            gamma=rbf_gamma,
            trainable=rbf_trainable,
            trainable_gamma=rbf_trainable_gamma,
        )
        feat_dim = (
            num_feature_channels * rbf_num_centers
            if feature_rbf_expand
            else num_feature_channels
        )
        if delta_channels and num_prob_channels % 2:
            raise ValueError(
                f"delta_channels는 ref/alt가 짝을 이루어야 한다: {num_prob_channels}"
            )
        # delta는 [-1,1]이라 [0,1]을 가정하는 RBF에 넣지 않고 raw로 덧붙인다
        # (alt-ref와 |alt-ref| 각각 num_prob_channels//2개).
        delta_dim = num_prob_channels if delta_channels else 0
        in_dim = (
            num_prob_channels * rbf_num_centers
            + feat_dim
            + delta_dim
            + (1 if use_marker else 0)
        )
        self.input_proj = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                _Mamba2Block(
                    d_model,
                    d_state,
                    d_conv,
                    expand,
                    headdim,
                    bidirectional=bidirectional,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model + num_scalar_features, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        channels: torch.Tensor,
        marker: torch.Tensor | None = None,
        scalars: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """변이당 SAV 로짓을 산출한다.

        Args:
            channels (torch.Tensor): shape
                ``(batch_size, seq_len, num_prob_channels + num_feature_channels)``.
                앞 ``num_prob_channels``는 [0,1] 확률(RBF 대상), 나머지는 정규화된
                피처 채널.
            marker (torch.Tensor | None): shape
                ``(batch_size, seq_len, 1)``. ``use_marker``이면 필수.
            scalars (torch.Tensor | None): shape
                ``(batch_size, num_scalar_features)``. ``num_scalar_features>0``
                이면 필수 — pool 뒤 벡터에 concat.

        Returns:
            torch.Tensor: shape ``(batch_size,)`` 로짓.
        """
        prob = channels[..., : self.num_prob_channels]
        feats = self.rbf(prob)
        if self.delta_channels:
            half = self.num_prob_channels // 2
            delta = prob[..., half:] - prob[..., :half]
            feats = torch.cat([feats, delta, delta.abs()], dim=-1)
        if self.num_feature_channels > 0:
            extra = channels[..., self.num_prob_channels :]
            feats = torch.cat(
                [feats, self.rbf(extra) if self.feature_rbf_expand else extra],
                dim=-1,
            )
        if self.use_marker:
            if marker is None:
                raise ValueError("use_marker=True이면 marker가 필요하다")
            feats = torch.cat([feats, marker], dim=-1)
        hidden = self.input_proj(feats)
        for block in self.blocks:
            hidden = block(hidden)
        keep = None
        if self.pool_coverage_mask:
            if self.num_feature_channels < 1:
                raise ValueError(
                    "pool_coverage_mask=True이면 coverage 피처 채널이 필요하다"
                )
            keep = channels[..., self.num_prob_channels] > 0
        return self.pool_and_head(hidden, scalars, keep)

    def _pool(
        self, hidden: torch.Tensor, keep: torch.Tensor | None = None
    ) -> torch.Tensor:
        """위치 축으로 pooling한다 (mean 또는 max).

        Args:
            hidden (torch.Tensor): shape ``(batch_size, seq_len, d_model)``.
            keep (torch.Tensor | None): shape ``(batch_size, seq_len)`` bool.
                주어지면 False인 위치를 pooling에서 제외한다. 한 행이 전부
                False면 그 행은 마스크 없이 pooling한다(빈 pool 방지).

        Returns:
            torch.Tensor: shape ``(batch_size, d_model)``.
        """
        if keep is not None:
            keep = keep & keep.any(dim=1, keepdim=True)
            m = keep.unsqueeze(-1)
            if self.pool == "max":
                return hidden.masked_fill(~m, float("-inf")).max(dim=1).values
            n = keep.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
            return (hidden * m).sum(dim=1) / n
        if self.pool == "max":
            return hidden.max(dim=1).values
        return hidden.mean(dim=1)

    def pool_and_head(
        self,
        hidden: torch.Tensor,
        scalars: torch.Tensor | None = None,
        keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mamba2 출력에 final-norm → pooling → (scalar concat) → head를 적용한다.

        Args:
            hidden (torch.Tensor): shape
                ``(batch_size, seq_len, d_model)``.
            scalars (torch.Tensor | None): shape
                ``(batch_size, num_scalar_features)``. ``num_scalar_features>0``
                이면 필수 — pool 결과에 concat.
            keep (torch.Tensor | None): pooling에 포함할 위치 마스크
                ``(batch_size, seq_len)``. None이면 전 위치.

        Returns:
            torch.Tensor: shape ``(batch_size,)`` 로짓.
        """
        pooled = self._pool(self.final_norm(hidden), keep)
        if self.num_scalar_features > 0:
            if scalars is None:
                raise ValueError("num_scalar_features>0이면 scalars가 필요하다")
            pooled = torch.cat([pooled, scalars], dim=-1)
        return self.head(pooled).squeeze(-1)
