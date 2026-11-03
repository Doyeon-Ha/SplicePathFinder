"""학습형 SAV 분류기용 Dataset.

per-nt feature 캐시 H5(``ref_probs``/``alt_probs`` ``(N, L, C)``)에서 변이를
중앙에 둔 ``window_left + window_right`` 길이 윈도를 잘라
``2*len(channel_indices)`` 확률 채널(ref subset + alt subset)로 반환한다.
단일 anchor 캐시는 청크(``L=15000``) 안에서 ``variant_offset``을 중심으로
잘라내고, multichunk 캐시(``build_multichunk_pernt_cache.py``)는 이미 변이
중심 8,750 윈도로 저장돼 있어 ``window_left`` attr로 그 사실을 알린다 —
어느 쪽이든 반환 텐서는 같은 의미다. multichunk 캐시는 위치마다 몇 개의
chunk가 그 자리를 덮었는지 세는 ``coverage``도 함께 담고 있으며, 있으면
채널 하나로 붙는다. 채널 수 ``C``는
``ref_probs.shape[-1]``에서 동적 인식한다(med_ac/SpliceAI=3 → 기본 6채널,
Pangolin=12 → ``channel_indices``로 조직별 splice-prob/usage subset; spec 019).

``feature_sidecar_h5``가 주어지면(spec 017 feature-twin) 행-정렬된 피처
sidecar에서 같은 윈도를 잘라 채널 뒤에 concat한다 —
``feature_families``로 intron(2ch)/conservation(2ch)을 선택한다. sidecar가
없으면 확률 채널만(기존 동작) 반환한다.
"""

from __future__ import annotations

import logging

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# sidecar dataset 이름과 채널 수 (concat 순서는 아래 튜플 순서를 따른다).
_FEATURE_DATASETS: dict[str, str] = {
    "intron": "feat_intron",
    "conservation": "feat_conservation",
}
_FEATURE_CHANNELS: dict[str, int] = {"intron": 2, "conservation": 2}


def clamp_impossible_directions(
    ref: np.ndarray, alt: np.ndarray, is_acc: np.ndarray, is_don: np.ndarray,
) -> np.ndarray:
    """``alt``를 ``ref``로 눌러 불가능한 방향의 델타를 0으로 만든다.

    ONBOARDING 4.54의 채널별 규칙 — 그 타입 site가 annotation에 **있는** 위치에서는
    loss만, **없는** 위치에서는 gain만 남긴다. 채널별이므로 annotated acceptor
    위치의 donor gain은 살아남는다. delta_max는 델타 트랙을 0으로 만들지만 head는
    델타를 만들지 않으므로 ``alt := ref``로 같은 결과를 낸다.

    Args:
        ref (np.ndarray): shape ``(L, C)`` ref 확률, 채널 순서 (acc, don, nei).
        alt (np.ndarray): shape ``(L, C)`` alt 확률.
        is_acc (np.ndarray): shape ``(L,)`` annotated acceptor 여부.
        is_don (np.ndarray): shape ``(L,)`` annotated donor 여부.

    Returns:
        np.ndarray: 클램프된 ``alt`` 복사본.
    """
    out = alt.copy()
    for channel, site in enumerate((is_acc, is_don)):
        gain = out[:, channel] > ref[:, channel]
        loss = out[:, channel] < ref[:, channel]
        kill = (site & gain) | (~site & loss)
        out[kill, channel] = ref[kill, channel]
    return out


class SavFeatureDataset(Dataset):
    """캐시된 ref/alt per-nt 확률(+선택적 피처)의 변이 중심 윈도를 제공한다."""

    def __init__(
        self,
        cache_h5: str,
        indices: np.ndarray,
        *,
        window_half_bp: int = 2500,
        window_left: int | None = None,
        window_right: int | None = None,
        use_coverage: bool | None = None,
        coverage_scale: float = 4.0,
        use_marker: bool = False,
        preload: bool = False,
        feature_sidecar_h5: str | None = None,
        feature_families: tuple[str, ...] = (),
        channel_indices: tuple[int, ...] | None = None,
        scalar_features_npz: str | None = None,
        scalar_features_key: str = "scalars",
        canonical_mask_h5: str | None = None,
        canonical_mask_snv_only: bool = False,
    ) -> None:
        """Dataset을 초기화한다.

        Args:
            cache_h5 (str): per-nt 확률 캐시 H5 경로
                (``ref_probs``/``alt_probs`` ``(N, L, C)``, ``labels``,
                ``variant_offset``). ``C``는 base마다 다르다(med_ac/SpliceAI=3,
                Pangolin=12) — 채널 수는 ``ref_probs.shape[-1]``에서 동적 인식.
            indices (np.ndarray): 사용할 캐시 행 인덱스(예: train 분할).
            window_half_bp (int): 변이 중심 crop 반폭. 입력 길이 =
                ``2*window_half_bp + 1``이며 변이는 정중앙에 온다.
                ``window_left``/``window_right``를 주면 무시된다.
            window_left (int | None): 변이 앞쪽으로 남길 위치 수. None이면
                캐시의 ``window_left`` attr(multichunk 캐시), 그것도 없으면
                ``window_half_bp``.
            window_right (int | None): 변이(포함) 뒤쪽으로 남길 위치 수.
                None이면 캐시 attr, 없으면 ``window_half_bp + 1``. 좌우를
                따로 두는 이유는 짝수 길이 윈도(multichunk의 8,750)가 변이를
                정중앙에 놓을 수 없어서다 — 왼쪽을 한 칸 더 준다.
            use_coverage (bool | None): 위치별 chunk 덮개 수를 채널로 붙일지.
                None이면 캐시에 ``coverage``가 있을 때만 붙인다. 확률 채널
                뒤, sidecar 피처 뒤에 온다.
            coverage_scale (float): coverage를 나눌 값. 다른 피처 채널처럼
                [0,1]에 두기 위한 것으로, 기본 4는 K=4 target-only grid에서
                한 위치를 덮을 수 있는 chunk 수의 상한이다(변이마다 정확히 4개
                — 임의의 정규화 상수가 아니다).
            use_marker (bool): 변이 위치 마커 채널 반환 여부.
            preload (bool): True면 crop된 윈도 채널을 RAM에 1회 선로딩한다.
            feature_sidecar_h5 (str | None): 행-정렬된 피처 sidecar H5 경로.
                None이면 확률 채널만 반환(기존 동작).
            feature_families (tuple[str, ...]): 사용할 피처 계열
                (``"intron"`` / ``"conservation"``). sidecar가 있을 때만 유효.
            channel_indices (tuple[int, ...] | None): ref/alt에서 골라 쓸 확률
                채널 인덱스(spec 019 Pangolin base swap). None이면 캐시의 전
                채널을 사용한다. 반환 확률 채널 수 = ``2 * len(channel_indices)``
                (ref subset + alt subset). med_ac/SpliceAI(3채널) 캐시는 None으로
                기존 6채널 동작을 유지한다.
            scalar_features_npz (str | None): per-variant 스칼라 피처 npz 경로.
                ``[key]`` 배열이 ``(N_cache, K)``로 캐시 행에 정렬돼 있어야 한다
                (row index로 lookup). None이면 스칼라 없음(기존 동작; spec 019
                conservation scalar-at-pool).
            scalar_features_key (str): 스칼라 npz 내 배열 키(기본 ``"scalars"``).
            canonical_mask_h5 (str | None): GENCODE v49 canonical site
                track(``acc_bits``/``don_bits``/``covered``, 캐시 행 정렬).
                주면 물리적으로 불가능한 방향에서 ``alt``를 ``ref``로
                클램프한다 — annotation에 그 타입 site가 있는 위치에서는
                gain을, 없는 위치에서는 loss를 지운다(채널별). ONBOARDING
                4.54의 delta_max 마스킹을 head 입력에 옮긴 것. None이면
                클램프 없음(기존 동작).
            canonical_mask_snv_only (bool): True면 substitution 행에만 클램프한다.
                indel은 v4 정렬 프레임이라 삽입 하류가 삽입 길이만큼 밀려
                genomic 좌표 기준 마스크가 어긋난다(ONBOARDING 4.54는 SNV에서만
                검증됐다). 마스크 파일의 ``is_snv``로 게이트한다.
        """
        self.cache_h5 = cache_h5
        self.indices = np.asarray(indices, dtype=np.int64)
        self.window_half_bp = int(window_half_bp)
        self.coverage_scale = float(coverage_scale)
        self.use_marker = use_marker
        self.feature_sidecar_h5 = feature_sidecar_h5
        self.feature_families = tuple(feature_families) if feature_sidecar_h5 else ()
        for fam in self.feature_families:
            if fam not in _FEATURE_DATASETS:
                raise ValueError(f"unknown feature family: {fam}")
        self.num_sidecar_channels = sum(
            _FEATURE_CHANNELS[f] for f in self.feature_families
        )
        # per-variant scalar features (cache-row aligned, e.g. conservation 6).
        self.scalar_features: np.ndarray | None = None
        self.num_scalar_features = 0
        if scalar_features_npz:
            arr = np.load(scalar_features_npz)[scalar_features_key]
            self.scalar_features = np.asarray(arr, dtype=np.float32)
            self.num_scalar_features = int(self.scalar_features.shape[1])
        self.canonical_mask_h5 = canonical_mask_h5
        self.canonical_mask_snv_only = bool(canonical_mask_snv_only)
        self._file: h5py.File | None = None
        self._feat_file: h5py.File | None = None
        self._mask_file: h5py.File | None = None
        self._windows: np.ndarray | None = None
        with h5py.File(cache_h5, "r") as f:
            self.total_length = int(f["ref_probs"].shape[1])
            self.num_cache_channels = int(f["ref_probs"].shape[2])
            # A multichunk cache is already variant-centred, so it carries no
            # per-row offset: the variant sits at window_left in every row.
            cache_left = f.attrs.get("window_left")
            if "variant_offset" in f:
                self.variant_offset = f["variant_offset"][:].astype(np.int64)
            elif cache_left is not None:
                self.variant_offset = np.full(
                    f["ref_probs"].shape[0], int(cache_left), dtype=np.int64
                )
            else:
                raise ValueError(
                    f"{cache_h5} has neither a 'variant_offset' dataset nor a "
                    f"'window_left' attribute, so the variant position within "
                    f"a row is unknown"
                )
            self.window_left = int(
                window_left if window_left is not None
                else (cache_left if cache_left is not None
                      else self.window_half_bp)
            )
            cache_right = f.attrs.get("window_right")
            self.window_right = int(
                window_right if window_right is not None
                else (cache_right if cache_right is not None
                      else self.window_half_bp + 1)
            )
            self.window_len = self.window_left + self.window_right
            if self.window_len != 2 * self.window_half_bp + 1:
                # The caller asked for one width and the cache declares
                # another; say which one won rather than silently resizing
                # the head's input.
                logger.info(
                    "window %d (-%d..+%d) from the cache, not the requested "
                    "%d from window_half_bp=%d",
                    self.window_len, self.window_left, self.window_right - 1,
                    2 * self.window_half_bp + 1, self.window_half_bp,
                )
            self.use_coverage = (
                "coverage" in f if use_coverage is None else bool(use_coverage)
            )
            # Coverage joins the sidecar channels as a per-nt feature: the head
            # splits its input at num_prob_channels and treats everything after
            # as features, so it has to be counted here to be wired up at all.
            self.num_feature_channels = self.num_sidecar_channels + int(
                self.use_coverage
            )
            if self.use_coverage and "coverage" not in f:
                raise ValueError(
                    f"use_coverage=True but {cache_h5} has no 'coverage' "
                    f"dataset (single-anchor caches do not)"
                )
            self.labels = f["labels"][:].astype(np.float32)
            if channel_indices is None:
                self.channel_indices = list(range(self.num_cache_channels))
            else:
                self.channel_indices = list(channel_indices)
                for c in self.channel_indices:
                    if not 0 <= c < self.num_cache_channels:
                        raise ValueError(
                            f"channel_indices {c} out of range "
                            f"[0,{self.num_cache_channels})"
                        )
            self.num_prob_channels = 2 * len(self.channel_indices)
            self.num_channels = (
                self.num_prob_channels + self.num_feature_channels
            )
            if preload:
                self._windows = self._preload_windows(f)

    def _preload_windows(self, handle: h5py.File) -> np.ndarray:
        """crop된 전체 채널 윈도를 RAM으로 1회 선로딩한다.

        Args:
            handle (h5py.File): 열린 확률 캐시 핸들.

        Returns:
            np.ndarray: shape ``(len(indices), window_len, num_channels)`` fp16.
        """
        feat_handle = (
            h5py.File(self.feature_sidecar_h5, "r")
            if self.feature_sidecar_h5
            else None
        )
        # _assemble reads the mask through self._mask_file, and preloading runs
        # before _ensure_open ever does, so open it here or the clamp is silently
        # skipped for every preloaded window.
        if self.canonical_mask_h5:
            self._mask_file = h5py.File(self.canonical_mask_h5, "r")
        out = np.zeros(
            (len(self.indices), self.window_len, self.num_channels),
            dtype=np.float16,
        )
        for i, row in enumerate(self.indices):
            out[i] = self._assemble(handle, feat_handle, int(row))
        if feat_handle is not None:
            feat_handle.close()
        if self._mask_file is not None:
            self._mask_file.close()
            self._mask_file = None
        logger.info(
            "preload 완료: %d windows (%d채널)", len(self.indices), self.num_channels
        )
        return out

    def _ensure_open(self) -> tuple[h5py.File, h5py.File | None]:
        """워커별로 H5 핸들을 lazy open한다 (fork 안전).

        Returns:
            tuple[h5py.File, h5py.File | None]: 확률 캐시 핸들과
                (있으면) 피처 sidecar 핸들.
        """
        if self._file is None:
            self._file = h5py.File(self.cache_h5, "r")
            if self.feature_sidecar_h5:
                self._feat_file = h5py.File(self.feature_sidecar_h5, "r")
            if self.canonical_mask_h5:
                self._mask_file = h5py.File(self.canonical_mask_h5, "r")
        return self._file, self._feat_file

    def __len__(self) -> int:
        """샘플 수를 반환한다.

        Returns:
            int: 인덱스 개수.
        """
        return len(self.indices)

    def _crop(self, arr: np.ndarray, var_pos: int, n_ch: int) -> np.ndarray:
        """변이 중심 윈도를 zero-pad와 함께 잘라낸다.

        Args:
            arr (np.ndarray): shape ``(L, n_ch)`` per-nt 배열.
            var_pos (int): 변이 위치(model space).
            n_ch (int): 채널 수.

        Returns:
            np.ndarray: shape ``(window_len, n_ch)``.
        """
        out = np.zeros((self.window_len, n_ch), dtype=np.float32)
        start = var_pos - self.window_left
        end = var_pos + self.window_right
        src_start = max(0, start)
        src_end = min(self.total_length, end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        out[dst_start:dst_end] = arr[src_start:src_end]
        return out

    def _clamp_canonical(
        self, ref: np.ndarray, alt: np.ndarray, row: int
    ) -> np.ndarray:
        """물리적으로 불가능한 방향에서 ``alt``를 ``ref``로 클램프한다.

        채널별 규칙은 ONBOARDING 4.54의 delta_max 마스킹과 같다 — annotation에
        그 타입 site가 있는 위치에서는 loss만, 없는 위치에서는 gain만 남긴다.
        head는 델타를 만들지 않으므로 ``alt := ref``로 눌러 델타를 0으로 만든다.
        annotation에 없는 유전자의 행은 ``covered``가 False라 손대지 않는다.

        Args:
            ref (np.ndarray): shape ``(L, C)`` ref 확률, 채널 순서 (acc, don, nei).
            alt (np.ndarray): shape ``(L, C)`` alt 확률.
            row (int): 캐시 행 인덱스(마스크와 공통).

        Returns:
            np.ndarray: 클램프된 ``alt`` 복사본.
        """
        mask_file = self._mask_file
        if mask_file is None or not bool(mask_file["covered"][row]):
            return alt
        if self.canonical_mask_snv_only and not bool(mask_file["is_snv"][row]):
            return alt
        length = ref.shape[0]
        return clamp_impossible_directions(
            ref, alt,
            np.unpackbits(mask_file["acc_bits"][row])[:length].astype(bool),
            np.unpackbits(mask_file["don_bits"][row])[:length].astype(bool),
        )

    def _assemble(
        self, handle: h5py.File, feat_handle: h5py.File | None, row: int
    ) -> np.ndarray:
        """한 행의 확률 6채널(+피처 채널)을 crop·concat한다.

        Args:
            handle (h5py.File): 확률 캐시 핸들.
            feat_handle (h5py.File | None): 피처 sidecar 핸들.
            row (int): 캐시/​sidecar 공통 행 인덱스.

        Returns:
            np.ndarray: shape ``(window_len, num_channels)`` float32.
        """
        var_pos = int(self.variant_offset[row])
        ci = self.channel_indices
        n_sel = len(ci)
        ref_full = np.asarray(handle["ref_probs"][row], dtype=np.float32)
        alt_full = np.asarray(handle["alt_probs"][row], dtype=np.float32)
        if self._mask_file is not None:
            alt_full = self._clamp_canonical(ref_full, alt_full, row)
        ref = self._crop(ref_full[:, ci], var_pos, n_sel)
        alt = self._crop(alt_full[:, ci], var_pos, n_sel)
        parts = [ref, alt]
        if feat_handle is not None:
            for fam in self.feature_families:
                ds = feat_handle[_FEATURE_DATASETS[fam]]
                cropped = self._crop(
                    ds[row].astype(np.float32), var_pos, _FEATURE_CHANNELS[fam]
                )
                parts.append(cropped)
        if self.use_coverage:
            cov = handle["coverage"][row].astype(np.float32)[:, None]
            parts.append(self._crop(cov, var_pos, 1) / self.coverage_scale)
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """한 변이의 윈도 채널·라벨(·마커)을 반환한다.

        Args:
            i (int): dataset 내 인덱스.

        Returns:
            dict[str, torch.Tensor]: ``channels`` ``(window_len, num_channels)``,
                ``label`` 스칼라, 그리고 ``use_marker``이면 ``marker``
                ``(window_len, 1)``.
        """
        row = int(self.indices[i])
        if self._windows is not None:
            channels = self._windows[i].astype(np.float32)
        else:
            handle, feat_handle = self._ensure_open()
            channels = self._assemble(handle, feat_handle, row)
        item: dict[str, torch.Tensor] = {
            "channels": torch.from_numpy(channels),
            "label": torch.tensor(self.labels[row], dtype=torch.float32),
        }
        if self.use_marker:
            positions = np.arange(self.window_len, dtype=np.float32)
            marker = (positions - self.window_left) / self.window_len
            item["marker"] = torch.from_numpy(marker).unsqueeze(-1)
        if self.scalar_features is not None:
            item["scalars"] = torch.from_numpy(self.scalar_features[row].copy())
        return item
