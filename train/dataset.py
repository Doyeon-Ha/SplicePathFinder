"""Datasets for splice-annotation training from HDF5 inputs."""



import bisect
import pickle
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get attribute from dict or NamedTuple-like object.

    Args:
        obj (Any): Object to get attribute from (dict, NamedTuple, or dataclass).
        key (str): Attribute/key name.
        default (Any): Default value if attribute doesn't exist.

    Returns:
        Any: The attribute value or default.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)







class SplicingH5Dataset(Dataset):
    """PyTorch Dataset for SpliceAI-format H5 files.

    Supports two storage formats (auto-detected):

    **Integer format** (preferred):
      - X{i}: (N, seq_len) int8 -- 0=N,1=A,2=C,3=G,4=T
      - Y{i}: (N, target_len) int8 -- 0=neither,1=acc,2=don,-1=pad

    **Legacy one-hot format**:
      - X{i}: (N, seq_len, 4) int8
      - Y{i}: (1, N, target_len, 3) int8

    Samples are flattened across all chunks for sequential
    indexing.
    """

    _INT_DECODE = np.array(["N", "A", "C", "G", "T"])
    _ONE_HOT_DECODE = np.array(["A", "C", "G", "T"])

    def __init__(
        self,
        h5_path: str,
        context_length: int | None = None,
        feature_h5_path: str | None = None,
        site_usage_soft_labels: bool = False,
        phylop_h5_path: str | None = None,
    ) -> None:
        """Initialize SplicingH5Dataset.

        Args:
            h5_path (str): Path to the SpliceAI H5 file.
            context_length (int | None): Desired context
                length (total flanking nucleotides).
                If None, use the maximum CL stored in
                the H5 file. Must be even and <= cl_max.
            feature_h5_path (str | None): Optional sidecar feature-H5 with
                ``FEAT{i}`` arrays row-aligned to ``X{i}`` (built by
                ``process_input.feature_sidecar_builder``). When set, each
                sample carries a ``"features"`` tensor of shape
                ``(seq_len, k)`` injected into the model's nt embeddings.
            site_usage_soft_labels (bool): Replace hard acceptor/donor targets
                with ``Y_SITE_USAGE`` at annotated sites. The file must encode
                canonical sites as 1 and measured added isoform sites as their
                usage rate.
            phylop_h5_path (str | None): Optional sidecar target-H5 with
                ``Y_PHYLOP{i}`` arrays row-aligned to ``Y{i}`` (built by
                ``process_input.phylop_target_sidecar_builder``). When set,
                each sample carries a ``"phylop_labels"`` float tensor of
                shape ``(target_len,)`` holding raw phyloP, NaN where the
                position has no value.

        Raises:
            ValueError: If a ``Y_PHYLOP{i}`` shape differs from its ``Y{i}``.
        """
        self.h5_path = h5_path
        self._file: h5py.File | None = None
        self.feature_h5_path = feature_h5_path
        self._feature_file: h5py.File | None = None
        self._has_features = feature_h5_path is not None
        self.site_usage_soft_labels = bool(site_usage_soft_labels)
        self.phylop_h5_path = phylop_h5_path
        self._phylop_file: h5py.File | None = None
        self._has_phylop = phylop_h5_path is not None

        with h5py.File(h5_path, "r") as f:
            x_keys = sorted(
                [k for k in f.keys()
                 if k.startswith("X")],
                key=lambda k: int(k[1:]),
            )
            self._chunk_info: (
                list[tuple[str, str, int]]
            ) = []
            for x_key in x_keys:
                y_key = f"Y{x_key[1:]}"
                size = f[x_key].shape[0]
                self._chunk_info.append(
                    (x_key, y_key, size)
                )

            self._integer_format = f["X0"].ndim == 2
            self._has_region_labels = "Y_REGION0" in f
            self._has_exon_labels = "Y_EXON0" in f
            self._has_usage_labels = "Y_USAGE0" in f
            self._has_site_usage_labels = "Y_SITE_USAGE0" in f

            x_dim = (
                f["X0"].shape[1]
                if self._integer_format
                else f["X0"].shape[1]
            )
            y_dim = (
                f["Y0"].shape[1]
                if self._integer_format
                else f["Y0"].shape[2]
            )

            if self.site_usage_soft_labels:
                if not self._integer_format:
                    raise ValueError(
                        "site_usage_soft_labels requires integer-format H5"
                    )
                if not self._has_site_usage_labels:
                    raise ValueError(
                        f"{h5_path} is missing Y_SITE_USAGE0"
                    )

        self._cl_max: int = x_dim - y_dim
        self._target_len: int = y_dim

        if self._has_phylop:
            # A row-count mismatch means the sidecar was built from a different
            # gene list or window geometry; every target would be shifted.
            with h5py.File(phylop_h5_path, "r") as fp:
                for x_key, _, size in self._chunk_info:
                    key = f"Y_PHYLOP{x_key[1:]}"
                    shape = fp[key].shape if key in fp else None
                    if shape != (size, y_dim):
                        raise ValueError(
                            f"{phylop_h5_path}: {key} shape {shape} != "
                            f"Y{x_key[1:]} shape ({size}, {y_dim})."
                        )

        if context_length is None:
            self._clip: int = 0
        else:
            if context_length > self._cl_max:
                raise ValueError(
                    f"context_length={context_length}"
                    f" > cl_max={self._cl_max}"
                )
            if context_length % 2 != 0:
                raise ValueError(
                    "context_length must be even,"
                    f" got {context_length}"
                )
            self._clip = (
                (self._cl_max - context_length) // 2
            )

        cumsum = 0
        self._cum_sizes: list[int] = []
        for _, _, size in self._chunk_info:
            cumsum += size
            self._cum_sizes.append(cumsum)
        self._length = cumsum

        return

    @property
    def cl_max(self) -> int:
        """Return maximum context length in the H5.

        Returns:
            int: Maximum context length.
        """
        return self._cl_max

    @property
    def target_len(self) -> int:
        """Return target region length in the H5.

        Returns:
            int: Target region length.
        """
        return self._target_len

    def _ensure_open(self) -> None:
        """Lazily open H5 file handle.

        h5py file handles cannot be forked across
        DataLoader workers, so each worker must open its
        own handle on first access.
        """
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        if self._has_features and self._feature_file is None:
            self._feature_file = h5py.File(self.feature_h5_path, "r")
        if self._has_phylop and self._phylop_file is None:
            self._phylop_file = h5py.File(self.phylop_h5_path, "r")

        return

    def __len__(self) -> int:
        """Return total number of samples.

        Returns:
            int: Dataset length.
        """
        return self._length

    def __getitem__(
        self, idx: int
    ) -> dict[str, torch.Tensor | str]:
        """Load sample at the given index.

        Args:
            idx (int): Sample index.

        Returns:
            dict[str, torch.Tensor | str]:
                'ref' (str) and 'labels' (Tensor [3, L]).
        """
        self._ensure_open()

        chunk_idx = bisect.bisect_right(
            self._cum_sizes, idx
        )
        local_idx = (
            idx - self._cum_sizes[chunk_idx - 1]
            if chunk_idx > 0
            else idx
        )

        x_key, y_key, _ = self._chunk_info[chunk_idx]

        if self._integer_format:
            return self._load_integer(
                x_key, y_key, local_idx
            )

        return self._load_onehot(
            x_key, y_key, local_idx
        )

    def _load_integer(
        self,
        x_key: str,
        y_key: str,
        local_idx: int,
    ) -> dict[str, torch.Tensor | str]:
        """Load a sample from integer-format H5.

        Args:
            x_key (str): X dataset key in H5 file.
            y_key (str): Y dataset key in H5 file.
            local_idx (int): Index within the chunk.

        Returns:
            dict[str, torch.Tensor | str]: A dict with
                'ref', 'labels', and -- when the H5 file
                carries auxiliary annotations --
                'region_labels' (long [L]) and
                'exon_labels' (long [L]) with -1 used for
                bases that should be ignored by the loss;
                'phylop_labels' (float [L], NaN = no value)
                when a phyloP sidecar is attached.
        """
        x = self._file[x_key][local_idx]
        y = self._file[y_key][local_idx]

        if self._clip > 0:
            x = x[self._clip : -self._clip]

        ref = "".join(self._INT_DECODE[x])

        suffix = x_key[1:]
        if self.site_usage_soft_labels:
            site_usage = self._file[f"Y_SITE_USAGE{suffix}"][local_idx]
            labels = torch.from_numpy(
                _site_usage_weighted_3class(y, site_usage)
            )
        else:
            y_t = torch.from_numpy(y.astype(np.int64))
            acc = (y_t == 1).long()
            don = (y_t == 2).long()
            nei = (y_t == 0).long()
            labels = torch.stack(
                [acc, don, nei], dim=0
            ).contiguous()

        sample: dict[str, torch.Tensor | str] = {
            "ref": ref,
            "labels": labels,
        }

        if self._has_region_labels:
            yr = self._file[f"Y_REGION{suffix}"][local_idx]
            sample["region_labels"] = torch.from_numpy(
                yr.astype(np.int64)
            )
        if self._has_exon_labels:
            ye = self._file[f"Y_EXON{suffix}"][local_idx]
            sample["exon_labels"] = torch.from_numpy(
                ye.astype(np.int64)
            )
        if self._has_usage_labels:
            yu = self._file[f"Y_USAGE{suffix}"][local_idx]
            sample["usage_labels"] = torch.from_numpy(
                yu.astype(np.float32)
            )
        if self._has_features:
            feat = self._feature_file[f"FEAT{suffix}"][local_idx]
            if self._clip > 0:
                feat = feat[self._clip : -self._clip]
            sample["features"] = torch.from_numpy(feat.astype(np.float32))
        if self._has_phylop:
            yp = self._phylop_file[f"Y_PHYLOP{suffix}"][local_idx]
            sample["phylop_labels"] = torch.from_numpy(yp.astype(np.float32))

        return sample

    def _load_onehot(
        self,
        x_key: str,
        y_key: str,
        local_idx: int,
    ) -> dict[str, torch.Tensor | str]:
        """Load a sample from legacy one-hot-format H5.

        Args:
            x_key (str): X dataset key in H5 file.
            y_key (str): Y dataset key in H5 file.
            local_idx (int): Index within the chunk.

        Returns:
            dict[str, torch.Tensor | str]:
                'ref' and 'labels'.
        """
        x = self._file[x_key][local_idx]
        y = self._file[y_key][0, local_idx]

        if self._clip > 0:
            x = x[self._clip : -self._clip]

        ref = self._onehot_to_seq(x)

        # One-hot channel order: (neither, acceptor, donor)
        # Internal label order:  (acceptor, donor, neither)
        y_reordered = y[:, [1, 2, 0]]
        labels = (
            torch.from_numpy(
                y_reordered.astype(np.int64)
            )
            .permute(1, 0)
            .contiguous()
        )

        return {
            "ref": ref,
            "labels": labels,
        }

    @classmethod
    def _onehot_to_seq(cls, x: np.ndarray) -> str:
        """Convert one-hot encoded array to DNA string.

        Args:
            x (np.ndarray): One-hot array of shape
                (seq_len, 4).

        Returns:
            str: DNA sequence string.
        """
        indices = np.argmax(x, axis=1)
        all_zero = np.all(x == 0, axis=1)
        chars = cls._ONE_HOT_DECODE[indices]
        chars[all_zero] = "N"
        return "".join(chars)

    def __del__(self) -> None:
        """Close H5 file handle on cleanup."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass

        return


_USAGE_IGNORE = -1.0


def _site_usage_weighted_3class(
    y3: np.ndarray, site_usage: np.ndarray
) -> np.ndarray:
    """Convert scalar site usage into the native 3-class soft target.

    Args:
        y3 (np.ndarray): Integer ``[L]`` labels: 1=acceptor, 2=donor,
            0=neither, -1=ignore/padding.
        site_usage (np.ndarray): Float ``[L]`` usage. Every annotated site
            must hold a value in [0,1]; other positions may be -1.

    Returns:
        np.ndarray: Float32 ``[3,L]`` target in acceptor, donor, neither order.

    Raises:
        ValueError: If an annotated site lacks a valid usage target.
    """
    y3 = np.asarray(y3)
    usage = np.asarray(site_usage, dtype=np.float32)
    if y3.shape != usage.shape:
        raise ValueError(f"Label/usage shape mismatch: {y3.shape} != {usage.shape}")

    sites = (y3 == 1) | (y3 == 2)
    invalid = sites & (~np.isfinite(usage) | (usage < 0.0) | (usage > 1.0))
    if np.any(invalid):
        raise ValueError(
            f"{int(invalid.sum()):,} annotated sites lack usage in [0,1]"
        )

    u = np.where(sites, usage, 0.0)
    acceptor = np.where(y3 == 1, u, 0.0)
    donor = np.where(y3 == 2, u, 0.0)
    neither = 1.0 - acceptor - donor
    return np.stack([acceptor, donor, neither]).astype(np.float32)


def _usage_weighted_3class(y3: np.ndarray, usage: np.ndarray) -> np.ndarray:
    """Turn the hard 3-class label into a usage-weighted soft target.

    A junction the table records as used carries a splice-site usage rate per
    tissue. Averaging the tissues that measured it gives one probability ``u``
    per position, which becomes the target of that position's acceptor or donor
    channel while ``neither`` takes the complement. A site used in a tenth of
    transcripts is then a tenth of a site rather than a whole one.

    Positions the table calls neither, and positions it cannot resolve at all
    (padding, and junctions whose GT-AG direction is ambiguous), both come out
    as ``neither=1`` -- the same target the hard path gives them, so the two
    label modes differ only at used junctions.

    Args:
        y3 (np.ndarray): Int ``[L]`` label; 1=acceptor, 2=donor, 0=neither,
            -1=ignore.
        usage (np.ndarray): Float ``[L, T]`` per-tissue usage in [0, 1] with
            ``-1.0`` where the tissue has no measurement.

    Returns:
        np.ndarray: Float32 ``[3, L]`` target ordered (acceptor, donor,
            neither), each column summing to 1.
    """
    measured = usage != _USAGE_IGNORE
    n_measured = measured.sum(axis=1)
    total = np.where(measured, usage, 0.0).sum(axis=1)
    # A used junction with no measured tissue keeps the hard target (u=1);
    # dividing by zero there would otherwise poison the column with NaN.
    u = np.divide(
        total,
        n_measured,
        out=np.ones_like(total, dtype=np.float32),
        where=n_measured > 0,
    )
    u = np.clip(u, 0.0, 1.0).astype(np.float32)

    acceptor = np.where(y3 == 1, u, 0.0)
    donor = np.where(y3 == 2, u, 0.0)
    neither = 1.0 - acceptor - donor
    return np.stack([acceptor, donor, neither]).astype(np.float32)
