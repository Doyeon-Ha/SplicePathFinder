"""Dataset classes for loading splicing data.

After the v2 config revamp the supervised splicing pipeline consumes only
the SpliceAI-format H5 datasets (``SplicingH5Dataset``). The LMDB dataset
(``SplicingLMDBDataset``) is retained solely as a dependency of the MLM
pretraining script (``train/train_mlm.py``); it is no longer used by the
main training entry point.

``SplicingH5MLMDataset`` is a lightweight H5-backed dataset for MLM
pretraining on random-genome windows (built by
``scripts/build_mlm_random_genome_h5.py``).
"""

import bisect
import pickle
from typing import Any

import h5py
import lmdb
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


class SplicingLMDBDataset(Dataset):
    """
    PyTorch Dataset for lazy-loading splicing data from LMDB.

    Each sample contains the following fields:
    - ref: reference sequence
    - labels: 3 x seq_len tensor containing acceptor, donor, and neither labels
    """

    def __init__(
        self,
        lmdb_path: str,
        transform: object | None = None,
        preload_keys: bool = False,
    ):
        """
        Initialize the SplicingLMDBDataset.

        Args:
            lmdb_path (str): Path to the LMDB file.
            transform (Any | None): Data transformation function (optional).
            preload_keys (bool): Whether to preload all keys at initialization.
        """
        self.lmdb_path = lmdb_path
        self.transform = transform

        # Set up LMDB environment
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,  # Set to False for lazy loading
            meminit=False,
        )

        # Dataset size and key information
        with self.env.begin() as txn:
            self.length = txn.stat()["entries"]

            if preload_keys:
                # Preload all keys (memory efficient)
                self.keys = [key for key, _ in txn.cursor()]
            else:
                self.keys = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """
        Load the sample at the given index and convert to tensor.

        Returns:
            dict containing:
            - 'ref': reference sequence string
            - 'labels': 3 x seq_len tensor containing acceptor, donor, and neither
                labels
        """
        # Start transaction
        with self.env.begin() as txn:
            # Get key
            if self.keys is not None:
                key = self.keys[idx]
            else:
                # If keys are not preloaded
                cursor = txn.cursor()
                cursor.set_range(idx.to_bytes(4, "big"))
                key = cursor.key()

            # Get value
            value = txn.get(key)
            if value is None:
                raise IndexError(f"Index {idx} not found in LMDB")

            # Deserialize
            sample = pickle.loads(value)

        # Extract required fields - keep sequences as strings, convert labels to tensors
        processed_sample = {
            "ref": self._to_string(_safe_get(sample, "ref")),
        }

        # Optional metadata: chromosome identifier (various possible keys)
        chrom_value = (
            _safe_get(sample, "chrom")
            or _safe_get(sample, "chr")
            or _safe_get(sample, "chromosome")
        )
        if chrom_value is not None:
            try:
                processed_sample["chrom"] = self._to_string(chrom_value)
            except Exception:
                # Best-effort fallback
                processed_sample["chrom"] = str(chrom_value)

        # Convert labels with proper tensor creation
        acceptor = self._to_tensor(_safe_get(sample, "acceptor"))
        donor = self._to_tensor(_safe_get(sample, "donor"))

        # Compute neither: 1 where both acceptor and donor are 0, else 0
        neither = torch.where(
            (acceptor == 0) & (donor == 0),
            torch.ones_like(acceptor),
            torch.zeros_like(acceptor),
        )

        # Stack into 3 x seq_len tensor with independent memory
        labels = torch.stack([acceptor, donor, neither], dim=0)
        processed_sample["labels"] = self._ensure_independent_tensor(labels)

        # Apply transform if available
        if self.transform:
            processed_sample = self.transform(processed_sample)

        return processed_sample

    def _ensure_independent_tensor(
        self, data: object, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """
        Ensure tensor has independent memory allocation (not a view).

        Args:
            data: Input data
            dtype: Desired dtype for the tensor

        Returns:
            torch.Tensor with independent memory
        """
        if isinstance(data, torch.Tensor):
            # Use detach().clone() to ensure complete independence
            tensor = data.detach().clone()
        elif isinstance(data, np.ndarray):
            # Create new tensor from numpy array (copy data)
            tensor = torch.from_numpy(data.copy())
        else:
            # Create new tensor from other data types
            tensor = torch.tensor(data)

        # Convert dtype if specified
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)

        # Ensure contiguous memory layout
        return tensor.contiguous()

    def _to_string(self, data: object) -> str:
        """
        Convert sequence data to string.

        Args:
            data (object): Data to be converted (bytes, numpy array, etc.)

        Returns:
            str: Converted sequence string (already uppercase from preprocessing)
        """
        if data is None:
            return ""

        # If data is bytes/bytearray (likely sequence data)
        if isinstance(data, (bytes, bytearray)):
            try:
                # Try decoding as string (already uppercase from preprocessing)
                decoded = data.decode("utf-8")
                return decoded
            except Exception as e:
                raise ValueError(f"Error decoding sequence: {e}")

        # If data is already a string
        elif isinstance(data, str):
            return data

        # Convert other types to string
        else:
            return str(data)

    def _to_tensor(self, data: object, length: int | None = None) -> torch.Tensor:
        """
        Convert data to a PyTorch tensor.

        Args:
            data (object): Data to be converted (bytes, numpy array, list, etc.)
            length (int): Length of the sequence to decode

        Returns:
            torch.Tensor: Converted data with independent memory
        """
        # When data is None, return zeros if length is provided, else empty tensor
        if data is None:
            if length is None:
                return torch.tensor([], dtype=torch.long)
            return torch.zeros(length, dtype=torch.long)

        # If data is bytes/bytearray (likely numeric data)
        if isinstance(data, (bytes, bytearray)):
            try:
                # Try to interpret as numpy array first
                arr = np.frombuffer(data, dtype=np.uint8)
                # Unpack from np.packbits
                unpacked = np.unpackbits(arr)
                if length is not None:
                    unpacked = unpacked[:length]
                # Create new tensor with copy of data
                # Using torch.tensor() instead of torch.from_numpy() to ensure copy
                return torch.tensor(unpacked, dtype=torch.long)
            except Exception:
                # If that fails, convert to list of integers
                return torch.tensor(list(data)[:length], dtype=torch.long)

        # If data is a numpy array
        elif isinstance(data, np.ndarray):
            # Ensure we get a 1D array of the correct length
            if data.ndim > 1:
                data = data.flatten()
            if length is not None:
                if len(data) != length:
                    # Pad or truncate as necessary
                    if len(data) < length:
                        data = np.pad(data, (0, length - len(data)), constant_values=0)
                    else:
                        data = data[:length]
            # Create new tensor with copy of data
            return torch.tensor(data, dtype=torch.long)

        # If data is a list or other sequence
        else:
            # Convert to list safely
            tensor_data = [] if data is None else list(data)
            if length is not None:
                tensor_data = tensor_data[:length]
                # Pad if necessary
                if len(tensor_data) < length:
                    tensor_data.extend([0] * (length - len(tensor_data)))
            return torch.tensor(tensor_data, dtype=torch.long)



class SplicingH5MLMDataset(Dataset):
    """Lazy-loading H5 dataset for MLM pretraining on random genome windows.

    Reads windows produced by ``scripts/build_mlm_random_genome_h5.py``
    (int8 ``X{i}`` arrays and string ``chrom{i}`` arrays). Each ``__getitem__``
    decodes the int8 row back to an ASCII DNA string and returns a dict
    ``{"ref": <str>, "chrom": <str>}`` matching the schema that
    ``HyenaMLMDataset`` in ``train/train_mlm.py`` expects.

    The H5 file handle is opened lazily on first access so the dataset
    is picklable for DataLoader workers (each worker opens its own handle).

    Attributes:
        h5_path (str): Path to the H5 file.
        length (int): Total number of samples across all groups.
        window_length (int): Per-sample sequence length.
    """

    # int8 code -> ASCII byte for fast decoding via numpy fancy indexing.
    _DECODE_TABLE = np.array(
        [ord("N"), ord("A"), ord("C"), ord("G"), ord("T")],
        dtype=np.uint8,
    )

    def __init__(self, h5_path: str) -> None:
        """Initialize the MLM H5 dataset.

        Args:
            h5_path (str): Path to the random-genome H5 file.

        Raises:
            FileNotFoundError: If ``h5_path`` does not exist.
            ValueError: If the H5 has no ``X{i}`` groups or zero samples.
        """
        self.h5_path = h5_path
        self._file: h5py.File | None = None
        # Scan once at init to build a flat sample index across groups.
        with h5py.File(h5_path, "r") as f:
            x_keys = sorted(
                (k for k in f.keys() if k.startswith("X")),
                key=lambda k: int(k[1:]),
            )
            if not x_keys:
                raise ValueError(f"No 'X*' groups found in {h5_path}")
            self._x_keys: list[str] = x_keys
            self._chrom_keys: list[str] = [
                k.replace("X", "chrom", 1) for k in x_keys
            ]
            self._group_sizes: list[int] = [int(f[k].shape[0]) for k in x_keys]
            self.window_length: int = int(f[x_keys[0]].shape[1])
        # Cumulative size table for O(log n) sample -> group lookup.
        self._cumsum: np.ndarray = np.concatenate(
            [[0], np.cumsum(self._group_sizes)]
        )
        self.length: int = int(self._cumsum[-1])
        if self.length == 0:
            raise ValueError(f"H5 dataset has zero samples: {h5_path}")

    def _ensure_open(self) -> None:
        """Open the H5 file handle if not yet opened (per worker)."""
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, str]:
        """Return the decoded sample at ``idx``.

        Args:
            idx (int): Global sample index in ``[0, len(self))``.

        Returns:
            dict[str, str]: ``{"ref": <DNA string>, "chrom": <"chr:start" tag>}``.

        Raises:
            IndexError: If ``idx`` is out of bounds.
        """
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range [0, {self.length})")
        self._ensure_open()
        group_id = int(np.searchsorted(self._cumsum, idx, side="right") - 1)
        local_idx = idx - int(self._cumsum[group_id])
        encoded = self._file[self._x_keys[group_id]][local_idx]
        # Map int8 0..4 -> ASCII byte -> Python str.
        decoded = self._DECODE_TABLE[encoded].tobytes().decode("ascii")
        chrom_raw = self._file[self._chrom_keys[group_id]][local_idx]
        if isinstance(chrom_raw, bytes):
            chrom_tag = chrom_raw.decode("utf-8")
        else:
            chrom_tag = str(chrom_raw)
        return {"ref": decoded, "chrom": chrom_tag}


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


class PangolinSSUDataset(Dataset):
    """Multi-file dataset for the Pangolin multi-species SSU build.

    Reads per-species H5 files produced by
    ``process_input.pangolin_multispecies_dataset_builder`` and flattens their
    chunks into one index, so a single DataLoader trains across all species
    (cross-species augmentation; the model sees raw DNA with no species
    conditioning, as in Pangolin).

    Each file stores, per chunk ``i``:
        X{i}:         (N, SL+CL_max) int8   sequence tokens (0=N,1=A,..,4=T)
        Y_SPLICED{i}: (N, SL, 4) int8       spliced/not per tissue; -1 ignore
        Y_USAGE{i}:   (N, SL, 4) float32    usage in [0,1]; -1.0 ignore

    ``__getitem__`` returns ``{"ref": str, "spliced_labels": Long[L,4],
    "usage_labels": Float[L,4]}`` to match the existing trainer's
    string-sequence contract (the collator re-tokenizes ``ref``).
    """

    _INT_DECODE = np.array(["N", "A", "C", "G", "T"])

    def __init__(
        self,
        h5_paths: list[str],
        context_length: int | None = None,
        soft_usage_labels: bool = False,
    ) -> None:
        """Initialize the multi-species SSU dataset.

        Args:
            h5_paths (list[str]): Per-species H5 paths to concatenate.
            context_length (int | None): Desired context length (even,
                <= cl_max). None keeps the full stored context.
            soft_usage_labels (bool): Emit the 3-class target as float
                usage-weighted probabilities instead of a hard one-hot. The
                acceptor/donor channel carries the tissue-mean usage rate and
                ``neither`` carries its complement. Requires
                ``training.classification.use_hard_labels=false``, which would
                otherwise round the target back to one-hot.
        """
        if not h5_paths:
            raise ValueError("h5_paths must be non-empty")
        self.h5_paths = list(h5_paths)
        self.soft_usage_labels = bool(soft_usage_labels)
        self._files: list[h5py.File | None] = [None] * len(self.h5_paths)

        self._chunk_info: list[tuple[int, str, int]] = []
        x_dim = y_dim = None
        for fi, path in enumerate(self.h5_paths):
            with h5py.File(path, "r") as f:
                if "Y_SPLICED0" not in f or "Y_USAGE0" not in f:
                    raise ValueError(f"{path} missing Y_SPLICED0/Y_USAGE0")
                x_keys = sorted(
                    [k for k in f.keys() if k.startswith("X")],
                    key=lambda k: int(k[1:]),
                )
                for x_key in x_keys:
                    self._chunk_info.append((fi, x_key, f[x_key].shape[0]))
                if x_dim is None:
                    x_dim = f["X0"].shape[1]
                    y_dim = f["Y_SPLICED0"].shape[1]

        self._cl_max = int(x_dim - y_dim)
        self._target_len = int(y_dim)

        if context_length is None:
            self._clip = 0
        else:
            if context_length > self._cl_max:
                raise ValueError(
                    f"context_length={context_length} > cl_max={self._cl_max}"
                )
            if context_length % 2 != 0:
                raise ValueError("context_length must be even")
            self._clip = (self._cl_max - context_length) // 2

        cumsum = 0
        self._cum_sizes: list[int] = []
        for _, _, size in self._chunk_info:
            cumsum += size
            self._cum_sizes.append(cumsum)
        self._length = cumsum

    @property
    def cl_max(self) -> int:
        """Maximum context length stored in the H5 files."""
        return self._cl_max

    @property
    def target_len(self) -> int:
        """Target region length (SL)."""
        return self._target_len

    def _ensure_open(self) -> None:
        """Lazily open per-file H5 handles (one set per DataLoader worker)."""
        for fi, path in enumerate(self.h5_paths):
            if self._files[fi] is None:
                self._files[fi] = h5py.File(path, "r")

    def __len__(self) -> int:
        """Total number of samples across all species files."""
        return self._length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Load one datapoint.

        Args:
            idx (int): Flattened sample index.

        Returns:
            dict[str, torch.Tensor | str]: ``ref`` sequence string,
                ``spliced_labels`` Long[L,4], ``usage_labels`` Float[L,4].
        """
        self._ensure_open()
        chunk_idx = bisect.bisect_right(self._cum_sizes, idx)
        local_idx = (
            idx - self._cum_sizes[chunk_idx - 1] if chunk_idx > 0 else idx
        )
        fi, x_key, _ = self._chunk_info[chunk_idx]
        suffix = x_key[1:]
        f = self._files[fi]

        x = f[x_key][local_idx]
        if self._clip > 0:
            x = x[self._clip : -self._clip]
        ref = "".join(self._INT_DECODE[x])

        ys = f[f"Y_SPLICED{suffix}"][local_idx]
        yu = f[f"Y_USAGE{suffix}"][local_idx]
        # Cryptic-aware 3-class label -> the trainer's native classification
        # head (med_ac-compatible). Y_3CLASS is 0=neither/1=acc/2=don/-1=ignore
        # (canonical + used-cryptic junctions, direction by GT-AG motif). Encode
        # as [3, L] acc/don/neither; -1 (ignore) -> all-zero column, treated as
        # ignore by the masked loss (same convention as SplicingH5Dataset).
        y3 = f[f"Y_3CLASS{suffix}"][local_idx].astype(np.int64)
        if self.soft_usage_labels:
            labels = torch.from_numpy(_usage_weighted_3class(y3, yu))
        else:
            labels = torch.from_numpy(
                np.stack([(y3 == 1), (y3 == 2), (y3 == 0)]).astype(np.int64)
            )
        return {
            "ref": ref,
            "labels": labels,
            "spliced_labels": torch.from_numpy(ys.astype(np.int64)),
            "usage_labels": torch.from_numpy(yu.astype(np.float32)),
        }

    def __del__(self) -> None:
        """Close any open H5 handles."""
        for f in self._files:
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
