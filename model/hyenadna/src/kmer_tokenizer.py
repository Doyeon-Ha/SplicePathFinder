"""Overlapping k-mer tokenizer for DNA sequences.

Converts DNA sequences into overlapping k-mer token IDs.
Output length = input_nucleotide_length - k + 1.

Vocabulary layout (sequential IDs):
  - 7 special tokens
  - 4^k canonical k-mers (ACGT only)
  - N-prefix k-mers: N{p}[ACGT]{k-p}, p=1..k
  - N-suffix k-mers: [ACGT]{k-s}N{s}, s=1..k-1

N only appears as a contiguous prefix or suffix;
mixed patterns like ACNTGA are mapped to [UNK].
"""

from itertools import product
from typing import Any

import torch

NUM_SPECIAL_TOKENS = 7
SPECIAL_TOKENS: dict[str, int] = {
    "[CLS]": 0,
    "[SEP]": 1,
    "[BOS]": 2,
    "[MASK]": 3,
    "[PAD]": 4,
    "[RESERVED]": 5,
    "[UNK]": 6,
}
BASES = "ACGT"


class OverlappingKmerTokenizer:
    """Overlapping k-mer tokenizer for DNA sequences.

    Slides a window of size k across the input sequence
    with stride 1, producing one token per window position.

    Supports canonical bases (A, C, G, T) and the
    ambiguous base N when it appears as a contiguous
    prefix or suffix of a k-mer. K-mers with N in
    non-prefix/suffix positions map to [UNK].

    Attributes:
        k: K-mer size.
        vocab_size: Total vocabulary size.
        pad_token_id: ID of the [PAD] token.
        unk_token_id: ID of the [UNK] token.
        mask_token_id: ID of the [MASK] token.
        model_max_length: Maximum token sequence length.
    """

    def __init__(
        self,
        k: int,
        model_max_length: int | None = None,
    ) -> None:
        """Initialize the overlapping k-mer tokenizer.

        Args:
            k (int): K-mer size (e.g. 6 for 6-mers).
            model_max_length (int | None): Maximum token
                sequence length used for padding.
        """
        self.k = k
        self.model_max_length = model_max_length
        self.pad_token_id = SPECIAL_TOKENS["[PAD]"]
        self.unk_token_id = SPECIAL_TOKENS["[UNK]"]
        self.mask_token_id = SPECIAL_TOKENS["[MASK]"]

        self._vocab: dict[str, int] = {}
        self._kmer_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._build_vocab()

        return

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens.

        Returns:
            int: Number of tokens in the vocabulary.
        """
        return len(self._vocab)

    def _build_vocab(self) -> None:
        """Build k-mer vocabulary.

        Token ID layout (sequential):
          1. 7 special tokens
          2. 4^k canonical ACGT-only k-mers
          3. N-prefix: N{p}[ACGT]{k-p}, p=1..k
          4. N-suffix: [ACGT]{k-s}N{s}, s=1..k-1
        """
        self._vocab = dict(SPECIAL_TOKENS)
        self._kmer_to_id = {}
        next_id = NUM_SPECIAL_TOKENS

        for kmer_tuple in product(BASES, repeat=self.k):
            kmer = "".join(kmer_tuple)
            self._vocab[kmer] = next_id
            self._kmer_to_id[kmer] = next_id
            next_id += 1

        for p in range(1, self.k + 1):
            n_prefix = "N" * p
            acgt_len = self.k - p
            if acgt_len == 0:
                self._vocab[n_prefix] = next_id
                self._kmer_to_id[n_prefix] = next_id
                next_id += 1
            else:
                for suffix in product(
                    BASES, repeat=acgt_len
                ):
                    kmer = n_prefix + "".join(suffix)
                    self._vocab[kmer] = next_id
                    self._kmer_to_id[kmer] = next_id
                    next_id += 1

        for s in range(1, self.k):
            n_suffix = "N" * s
            acgt_len = self.k - s
            for prefix_tuple in product(
                BASES, repeat=acgt_len
            ):
                kmer = "".join(prefix_tuple) + n_suffix
                self._vocab[kmer] = next_id
                self._kmer_to_id[kmer] = next_id
                next_id += 1

        self._id_to_token = {
            v: k for k, v in self._vocab.items()
        }

        return

    def get_vocab(self) -> dict[str, int]:
        """Return the full vocabulary mapping.

        Returns:
            dict[str, int]: Token string to ID mapping.
        """
        return dict(self._vocab)

    def _encode_sequence(
        self, sequence: str
    ) -> list[int]:
        """Encode a single DNA sequence into k-mer IDs.

        Args:
            sequence (str): DNA sequence string.

        Returns:
            list[int]: List of k-mer token IDs.
        """
        tokens: list[int] = []
        seq = sequence.upper()
        end = len(seq) - self.k + 1
        for i in range(end):
            kmer = seq[i : i + self.k]
            tokens.append(
                self._kmer_to_id.get(
                    kmer, self.unk_token_id
                )
            )
        return tokens

    def __call__(
        self,
        sequences: str | list[str],
        return_tensors: str | None = None,
        padding: str | bool | None = None,
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Tokenize a batch of DNA sequences.

        Mimics the HuggingFace PreTrainedTokenizer.__call__
        interface for drop-in compatibility.

        Args:
            sequences (str | list[str]): One or more DNA
                sequences.
            return_tensors (str | None): If "pt", returns
                PyTorch tensors.
            padding (str | bool | None): Padding strategy.
                "max_length" or True pads to max_length;
                "longest" pads to the longest in batch.
            truncation (bool): Truncate to max_length.
            max_length (int | None): Target length for
                padding/truncation.
            add_special_tokens (bool): Unused, kept for
                API compatibility.
            return_offsets_mapping (bool): Unused, kept
                for API compatibility.
            **kwargs: Additional keyword arguments
                (ignored).

        Returns:
            dict[str, Any]: Dictionary with "input_ids".
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        all_ids = [
            self._encode_sequence(seq)
            for seq in sequences
        ]

        if truncation and max_length is not None:
            all_ids = [
                ids[:max_length] for ids in all_ids
            ]

        if padding in ("max_length", True):
            target_len = (
                max_length
                or self.model_max_length
                or max(len(ids) for ids in all_ids)
            )
            for i in range(len(all_ids)):
                deficit = target_len - len(all_ids[i])
                if deficit > 0:
                    all_ids[i] += (
                        [self.pad_token_id] * deficit
                    )
        elif padding == "longest":
            longest = max(len(ids) for ids in all_ids)
            for i in range(len(all_ids)):
                deficit = longest - len(all_ids[i])
                if deficit > 0:
                    all_ids[i] += (
                        [self.pad_token_id] * deficit
                    )

        attention_masks = [
            [
                0 if tid == self.pad_token_id else 1
                for tid in ids
            ]
            for ids in all_ids
        ]

        result: dict[str, Any] = {
            "input_ids": all_ids,
            "attention_mask": attention_masks,
        }

        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(
                all_ids, dtype=torch.long
            )
            result["attention_mask"] = torch.tensor(
                attention_masks, dtype=torch.long
            )

        return result

    def decode(
        self, token_ids: list[int]
    ) -> str:
        """Decode token IDs back to a string.

        Args:
            token_ids (list[int]): List of token IDs.

        Returns:
            str: Reconstructed DNA sequence.
        """
        if not token_ids:
            return ""

        chars: list[str] = []
        for i, tid in enumerate(token_ids):
            token = self._id_to_token.get(tid, "")
            if token.startswith("["):
                continue
            if i == len(token_ids) - 1:
                chars.append(token)
            else:
                chars.append(token[0])

        return "".join(chars)
