"""Centred k-mer (N-gram) embedding for the dynamic-chunker model.

A DNA adaptation of the N-gram embedding of Qwen3.8-Next, itself a port of
DeepSeek Engram. Each position looks up one vector per k-mer size, for the
k-mer centred on it, from an exact table over the five symbols A, C, G, T and
N: ``5**k`` rows, no hashing and therefore no collisions. Positions past
either end of the sequence read as N. The concatenated vectors are projected
to a value, optionally gated by a key compared with the current hidden state:

    e_t   = concat_k E_k[kmer_k(t)]
    out_t = W_V e_t                                         (ungated)
    out_t = sigmoid(sign(g_t) * sqrt(|g_t|)) * W_V e_t      (gated)
    g_t   = <RMSNorm(W_K e_t), RMSNorm(h_t)> / sqrt(d)

The gate needs a hidden state that already carries context: on an input
embedding, ``h_t`` is the centre nucleotide alone and the gate is a fixed
value per k-mer. The caller adds ``out_t`` to its residual stream. Table rows
start at zero, so the module is an exact no-op at initialisation. Trained
without weight decay, a row whose k-mer never occurs keeps zero gradient and
stays zero.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Symbol order of ``IssaSplice._convert_tokens_to_onehot``: A, C, G, T, N.
NUM_SYMBOLS: int = 5
N_SYMBOL: int = 4


class NgramEmbedding(nn.Module):
    """Lookup of centred k-mer vectors from exact ``5**k`` tables.

    Attributes:
        kmer_sizes (tuple[int, ...]): Odd k-mer sizes, one table each.
        gated (bool): Whether the value is gated against a hidden state.
        tables (nn.ModuleList): ``nn.Embedding(5**k, table_dim)`` per size,
            zero-initialised.
        value_proj (nn.Linear): ``W_V``, concatenated vectors to ``hidden_dim``.
        key_proj (nn.Linear | None): ``W_K`` of the gate.
        key_norm (nn.RMSNorm | None): Normalises the key before the gate.
        query_norm (nn.RMSNorm | None): Normalises the hidden state before the
            gate.
    """

    def __init__(
        self,
        hidden_dim: int,
        kmer_sizes: tuple[int, ...] = (3, 7, 9),
        table_dim: int = 128,
        gated: bool = True,
    ) -> None:
        """Build the tables, the value projection and, if gated, the gate.

        Args:
            hidden_dim (int): Width ``d`` of the residual stream.
            kmer_sizes (tuple[int, ...]): Distinct odd k-mer sizes.
            table_dim (int): Width of each table row.
            gated (bool): Gate the value against the hidden state.

        Raises:
            ValueError: If ``kmer_sizes`` is empty, repeats a size, or holds an
                even or non-positive size (an even k-mer has no centre).
        """
        super().__init__()
        sizes = tuple(int(k) for k in kmer_sizes)
        if not sizes or len(set(sizes)) != len(sizes):
            raise ValueError(f"kmer_sizes must be distinct and non-empty; got {sizes}.")
        if any(k < 1 or k % 2 == 0 for k in sizes):
            raise ValueError(
                f"kmer_sizes must be odd so each k-mer is centred; got {sizes}."
            )
        self.kmer_sizes: tuple[int, ...] = sizes
        self.gated: bool = bool(gated)
        self.tables = nn.ModuleList(
            nn.Embedding(NUM_SYMBOLS**k, int(table_dim)) for k in sizes
        )
        for table in self.tables:
            nn.init.zeros_(table.weight)
        concat_dim = int(table_dim) * len(sizes)
        # Creation order (key, value, norms) fixes parameter order and the
        # initial draws, so gated checkpoints and optimizer states line up.
        self.key_proj: nn.Linear | None = (
            nn.Linear(concat_dim, int(hidden_dim), bias=False) if self.gated else None
        )
        self.value_proj = nn.Linear(concat_dim, int(hidden_dim), bias=False)
        self.key_norm: nn.RMSNorm | None = (
            nn.RMSNorm(int(hidden_dim)) if self.gated else None
        )
        self.query_norm: nn.RMSNorm | None = (
            nn.RMSNorm(int(hidden_dim)) if self.gated else None
        )
        self._scale: float = 1.0 / math.sqrt(int(hidden_dim))

    @staticmethod
    def kmer_ids(symbols: torch.Tensor, k: int) -> torch.Tensor:
        """Row index of the k-mer centred on each position.

        Args:
            symbols (torch.Tensor): ``[B, L]`` int64 symbols in ``0..4``
                (A, C, G, T, N).
            k (int): Odd k-mer size.

        Returns:
            torch.Tensor: ``[B, L]`` int64. The k-mer read as a base-5 number,
                leftmost symbol most significant; positions past either end
                count as N.
        """
        half = k // 2
        length = symbols.shape[1]
        padded = F.pad(symbols, (half, half), value=N_SYMBOL)
        ids = torch.zeros_like(symbols)
        for offset in range(k):
            ids = ids * NUM_SYMBOLS + padded[:, offset : offset + length]
        return ids

    def forward(
        self, symbols: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the k-mer value to add to the residual stream.

        Args:
            symbols (torch.Tensor): ``[B, L]`` int64 symbols in ``0..4``.
            hidden (torch.Tensor | None): ``[B, L, d]`` residual stream, the
                gate's query. Required when gated, unused otherwise.

        Returns:
            torch.Tensor: ``[B, L, d]``.

        Raises:
            ValueError: If the module is gated and ``hidden`` is None.
        """
        vectors = torch.cat(
            [
                table(self.kmer_ids(symbols, k))
                for k, table in zip(self.kmer_sizes, self.tables, strict=True)
            ],
            dim=-1,
        )
        value = self.value_proj(vectors)
        if not self.gated:
            return value
        if hidden is None:
            raise ValueError("a gated N-gram embedding needs the hidden state.")
        # The gate runs in fp32: under autocast the key projection returns
        # bf16, while the residual stream it is compared with stays fp32.
        key = self.key_norm(self.key_proj(vectors).float())
        query = self.query_norm(hidden.float())
        score = (key * query).sum(dim=-1, keepdim=True) * self._scale
        # Signed square root as in Engram; the clamp keeps the gradient finite
        # at a zero score.
        gate = torch.sigmoid(score.abs().clamp_min(1e-6).sqrt() * score.sign())
        return gate * value
