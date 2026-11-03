"""ByteTokenizer for Evo2 (StripedHyena2).

Wraps raw UTF-8 byte values into a vocab of size 512 (only [0, 255] are real
bytes; the upper range pads out to match the model's vocab embedding size).
Mirrors vortex.model.tokenizer.CharLevelTokenizer exactly:

  * eod / eos = byte 0 (chr(0))
  * pad       = byte 1 (chr(1))

No real special tokens are added at encoding time -- Evo2 is a pure byte-level
model and downstream pooling should treat every non-pad position as a real
token.
"""

from __future__ import annotations

from os import PathLike
from typing import List, Tuple

import numpy as np

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy
from transformers.utils.generic import PaddingStrategy


EMPTY: str = ""


class ByteTokenizer(PreTrainedTokenizer):
    """UTF-8 byte-level tokenizer for Evo2 (vocab_size = 512)."""

    def __init__(self, byte_level: bool = True, **kwargs):
        kwargs.setdefault("pad_token", chr(1))
        kwargs.setdefault("padding_side", "left")
        super().__init__(byte_level=byte_level, **kwargs)
        self.model_input_names = ["input_ids", "attention_mask"]

    @property
    def vocab_size(self) -> int:
        return 512

    @property
    def byte_level(self) -> bool:
        return self.init_kwargs.get("byte_level", True)

    def get_vocab(self) -> dict:
        return {chr(i): i for i in range(self.vocab_size)}

    def __len__(self) -> int:
        return self.vocab_size

    def clamp(self, n: int) -> int:
        # Matches vortex CharLevelTokenizer.clamp.
        return max(0, min(n, self.vocab_size - 1))

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text)

    def byte_tokenize(self, text: str) -> np.ndarray:
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)

    def _convert_token_to_id(self, token: str) -> int:
        return self.clamp(ord(token))

    def _convert_id_to_token(self, index: int) -> str:
        return chr(self.clamp(index))

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return EMPTY.join(tokens)

    def _decode(self, token_ids: List[int], **kwargs) -> str:
        indices = np.asarray(token_ids, dtype=np.int16)
        indices = indices.clip(min=0, max=255).astype(np.uint8)
        return indices.tobytes().decode("utf-8", errors="replace")

    def _encode_plus(self, text: str, **kwargs) -> BatchEncoding:
        first_ids = self.byte_tokenize(text).tolist()
        return self.prepare_for_model(
            first_ids,
            pair_ids=None,
            add_special_tokens=kwargs.get("add_special_tokens", False),
            padding=kwargs.get("padding_strategy", PaddingStrategy.DO_NOT_PAD).value,
            truncation=kwargs.get("truncation_strategy", TruncationStrategy.DO_NOT_TRUNCATE).value,
            max_length=kwargs.get("max_length"),
            stride=kwargs.get("stride", 0),
            pad_to_multiple_of=kwargs.get("pad_to_multiple_of"),
            return_tensors=kwargs.get("return_tensors"),
            prepend_batch_axis=True,
            return_attention_mask=kwargs.get("return_attention_mask"),
            return_token_type_ids=kwargs.get("return_token_type_ids"),
            return_overflowing_tokens=kwargs.get("return_overflowing_tokens", False),
            return_special_tokens_mask=kwargs.get("return_special_tokens_mask", False),
            return_length=kwargs.get("return_length", False),
            verbose=kwargs.get("verbose", True),
        )

    def _batch_encode_plus(self, batch_text_or_text_pairs, **kwargs) -> BatchEncoding:
        input_ids = [(self.byte_tokenize(t).tolist(), None) for t in batch_text_or_text_pairs]
        return self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=kwargs.get("add_special_tokens", False),
            padding_strategy=kwargs.get("padding_strategy", PaddingStrategy.DO_NOT_PAD),
            truncation_strategy=kwargs.get("truncation_strategy", TruncationStrategy.DO_NOT_TRUNCATE),
            max_length=kwargs.get("max_length"),
            stride=kwargs.get("stride", 0),
            pad_to_multiple_of=kwargs.get("pad_to_multiple_of"),
            return_attention_mask=kwargs.get("return_attention_mask"),
            return_token_type_ids=kwargs.get("return_token_type_ids"),
            return_overflowing_tokens=kwargs.get("return_overflowing_tokens", False),
            return_special_tokens_mask=kwargs.get("return_special_tokens_mask", False),
            return_length=kwargs.get("return_length", False),
            return_tensors=kwargs.get("return_tensors"),
            verbose=kwargs.get("verbose", True),
        )

    def _save_pretrained(
        self, save_directory: str | PathLike, file_names: Tuple[str], **kwargs
    ) -> Tuple[str]:
        return file_names
