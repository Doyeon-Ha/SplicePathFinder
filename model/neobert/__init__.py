"""Public API for the vendored NeoBERT backbone.

See model/neobert/README.md for vendor source, license, and patch notes.
``NeoBERTModel`` is an alias for the upstream ``NeoBERT`` class to match
the naming used in plan.md / data-model.md of feature 009.
"""

from model.neobert.modeling_neobert import (
    NeoBERT,
    NeoBERTConfig,
    NeoBERTLMHead,
)

NeoBERTModel = NeoBERT

__all__ = [
    "NeoBERT",
    "NeoBERTConfig",
    "NeoBERTLMHead",
    "NeoBERTModel",
]
