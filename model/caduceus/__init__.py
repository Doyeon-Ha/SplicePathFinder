"""Public API for the vendored Caduceus backbone.

See model/caduceus/README.md for vendor source, license, and patch notes.
``Caduceus`` is the bare bidirectional backbone returning
``BaseModelOutputWithNoAttention``; ``CaduceusForMaskedLM`` adds the pretraining
head and is used only to verify checkpoint faithfulness.
"""

from model.caduceus.configuration_caduceus import CaduceusConfig
from model.caduceus.modeling_caduceus import (
    Caduceus,
    CaduceusForMaskedLM,
    CaduceusForSequenceClassification,
)

CaduceusModel = Caduceus

__all__ = [
    "Caduceus",
    "CaduceusConfig",
    "CaduceusForMaskedLM",
    "CaduceusForSequenceClassification",
    "CaduceusModel",
]
