"""Public API for the vendored Evo 2 backbone.

See model/evo2/README.md for vendor source, license, and patch notes. The code
is verbatim from upstream; vendoring exists only so the modeling file is
imported as a normal package instead of through ``trust_remote_code``, whose
static import check rejects the file over its optional ``flash_attn`` /
``transformer_engine`` imports.
"""

from model.evo2.configuration_evo2 import Evo2Config
from model.evo2.modeling_evo2 import Evo2ForCausalLM, Evo2Model
from model.evo2.tokenization_evo2 import ByteTokenizer

__all__ = ["ByteTokenizer", "Evo2Config", "Evo2ForCausalLM", "Evo2Model"]
