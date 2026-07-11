"""Model architecture: ConvNeXt-V2 encoder + Transformer decoder."""

from chant_omr.model.chant_omr_model import (
    ChantOMR,
    ChantOMRConfig,
    ParameterBreakdown,
    build_model,
    count_model_parameters,
)
from chant_omr.model.decoder import ChantDecoder, DecoderConfig, build_decoder, count_parameters
from chant_omr.model.encoder import ChantEncoder, EncoderOutput, build_encoder

__all__ = [
    "ChantDecoder",
    "ChantEncoder",
    "ChantOMR",
    "ChantOMRConfig",
    "DecoderConfig",
    "EncoderOutput",
    "ParameterBreakdown",
    "build_decoder",
    "build_encoder",
    "build_model",
    "count_model_parameters",
    "count_parameters",
]
