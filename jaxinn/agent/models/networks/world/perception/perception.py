import jax
from jaxtyping import PRNGKeyArray
import equinox as eqx

from jaxinn.configs.model import PerceptionConfig

from .base import Encoder, Decoder


class Perception(eqx.Module):
    encoder: Encoder
    decoder: Decoder

    @classmethod
    def create(cls, config: PerceptionConfig, *, key: PRNGKeyArray):
        key_encoder, key_decoder = jax.random.split(key, 2)
        encoder = Encoder.create(config.encoder, key=key_encoder)
        extra = {}
        if hasattr(encoder, "feature_map_shape"):
            extra["feature_map_shape"] = encoder.feature_map_shape
        decoder = Decoder.create(config.decoder, **extra, key=key_decoder)
        return cls(encoder=encoder, decoder=decoder)
