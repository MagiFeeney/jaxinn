import jax
from jaxtyping import PRNGKeyArray
import equinox as eqx

from jaxinn.configs.model import WorldConfig

from ..perception import Encoder, Decoder
from .representation import Representation
from .transition import Transition
from .reward import Reward


class World(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    representation: Representation
    transition: Transition
    reward: Reward

    @classmethod
    def create(cls, config: WorldConfig, *, key: PRNGKeyArray):
        key_encoder, key_decoder, key_representation, key_transition, key_reward = jax.random.split(key, 5)

        encoder = Encoder.create(config.encoder, key=key_encoder)
        extra = {}
        if hasattr(encoder, "feature_map_shape"):
            extra["feature_map_shape"] = encoder.feature_map_shape
        decoder = Decoder.create(config.decoder, **extra, key=key_decoder)

        representation = Representation(**config.representation(), key=key_representation)
        transition = Transition(**config.transition(), key=key_transition)
        reward = Reward(**config.reward(), key=key_reward)

        return cls(
            encoder=encoder,
            decoder=decoder,
            representation=representation,
            transition=transition,
            reward=reward
        )
