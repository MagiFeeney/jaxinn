import jax
from jaxtyping import PRNGKeyArray

from jaxinn.configs.model import WorldConfig

from ..base import Model
from ..perception import Encoder, Decoder
from .representation import Representation
from .transition import Transition
from .reward import Reward
from .continuation import Continuation


class World(Model):
    encoder: Encoder
    decoder: Decoder
    representation: Representation
    transition: Transition
    reward: Reward
    continuation: Continuation

    @classmethod
    def create(cls, config: WorldConfig, *, key: PRNGKeyArray):
        key_model, key_init = jax.random.split(key, 2)
        key_encoder, key_decoder, key_representation, key_transition, key_reward, key_continuation = jax.random.split(key_model, 6)

        encoder = Encoder.create(config.encoder, key=key_encoder)
        if config.decoder is not None:
            extra = {}
            if hasattr(encoder, "feature_map_shape"):
                extra["feature_map_shape"] = encoder.feature_map_shape
            decoder = Decoder.create(config.decoder, **extra, key=key_decoder)
        else:
            decoder = None

        if config.representation.embedding_size is None:
            config.representation.embedding_size = encoder.embedding_size

        representation = Representation.create(config.representation, key=key_representation)
        transition = Transition.create(config.transition, key=key_transition)
        reward = Reward.create(config.reward, key=key_reward)

        if config.continuation is not None:
            continuation = Continuation.create(config.continuation, key=key_continuation)
        else:
            continuation = None

        return cls(
            encoder=encoder,
            decoder=decoder,
            representation=representation,
            transition=transition,
            reward=reward,
            continuation=continuation
        ).apply_init(config.initializer, key=key_init)
