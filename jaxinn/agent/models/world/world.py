import jax
from jaxtyping import PRNGKeyArray
import equinox as eqx

from jaxinn.configs.model import WorldConfig
from jaxinn.configs.base import ComplexConfig, HierarchicalConfig

from ..perception import TreeEncoder, TreeDecoder
from .representation import Representation
from .transition import Transition
from .reward import Reward


class World(eqx.Module):
    encoder: TreeEncoder
    decoder: TreeDecoder
    representation: Representation
    transition: Transition
    reward: Reward

    @classmethod
    def create(cls, config: WorldConfig, *, key: PRNGKeyArray):
        key_encoder, key_decoder, key_representation, key_transition, key_reward = jax.random.split(key, 5)

        if not isinstance(config.encoder, ComplexConfig):
            encoder = Encoder.create(config.encoder, key=key_encoder)
        elif isinstance(config.encoder, HierarchicalConfig):
            encoder = HierarchicalEncoder.create(config.encoder, key=key_encoder)
        else:
            encoder = TreeEncoder.create(config.encoder, key=key_encoder)

        # TODO: tree map over extra here or getting feature_map_shape inside TreeEncoder
        extra = {}
        if hasattr(encoder, "feature_map_shape"):
            extra["feature_map_shape"] = encoder.feature_map_shape
        decoder = TreeDecoder.create(config.decoder, **extra, key=key_decoder)

        if not isinstance(config.encoder, ComplexConfig):
            encoder = Encoder.create(config.encoder, **extra, key=key_encoder)
        elif isinstance(config.encoder, HierarchicalConfig):
            encoder = HierarchicalDecoder.create(config.encoder, extra, key=key_encoder)
        else:
            encoder = TreeDecoder.create(config.encoder, extra, key=key_encoder)

        config.representation.embedding_size = encoder.embedding_size
        representation = Representation.create(config.representation, key=key_representation)
        transition = Transition.create(config.transition, key=key_transition)
        reward = Reward.create(config.reward, key=key_reward)

        return cls(
            encoder=encoder,
            decoder=decoder,
            representation=representation,
            transition=transition,
            reward=reward
        )
