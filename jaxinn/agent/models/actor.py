import math
from collections.abc import Callable

import jax
from jaxtyping import PRNGKeyArray, PyTree
import equinox as eqx

from jaxinn.common.structs import LatentState
from jaxinn.configs.model import ActorConfig, PerceptionActorConfig
from jaxinn.configs.head import (
    HeadConfig,
    ComplexHeadConfig,
    HierarchicalHeadConfig
)

from .base import Model
from .heads import Head, TreeHead, HierarchicalHead
from .distributions import DistributionLike
from .perception import Encoder
from .utils import make_mlp


class Actor(Model):
    net: eqx.Module
    head: TreeHead

    @classmethod
    def create(cls, config: ActorConfig, *, key: PRNGKeyArray):
        key_model, key_init = jax.random.split(key, 2)
        return cls(**config(), head_config=config.head, key=key_model).apply_init(config.initializer, key=key_init)

    def __init__(
        self,
        belief_size: int,
        state_size: int | tuple[int, ...],
        hidden_size: list[int],
        action_size: PyTree[int],
        head_config: PyTree[HeadConfig],
        activation_function: str | Callable = "elu",
        *,
        key: PRNGKeyArray,
    ):
        if not isinstance(head_config, ComplexHeadConfig):
            self.head = Head.create(head_config, event_size=action_size)
        elif isinstance(head_config, HierarchicalHeadConfig):
            self.head = HierarchicalHead.create(head_config, event_size=action_size)
        else:
            self.head = TreeHead.create(head_config, event_size=action_size)

        if isinstance(state_size, tuple): # TODO: fix state_size when it is Categorical
            state_size = math.prod(state_size)

        # Build network
        self.net = make_mlp(
            input_size = belief_size + state_size,
            hidden_size = hidden_size,
            output_size = self.head.param_size,
            activation = activation_function,
            key = key
        )

    def __call__(
        self,
        latent_state: jax.Array | LatentState,
    ) -> DistributionLike:
        if isinstance(latent_state, LatentState):
            latent_state = latent_state.feature

        out = self.net(latent_state)

        return self.head(out)


class PerceptionActor(Model):
    encoder: Encoder
    actor: Actor

    @classmethod
    def create(cls, config: PerceptionActorConfig, *, key: PRNGKeyArray):
        key_encoder, key_actor, key_init = jax.random.split(key, 3)

        encoder = Encoder.create(config.encoder, key=key_encoder)
        actor = Actor.create(
            config.actor,
            key=key_actor
        )

        return cls(encoder=encoder, actor=actor).apply_init(config.initializer, key=key_init)

    def __call__(self, obs: jax.Array) -> DistributionLike:
        feature = self.encoder(obs)
        return self.actor(feature)
