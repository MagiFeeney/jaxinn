import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, PRNGKeyArray
import equinox as eqx

from jaxinn.common.structs import LatentState
from jaxinn.configs.model import CriticConfig, PerceptionCriticConfig
from jaxinn.configs.head import HeadConfig

from .base import Model
from .perception import ActionEncoder, Encoder
from .heads import Head
from .distributions import DistributionLike
from .utils import make_mlp


class Critic(Model):
    action_encoder: ActionEncoder | None
    net: eqx.Module
    head: Head

    @classmethod
    def create(cls, config: CriticConfig, *, key: PRNGKeyArray):
        key_model, key_init = jax.random.split(key, 2)
        return cls(**config(), head_config=config.head, key=key_model).apply_init(config.initializer, key=key_init)

    def __init__(
            self,
            belief_size: int,
            state_size: int | tuple[int, ...],
            hidden_size: list[int],
            head_config: HeadConfig,
            activation_function: str | Callable = "elu",
            action_shape: PyTree[tuple[int, ...]] | None = None,
            action_embedding_size: int | None = None,
            *,
            key: PRNGKeyArray,
    ):  # if action_shape is not None, use Q fn
        self.head = Head.create(head_config, event_size=1)

        if isinstance(state_size, tuple):
            state_size = math.prod(state_size)

        if action_shape is not None:
            key, key_encoder = jax.random.split(key, 2)
            self.action_encoder = ActionEncoder(action_shape, action_embedding_size, key=key_encoder)
            encoded_action_size = self.action_encoder.output_size
        else:
            self.action_encoder = None
            encoded_action_size = 0

        input_size = belief_size + state_size + encoded_action_size

        self.net = make_mlp(
            input_size = input_size,
            hidden_size = hidden_size,
            output_size = self.head.param_size,
            activation = activation_function,
            key = key
        )

    def __call__(
        self,
        latent_state: jax.Array | LatentState,
        action: jax.Array | None = None,
    ) -> DistributionLike:
        if isinstance(latent_state, LatentState):
            latent_state = latent_state.feature

        assert (action is None) == (self.action_encoder is None)

        if action is not None:
            encoded_action = self.action_encoder(action)
            latent_state = jnp.concatenate([latent_state, encoded_action], axis=-1)

        out = self.net(latent_state)

        return self.head(out)


class PerceptionCritic(Model):
    encoder: Encoder
    critic: Critic

    @classmethod
    def create(cls, config: PerceptionCriticConfig, *, key: PRNGKeyArray):
        key_model, key_init = jax.random.split(key, 2)
        key_encoder, key_critic = jax.random.split(key_model, 2)

        encoder = Encoder.create(config.encoder, key=key_encoder)
        critic = Critic.create(
            config.critic,
            key=key_critic
        )

        return cls(encoder=encoder, critic=critic).apply_init(config.initializer, key=key_init)

    def __call__(self, obs: jax.Array, action: jax.Array | None = None) -> DistributionLike:
        feature = self.encoder(obs)
        return self.critic(feature, action)
