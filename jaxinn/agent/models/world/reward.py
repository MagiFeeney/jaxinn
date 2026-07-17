import math
from typing import Optional, Union, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, PRNGKeyArray
import equinox as eqx

from jaxinn.structs import LatentState
from jaxinn.configs.head import HeadConfig
from jaxinn.configs.model import RewardConfig

from ..utils import make_mlp
from ..perception import TreeEncoder
from ..heads import Head
from ..distributions import DistributionLike


class Reward(eqx.Module):
    net: eqx.Module
    head: Head
    action_encoder: Optional[TreeEncoder]

    @classmethod
    def create(cls, config: RewardConfig, *, key: PRNGKeyArray):
        return cls(**config(), head_config=config.head, key=key)

    def __init__(
            self,
            belief_size: int,
            state_size: Union[int, Tuple[int, ...]],
            hidden_size: list[int],
            head_config: HeadConfig,
            activation_function="elu",
            action_shape: Optional[PyTree[Tuple[int, ...]]] = None,
            action_embedding_size: Optional[int] = None,
            *,
            key: PRNGKeyArray,
    ):  # if action_size is not None, Q fn
        self.head = Head.create(head_config, event_size=1)

        if isinstance(state_size, tuple):
            state_size = math.prod(state_size)

        if action_shape is not None:
            key, key_encoder = jax.random.split(key, 2)
            self.action_encoder = TreeEncoder.build_flat(action_shape, action_embedding_size, key=key_encoder)
            encoded_action_size = self.action_encoder.embedding_size
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
        latent_state: Union[jax.Array, LatentState],
        action: Optional[jax.Array] = None,
    ) -> DistributionLike:
        if isinstance(latent_state, LatentState):
            latent_state = latent_state.feature

        assert (action is None) == (self.action_encoder is None)

        if action is not None:
            encoded_action = self.action_encoder(action)
            latent_state = jnp.concatenate([latent_state, encoded_action], axis=-1)

        out = self.net(latent_state)

        return self.head(out)
