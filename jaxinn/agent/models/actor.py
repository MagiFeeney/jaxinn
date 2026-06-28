import math
from typing import Tuple, Union, Callable

import jax
from jaxtyping import PRNGKeyArray, PyTree
import equinox as eqx
import distrax

from jaxinn.structs import LatentState
from jaxinn.configs.model import ActorConfig
from jaxinn.configs.head import HeadConfig

from .utils import make_mlp
from .heads import Head


class Actor(eqx.Module):
    net: eqx.Module
    head: Head

    @classmethod
    def create(cls, config: ActorConfig, *, key: PRNGKeyArray):
        return cls(**config(), head=config.head, key=key)

    def __init__(
        self,
        belief_size: int,
        state_size: Union[int, Tuple[int, ...]],
        hidden_size: list[int],
        action_size: PyTree[int],
        head_config: HeadConfig,
        activation_function: Union[str, Callable] = "elu",
        *,
        key: PRNGKeyArray,
    ):
        self.head = Head.create(head_config, event_size=action_size)

        if isinstance(state_size, tuple):
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
        latent_state: Union[jax.Array, LatentState],
    ) -> distrax.Distribution:
        if isinstance(latent_state, LatentState):
            latent_state = latent_state.feature

        out = self.net(latent_state)

        return self.head(out)

    def sample(
            self,
            dist: distrax.Distribution,
            key: PRNGKeyArray,
            det: bool = False,      # default to training
    ) -> jax.Array:
        return self.head.sample(dist, key=key, det=det)
