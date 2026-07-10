from typing import Union, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx

from jaxinn.configs.head import HeadConfig
from jaxinn.configs.model import RepresentationConfig

from ..utils import make_mlp
from ..heads import Head
from ..distributions import DistributionLike


# Representation
class Representation(eqx.Module):
    """Representation learning of state, inferred from history and the latest observation: p(s_t | h_t, o_t)
    """
    net: eqx.Module
    head: Head

    @classmethod
    def create(cls, config: RepresentationConfig, *, key: PRNGKeyArray):
        return cls(**config(), head_config=config.head, key=key)

    def __init__(
            self,
            belief_size: int,
            embedding_size: int,
            state_size: Union[int, Tuple[int, ...]],
            hidden_size: list[int],
            head_config: HeadConfig,
            activation_function="elu",
            *,
            key: PRNGKeyArray,
    ):
        self.head = Head.create(head_config, event_size=state_size)

        self.net = make_mlp(
            input_size = belief_size + embedding_size,
            hidden_size = hidden_size,
            output_size = self.head.param_size,
            activation = activation_function,
            key = key
        )

    def __call__(
            self,
            belief: jax.Array,
            obs: jax.Array,
    ) -> Tuple[
        DistributionLike,
        jax.Array,
    ]:
        input_tensor = jnp.concatenate([belief, obs], axis=-1)
        out = self.net(input_tensor)

        return self.head(out), belief

    def sample(
            self,
            dist: DistributionLike,
            key: PRNGKeyArray,
    ) -> jax.Array:
        state = dist.sample(seed=key)
        return state
