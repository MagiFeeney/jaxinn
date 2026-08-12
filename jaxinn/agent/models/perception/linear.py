import math
from typing import ClassVar
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx
from equinox._module import Static

from jaxinn.common.structs import LatentState
from jaxinn.configs.model import LinearEncoderConfig, LinearDecoderConfig

from .base import Encoder, Decoder
from ..utils import dx, make_mlp
from ..distributions import DistributionLike


class LinearEncoder(Encoder):
    config_cls: ClassVar[type] = LinearEncoderConfig

    net: eqx.Module

    def __init__(
            self,
            obs_shape: tuple[int, ...],
            hidden_size: list[int] | None = None,
            embedding_size: int | None = None,
            activation_function: str | Callable = "elu",
            *,
            key: PRNGKeyArray
    ):
        assert len(obs_shape) <= 1, (
            f"Expected a scalar or 1D observation shape in {self.__class__.__name__}, "
            f"but got {obs_shape} with {len(obs_shape)} dimensions."
        )

        if hidden_size is None:
            hidden_size = []

        if hidden_size and embedding_size is None:
            raise ValueError("Provided `hidden_size` but lacked `embedding_size`, making the final layer undecided.")

        if not hidden_size and embedding_size is None:
            self.net = eqx.nn.Identity()
        else:
            self.net = make_mlp(
                input_size=math.prod(obs_shape),
                hidden_size=hidden_size,
                output_size=embedding_size,
                activation=activation_function,
                key=key
            )

    def __call__(
            self,
            obs: jax.Array
    ) -> jax.Array:
        return self.net(jnp.atleast_1d(obs))


class LinearDecoder(Decoder):
    config_cls: ClassVar[type] = LinearDecoderConfig

    net: eqx.Module
    obs_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
            self,
            obs_shape: tuple[int, ...],
            belief_size: int,
            state_size: int | tuple[int, ...],
            hidden_size: list[int],
            activation_function: str | Callable = "elu",
            *,
            key: PRNGKeyArray
    ):
        assert len(obs_shape) <= 1, (
            f"Expected a scalar or 1D observation shape in {self.__class__.__name__}, "
            f"but got {obs_shape} with {len(obs_shape)} dimensions."
        )
        self.obs_shape = obs_shape

        if isinstance(state_size, tuple):
            state_size = math.prod(state_size)

        self.net = make_mlp(
            input_size=belief_size + state_size,
            hidden_size=hidden_size,
            output_size=math.prod(obs_shape),
            activation=activation_function,
            key=key
        )

    def __call__(
            self,
            latent_state: jax.Array | LatentState,
    ) -> DistributionLike:
        if isinstance(latent_state, LatentState):
            latent_state = latent_state.feature
        out = self.net(latent_state)
        out = out.reshape(out.shape[:-1] + self.obs_shape)
        dist = dx.Normal(out, jnp.ones_like(out))
        return dx.Independent(dist, reinterpreted_batch_ndims=Static(len(self.obs_shape)))
