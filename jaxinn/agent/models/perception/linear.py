import math
from typing import Optional, Callable, Union, Tuple, ClassVar, Type

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx
from equinox._module import Static

from jaxinn.structs import LatentState
from jaxinn.configs.model import LinearEncoderConfig, LinearDecoderConfig

from .base import Encoder, Decoder
from ..utils import dx, make_mlp
from ..distributions import DistributionLike


class LinearEncoder(Encoder):
    config_cls: ClassVar[Type] = LinearEncoderConfig

    net: eqx.Module

    def __init__(
            self,
            shape: Tuple[int, ...],
            hidden_size: Optional[list[int]] = None,
            embedding_size: Optional[int] = None,
            activation_function: Union[str, Callable] = "elu",
            *,
            key: PRNGKeyArray
    ):
        assert len(shape) <= 1, (
            f"Expected a scalar or 1D observation shape in {self.__class__.__name__}, "
            f"but got {shape} with {len(shape)} dimensions."
        )

        if hidden_size is not None and \
           embedding_size is not None:
            self.net = make_mlp(
                input_size=math.prod(shape),
                hidden_size=hidden_size,
                output_size=embedding_size,
                activation=activation_function,
                key=key
            )
        else:
            self.net = eqx.nn.Identity()

    def __call__(
            self,
            obs: jax.Array
    ) -> jax.Array:
        return self.net(jnp.atleast_1d(obs))


class LinearDecoder(Decoder):
    config_cls: ClassVar[Type] = LinearDecoderConfig

    net: eqx.Module
    shape: Tuple[int, ...] = eqx.field(static=True)

    def __init__(
            self,
            shape: Tuple[int, ...],
            belief_size: int,
            state_size: Union[int, Tuple[int, ...]],
            hidden_size: list[int],
            activation_function: Union[str, Callable] = "elu",
            *,
            key: PRNGKeyArray
    ):
        assert len(shape) <= 1, (
            f"Expected a scalar or 1D observation shape in {self.__class__.__name__}, "
            f"but got {shape} with {len(shape)} dimensions."
        )
        self.shape = shape

        if isinstance(state_size, tuple):
            state_size = math.prod(state_size)

        self.net = make_mlp(
            input_size=belief_size + state_size,
            hidden_size=hidden_size,
            output_size=math.prod(shape),
            activation=activation_function,
            key=key
        )

    def __call__(
            self,
            latent_state: Union[jax.Array, LatentState],
    ) -> DistributionLike:
        if isinstance(latent_state, LatentState):
            latent_state = latent_state.feature
        out = self.net(latent_state)
        out = out.reshape(out.shape[:-1] + self.shape)
        dist = dx.Normal(out, jnp.ones_like(out))
        return dx.Independent(dist, reinterpreted_batch_ndims=Static(len(self.shape)))
