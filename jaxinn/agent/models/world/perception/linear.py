import math
from typing import Optional, Callable, Union, Tuple, ClassVar, Type

import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import equinox as eqx
from equinox._module import Static
import distrax

from jaxinn.configs import LinearEncoderConfig, LinearDecoderConfig
from jaxinn.agent.models.utils import get_activation_fn, dx, StaticCallable, make_mlp

from .base import Encoder, Decoder
from ..primitives import LatentState


class LinearEncoder(Encoder):
    config_cls: ClassVar[Type] = LinearEncoderConfig

    net: eqx.Module

    def __init__(
            self,
            shape: Tuple[int, ...],
            hidden_size: Optional[int] = None,
            embedding_size: Optional[int] = None,
            num_layers: Optional[int] = None,
            activation_function: Union[str, Callable] = "elu",
            *,
            key: PRNGKeyArray
    ):
        assert len(shape) == 1, (
            f"Expected a 1D shape, but got shape {shape} with {len(shape)} dimensions "
            f"in {self.__class__.__name__}."
        )
        activation = get_activation_fn(activation_function)

        if hidden_size is not None and \
           embedding_size is not None and \
           num_layers is not None:
            self.net = make_mlp(
                input_size = shape[0],
                hidden_size = hidden_size,
                output_size = embedding_size,
                num_layers = num_layers,
                activation = StaticCallable(activation),
                key = key
            )
        else:
            self.net = eqx.nn.Identity()

    def __call__(
            self,
            obs: Float[Array, "... obs_size"]
    ) -> Float[Array, "... output_size"]:
        return self.net(obs)


class LinearDecoder(Decoder):
    config_cls: ClassVar[Type] = LinearDecoderConfig

    net: eqx.Module

    def __init__(
            self,
            shape: Tuple[int, ...],
            belief_size: int,
            state_size: Union[int, Tuple[int, ...]],
            hidden_size: int,
            num_layers: int,
            activation_function: Union[str, Callable] = "elu",
            *,
            key: PRNGKeyArray
    ):
        assert len(shape) == 1, (
            f"Expected a 1D shape, but got shape {shape} with {len(shape)} dimensions "
            f"in {self.__class__.__name__}."
        )
        activation = get_activation_fn(activation_function)

        if isinstance(state_size, tuple):
            state_size = math.prod(state_size)

        self.net = make_mlp(
            input_size = belief_size + state_size,
            hidden_size = hidden_size,
            output_size = shape[0],
            num_layers = num_layers,
            activation = StaticCallable(activation),
            key = key
        )

    def __call__(
            self,
            latent_state: Union[Float[Array, "... input_size"], LatentState],
    ) -> distrax.Distribution:
        if isinstance(latent_state, LatentState):
            latent_state = latent_state.feature
        out = self.net(latent_state)
        dist = dx.Normal(out, jnp.ones_like(out))
        return dx.Independent(dist, reinterpreted_batch_ndims=Static(1))
