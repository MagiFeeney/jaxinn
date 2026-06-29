import math
from typing import Callable, Union, Tuple, ClassVar, Type

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx
from equinox._module import Static
import distrax

from jaxinn.structs import LatentState
from jaxinn.configs.model import CNNEncoderConfig, CNNDecoderConfig

from .base import Encoder, Decoder
from ..utils import get_activation_fn, get_precision_fn, dx, StaticCallable


# Perception
class CNNEncoder(Encoder):
    config_cls: ClassVar[Type] = CNNEncoderConfig

    body: eqx.nn.Sequential
    head: Union[eqx.nn.Linear, eqx.nn.Identity]
    obs_shape: Tuple[int, int, ...] = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    embedding_size: int = eqx.field(static=True)
    dtype: str = eqx.field(static=True)

    feature_map_shape: Tuple[int, int, int] = eqx.field(static=True)

    def __init__(               # TODO: add num_layers
            self,
            obs_shape: Tuple[int, int, ...],
            embedding_size: int,
            kernel_size: int = 4,
            depth: int = 32,
            stride: int = 2,
            activation_function: Union[str, Callable] = "elu",
            dtype: str = "float32",
            *,
            key: PRNGKeyArray
    ):
        assert len(obs_shape) >= 2, (
            f"Expected a 2D+ observation shape in {self.__class__.__name__}, "
            f"but got {obs_shape} with {len(obs_shape)} dimensions."
        )
        activation = get_activation_fn(activation_function)
        self.dtype = get_precision_fn(dtype)

        keys = jax.random.split(key, 5)

        self.body = eqx.nn.Sequential([
            eqx.nn.Conv2d(obs_shape[0], 1 * depth, kernel_size=kernel_size, stride=stride, padding="SAME", key=keys[0], dtype=self.dtype),
            StaticCallable(activation),
            eqx.nn.Conv2d(1 * depth, 2 * depth, kernel_size=kernel_size, stride=stride, padding="SAME", key=keys[1], dtype=self.dtype),
            StaticCallable(activation),
            eqx.nn.Conv2d(2 * depth, 4 * depth, kernel_size=kernel_size, stride=stride, padding="SAME", key=keys[2], dtype=self.dtype),
            StaticCallable(activation),
            eqx.nn.Conv2d(4 * depth, 8 * depth, kernel_size=kernel_size, stride=stride, padding="SAME", key=keys[3], dtype=self.dtype),
            StaticCallable(activation),
            StaticCallable(jnp.ravel),
        ])

        self.feature_map_shape = self.get_feature_map_shape(obs_shape)
        feature_map_size = math.prod(self.feature_map_shape)
        self.head = eqx.nn.Linear(feature_map_size, embedding_size, key=keys[4])

        self.obs_shape = obs_shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.stride = stride
        self.embedding_size = embedding_size

    def __call__(
            self,
            obs: jax.Array
    ) -> jax.Array:
        obs = jnp.atleast_3d(obs)
        obs = obs.astype(self.dtype)
        feature = eqx.filter_checkpoint(self.body)(obs)
        feature = feature.astype(jnp.float32) # upcast for stability
        out = self.head(feature)
        return out

    def get_feature_map_shape(self, shape) -> int:
        dummy_input = jnp.zeros(shape, dtype=self.dtype)
        out = jax.eval_shape(self.body[:-1], dummy_input) # exclude ravel
        return out.shape


class CNNDecoder(Decoder):
    config_cls: ClassVar[Type] = CNNDecoderConfig

    embedding: eqx.nn.Linear
    body: eqx.nn.Sequential
    obs_shape: Tuple[int, int, ...] = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    dtype: str = eqx.field(static=True)

    feature_map_shape: Tuple[int, int, int] = eqx.field(static=True)

    def __init__(
            self,
            belief_size: int,
            state_size: Union[int, Tuple[int, ...]],
            obs_shape: Tuple[int, int, ...],
            feature_map_shape: Tuple[int, int, int],
            kernel_size: int = 4,
            depth: int = 32,
            stride: int = 2,
            activation_function: Union[str, Callable] = "elu",
            dtype: str = "float32",
            *,
            key: PRNGKeyArray
    ):
        assert len(obs_shape) >= 2, (
            f"Expected a 2D+ observation shape in {self.__class__.__name__}, "
            f"but got {obs_shape} with {len(obs_shape)} dimensions."
        )
        activation = get_activation_fn(activation_function)
        self.dtype = get_precision_fn(dtype)
        self.feature_map_shape = feature_map_shape
        feature_map_size = math.prod(feature_map_shape)

        keys = jax.random.split(key, 5)

        if isinstance(state_size, tuple):
            state_size = math.prod(state_size)

        out_channels = obs_shape[0] if len(obs_shape) > 2 else 1

        self.embedding = eqx.nn.Linear(belief_size + state_size, feature_map_size, key=keys[0])
        self.body = eqx.nn.Sequential([ # TODO: support 4D spatial dims and fix this line
            StaticCallable(activation),
            eqx.nn.ConvTranspose2d(feature_map_shape[0], 4 * depth, kernel_size=kernel_size, stride=stride, padding="SAME", key=keys[1], dtype=self.dtype),
            StaticCallable(activation),
            eqx.nn.ConvTranspose2d(4 * depth, 2 * depth, kernel_size=kernel_size, stride=stride, padding="SAME", key=keys[2], dtype=self.dtype),
            StaticCallable(activation),
            eqx.nn.ConvTranspose2d(2 * depth, 1 * depth, kernel_size=kernel_size, stride=stride, padding="SAME", key=keys[3], dtype=self.dtype),
            StaticCallable(activation),
            eqx.nn.ConvTranspose2d(1 * depth, out_channels, kernel_size=kernel_size, stride=stride, padding="SAME", key=keys[4], dtype=self.dtype),
        ])

        self.obs_shape = obs_shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.stride = stride

    def __call__(
            self,
            latent_state: Union[jax.Array, LatentState],
    ) -> distrax.Distribution:
        if isinstance(latent_state, LatentState):
            latent_state = latent_state.feature
        embedding = self.embedding(latent_state)
        embedding = embedding.reshape(embedding.shape[:-1] + self.feature_map_shape).astype(self.dtype)
        out = eqx.filter_checkpoint(self.body)(embedding)
        out = out.astype(jnp.float32)

        out = out.reshape(out.shape[:-3] + self.obs_shape) # TODO: support 4D spatial dims and fix this line

        if out.shape[-3:] != self.obs_shape: # TODO: support 4D spatial dims and fix this line
            out = jax.image.resize(out, shape=out.shape[:-3] + self.obs_shape, method="bilinear")

        dist = dx.Normal(out, jnp.ones_like(out))
        return dx.Independent(dist, reinterpreted_batch_ndims=Static(len(self.obs_shape)))
