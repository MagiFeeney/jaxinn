import math
from typing import Callable, Union, Tuple, ClassVar, Type, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx
from equinox._module import Static

from jaxinn.structs import LatentState
from jaxinn.configs.model import CNNEncoderConfig, CNNDecoderConfig

from .base import Encoder, Decoder
from ..utils import get_activation_fn, get_precision_fn, dx, StaticCallable, make_cnn, make_cnn_transposed
from ..distributions import DistributionLike


# Perception
class CNNEncoder(Encoder):
    config_cls: ClassVar[Type] = CNNEncoderConfig

    body: eqx.nn.Sequential
    head: Union[eqx.nn.Linear, eqx.nn.Identity]
    embedding_size: int = eqx.field(static=True)
    dtype: str = eqx.field(static=True)

    feature_map_shape: Tuple[int, int, int] = eqx.field(static=True)

    def __init__(
            self,
            shape: Tuple[int, int, ...],
            embedding_size: int,
            num_layers: int | None = 4,
            kernel_size: int | Sequence[int] = 4,
            depth: int | Sequence[int] = 32,
            stride: int | Sequence[int] = 2,
            activation_function: Union[str, Callable] = "elu",
            dtype: str = "float32",
            *,
            key: PRNGKeyArray
    ):
        assert len(shape) >= 2, (
            f"Expected a 2D+ observation shape in {self.__class__.__name__}, "
            f"but got {shape} with {len(shape)} dimensions."
        )
        self.dtype = get_precision_fn(dtype)

        key_body, key_head = jax.random.split(key, 2)

        self.body = make_cnn(
            in_channels=shape[0],
            num_spatial_dims=len(shape) - 1,
            activation=activation_function,
            kernel_size=kernel_size,
            depth=depth,
            depth_factor=2,
            stride=stride,
            padding="SAME",
            dtype=self.dtype,
            num_layers=num_layers,
            key=key_body
        )

        self.feature_map_shape = self.get_feature_map_shape(shape)
        feature_map_size = math.prod(self.feature_map_shape)
        self.head = eqx.nn.Linear(feature_map_size, embedding_size, key=key_head)

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

    embedding: eqx.nn.Sequential
    body: eqx.nn.Sequential
    shape: Tuple[int, int, ...] = eqx.field(static=True)
    event_ndim: int = eqx.field(static=True)
    dtype: str = eqx.field(static=True)

    feature_map_shape: Tuple[int, int, int] = eqx.field(static=True)

    def __init__(
            self,
            belief_size: int,
            state_size: Union[int, Tuple[int, ...]],
            shape: Tuple[int, int, ...],
            feature_map_shape: Tuple[int, ...],
            num_layers: int | None = 4,
            kernel_size: int | Sequence[int] = 4,
            depth: int | Sequence[int] = 32,
            stride: int | Sequence[int] = 2,
            activation_function: Union[str, Callable] = "elu",
            dtype: str = "float32",
            *,
            key: PRNGKeyArray
    ):
        assert len(shape) >= 2, (
            f"Expected a 2D+ observation shape in {self.__class__.__name__}, "
            f"but got {shape} with {len(shape)} dimensions."
        )
        activation = get_activation_fn(activation_function)
        self.dtype = get_precision_fn(dtype)
        self.feature_map_shape = feature_map_shape
        feature_map_size = math.prod(feature_map_shape)

        key_embedding, key_body = jax.random.split(key, 2)

        if isinstance(state_size, tuple):
            state_size = math.prod(state_size)

        self.embedding = eqx.nn.Sequential([
            eqx.nn.Linear(belief_size + state_size, feature_map_size, key=key_embedding),
            StaticCallable(activation),
        ])

        self.body = make_cnn_transposed(
            in_channels=feature_map_shape[0],
            out_channels=shape[0],
            num_spatial_dims=len(shape) - 1,
            activation=activation,
            kernel_size=kernel_size,
            depth=depth,
            depth_factor=2,
            stride=stride,
            padding="SAME",
            dtype=self.dtype,
            num_layers=num_layers,
            key=key_body
        )

        self.shape = shape
        self.event_ndim = len(shape)

    def __call__(
            self,
            latent_state: Union[jax.Array, LatentState],
    ) -> DistributionLike:
        if isinstance(latent_state, LatentState):
            latent_state = latent_state.feature
        embedding = self.embedding(latent_state)
        embedding = embedding.reshape(embedding.shape[:-1] + self.feature_map_shape).astype(self.dtype)
        out = eqx.filter_checkpoint(self.body)(embedding)
        out = out.astype(jnp.float32)

        if out.shape[-self.event_ndim:] != self.shape:
            out = jax.image.resize(out, shape=out.shape[:-self.event_ndim] + self.shape, method="bilinear")

        dist = dx.Normal(out, jnp.ones_like(out))
        return dx.Independent(dist, reinterpreted_batch_ndims=Static(self.event_ndim))
