import math
from typing import ClassVar
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx
from equinox._module import Static

from jaxinn.common.structs import LatentState
from jaxinn.configs.model import CNNEncoderConfig, CNNDecoderConfig

from .base import Encoder, Decoder
from ..utils import get_activation_fn, get_precision_fn, dx, StaticCallable, make_cnn, make_cnn_transposed
from ..distributions import DistributionLike


# Perception
class CNNEncoder(Encoder):
    config_cls: ClassVar[type] = CNNEncoderConfig

    body: eqx.nn.Sequential
    head: eqx.nn.Linear | eqx.nn.Identity
    embedding_size: int = eqx.field(static=True)
    dtype: str = eqx.field(static=True)

    feature_map_shape: tuple[int, int, int] = eqx.field(static=True)

    def __init__(
            self,
            obs_shape: tuple[int, int, ...],
            embedding_size: int | None = None,
            num_layers: int | None = 4,
            kernel_size: int | Sequence[int] = 4,
            depth: int | Sequence[int] = 32,
            stride: int | Sequence[int] = 2,
            padding: str | int | Sequence[int] | Sequence[tuple[int, int]] = "VALID",
            activation_function: str | Callable = "elu",
            dtype: str = "float32",
            *,
            key: PRNGKeyArray
    ):
        assert len(obs_shape) >= 2, (
            f"Expected a 2D+ observation shape in {self.__class__.__name__}, "
            f"but got {obs_shape} with {len(obs_shape)} dimensions."
        )
        self.dtype = get_precision_fn(dtype)

        key_body, key_head = jax.random.split(key, 2)

        self.body = make_cnn(
            in_channels=obs_shape[0],
            num_spatial_dims=len(obs_shape) - 1,
            activation=activation_function,
            kernel_size=kernel_size,
            depth=depth,
            depth_factor=2,
            stride=stride,
            padding=padding,
            dtype=self.dtype,
            num_layers=num_layers,
            key=key_body
        )

        self.feature_map_shape = self.get_feature_map_shape(obs_shape)
        feature_map_size = math.prod(self.feature_map_shape)

        if embedding_size is not None:
            self.head = eqx.nn.Linear(feature_map_size, embedding_size, key=key_head)
            self.embedding_size = embedding_size
        else:
            self.head = eqx.nn.Identity()
            self.embedding_size = feature_map_size


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
    config_cls: ClassVar[type] = CNNDecoderConfig

    embedding: eqx.nn.Sequential
    body: eqx.nn.Sequential
    obs_shape: tuple[int, int, ...] = eqx.field(static=True)
    event_ndim: int = eqx.field(static=True)
    dtype: str = eqx.field(static=True)

    feature_map_shape: tuple[int, int, int] = eqx.field(static=True)
    flatten_embedding: bool = eqx.field(static=True)

    def __init__(
            self,
            belief_size: int,
            state_size: int | tuple[int, ...],
            obs_shape: tuple[int, int, ...],
            feature_map_shape: tuple[int, ...],
            flatten_embedding: bool = True,
            num_layers: int | None = 4,
            kernel_size: int | Sequence[int] = 4,
            depth: int | Sequence[int] = 32,
            stride: int | Sequence[int] = 2,
            padding: str | int | Sequence[int] | Sequence[tuple[int, int]] = "VALID",
            activation_function: str | Callable = "elu",
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

        key_embedding, key_body = jax.random.split(key, 2)

        if isinstance(state_size, tuple):
            state_size = math.prod(state_size)

        self.embedding = eqx.nn.Sequential([
            eqx.nn.Linear(belief_size + state_size, feature_map_size, key=key_embedding),
            StaticCallable(activation),
        ])

        if flatten_embedding:
            in_channels = feature_map_size
        else:
            in_channels = feature_map_shape[0]

        self.body = make_cnn_transposed(
            in_channels=in_channels,
            out_channels=obs_shape[0],
            num_spatial_dims=len(obs_shape) - 1,
            activation=activation,
            kernel_size=kernel_size,
            depth=depth,
            depth_factor=2,
            stride=stride,
            padding=padding,
            dtype=self.dtype,
            num_layers=num_layers,
            key=key_body
        )

        self.obs_shape = obs_shape
        self.event_ndim = len(obs_shape)
        self.flatten_embedding = flatten_embedding

    def __call__(
            self,
            latent_state: jax.Array | LatentState,
    ) -> DistributionLike:
        if isinstance(latent_state, LatentState):
            latent_state = latent_state.feature
        embedding = self.embedding(latent_state)

        if self.flatten_embedding:
            embedding = embedding[..., None, None].astype(self.dtype)
        else:
            embedding = embedding.reshape(embedding.shape[:-1] + self.feature_map_shape).astype(self.dtype)

        out = eqx.filter_checkpoint(self.body)(embedding)
        out = out.astype(jnp.float32)

        if out.shape[-self.event_ndim:] != self.obs_shape:
            out = jax.image.resize(out, shape=out.shape[:-self.event_ndim] + self.obs_shape, method="bilinear")

        dist = dx.Normal(out, jnp.ones_like(out))
        return dx.Independent(dist, reinterpreted_batch_ndims=Static(self.event_ndim))
