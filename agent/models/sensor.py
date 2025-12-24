import jax
import jax.numpy as jnp
import equinox as eqx
import distrax

from typing import Tuple, Union, Optional
from jaxtyping import Array, Float, PRNGKeyArray
from .utils import get_activation_fn


class Encoder(eqx.Module):
    body: eqx.nn.Sequential
    head: Union[eqx.nn.Linear, eqx.nn.Identity]
    shape: Tuple[int, int, int] = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    embedding_size: Optional[int] = eqx.field(static=True)

    def __init__(
            self,
            shape: Tuple[int, int, int],
            kernel_size: int = 4,
            depth: int = 48,
            stride: int = 2,
            embedding_size: Optional[int] = None,
            activation_function: str = "elu",
            *,
            key: jax.random.PRNGKey
    ):
        activation = get_activation_fn(activation_function)

        if embedding_size is not None:
            keys = jax.random.split(key, 5)
        else:
            keys = jax.random.split(key, 4)

        self.body = eqx.nn.Sequential([
            eqx.nn.Conv2d(shape[0], 1 * depth, kernel_size=kernel_size, stride=stride, key=keys[0]),
            eqx.nn.Lambda(activation),
            eqx.nn.Conv2d(1 * depth, 2 * depth, kernel_size=kernel_size, stride=stride, key=keys[1]),
            eqx.nn.Lambda(activation),
            eqx.nn.Conv2d(2 * depth, 4 * depth, kernel_size=kernel_size, stride=stride, key=keys[2]),
            eqx.nn.Lambda(activation),
            eqx.nn.Conv2d(4 * depth, 8 * depth, kernel_size=kernel_size, stride=stride, key=keys[3]),
            eqx.nn.Lambda(activation),
        ])

        feature_map_size = self.get_feature_map_size(shape)
        if embedding_size is not None:
            self.embedding_size = embedding_size
            self.head = eqx.nn.Linear(feature_map_size, embedding_size, key=keys[4])
        else:
            self.embedding_size = feature_map_size
            self.head = eqx.nn.Identity()

        self.shape = shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.stride = stride

    def __call__(
            self,
            input_tensor: Float[Array, "... input_dim"]
    ) -> Float[Array, "... output_dim"]:
        feature = self.body(obs)
        out = self.head(feature)
        return out

    def get_feature_map_size(self, shape) -> int:
        dummy_input = jnp.zeros(shape)
        out_shape_struct = jax.eval_shape(self.body, dummy_input)
        out_shape = jnp.array(out_shape_struct.shape)
        feature_map_size = int(jnp.prod(out_shape))
        return feature_map_size
