from typing import Optional, Any, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, PRNGKeyArray
import equinox as eqx

from .cnn import CNNEncoder
from .linear import LinearEncoder
from ..utils import get_flatten_size, is_shape_leaf


class ActionEncoder(eqx.Module):
    action_embedding: Optional[Any]
    output_size: int = eqx.field(static=True)

    def __init__(
        self,
        action_shape: PyTree[Tuple[int, ...]],
        action_embedding_size: Optional[int] = None,
        *,
        key: PRNGKeyArray
    ):
        action_size = get_flatten_size(action_shape)

        if action_embedding_size and not is_shape_leaf(action_shape):
            flat_shapes, treedef = jax.tree.flatten(action_shape)
            num_leaves = len(flat_shapes)
            keys = jax.random.split(key, num_leaves)

            flat_embeddings = [
                LinearEncoder(
                    shape,
                    hidden_size=[], # Single layer embedding
                    embedding_size=action_embedding_size,
                    key=k
                ) if len(shape) <= 1 else CNNEncoder(
                    shape,
                    action_embedding_size,
                    key=k
                )
                for shape, k in zip(flat_shapes, keys)
            ]

            self.action_embedding = jax.tree.unflatten(treedef, flat_embeddings)
            self.output_size = num_leaves * action_embedding_size
        else:
            self.action_embedding = None
            self.output_size = action_size

    def __call__(self, action: PyTree[jax.Array]) -> jax.Array:
        if self.action_embedding is not None:
            action = jax.tree.map(
                lambda ebd, act: ebd(act),
                self.action_embedding,
                action
            )
        else:
            action = jax.tree.map(jnp.ravel, action)

        flat_action, _ = jax.tree.flatten(action)
        return jnp.concatenate(flat_action, axis=-1)
