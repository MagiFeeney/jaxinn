from typing import Optional, Any, Tuple, ClassVar, Type

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, PRNGKeyArray
import equinox as eqx

from jaxinn.configs.base import ComplexConfig

from .base import Encoder, Decoder
from .cnn import CNNEncoder
from .linear import LinearEncoder
from ..distributions import TreeJointDistribution, HierarchicalJointDistribution
from ..utils import get_flatten_size, is_shape_leaf


class TreeEncoder(Encoder):
    encoders: PyTree[Encoder]

    @classmethod
    def create(cls, config: PyTree[EncoderConfig], *, key: PRNGKeyArray):
        treedef = jax.tree.structure(config, is_leaf=lambda x: isinstance(x, EncoderConfig))
        flat_keys = jax.random.split(key, treedef.num_leaves)
        keys = jax.tree.unflatten(treedef, flat_keys)
        encoders = jax.tree.map(
            lambda cfg, key: Encoder.create(cfg, key=key),
            config() if isinstance(config, ComplexConfig) else config,
            keys
        )
        embedding_sizes = jax.tree.map(lambda enc: enc.embedding_size, encoders)
        embedding_size = jax.tree.reduce(jnp.add, embedding_sizes)
        return cls(
            encoders=encoders,
            embedding_size=embedding_size,
        )

    @classmethod
    def build_flat(
            cls,
            shape: PyTree[Tuple[int, ...]],
            embedding_size: Optional[int] = None,
            *,
            key: PRNGKeyArray
    ):
        if embedding_size and not is_shape_leaf(shape):
            flat_shapes, treedef = jax.tree.flatten(shape)
            num_leaves = len(flat_shapes)
            keys = jax.random.split(key, num_leaves)

            flat_encoders = [
                LinearEncoder(
                    shape,
                    hidden_size=[], # Single layer embedding
                    embedding_size=embedding_size,
                    key=k
                ) if len(shape) <= 1 else CNNEncoder(
                    shape,
                    embedding_size,
                    key=k
                )
                for shape, k in zip(flat_shapes, keys)
            ]
            encoders = jax.tree.unflatten(treedef, flat_encoders)
            embedding_size = num_leaves * embedding_size
        else:
            encoders = None
            embedding_size = get_flatten_size(shape)

        return cls(
            encoders=encoders,
            embedding_size=embedding_size,
        )

    def __call__(self, tree_x: PyTree[jax.Array]) -> jax.Array:
        if self.encoders is not None:
            tree_embs = jax.tree.map(
                lambda enc, x: enc(x),
                self.encoders,
                tree_x,
                is_leaf=lambda enc: isinstance(enc, Encoder)
            )
        else:
            tree_embs = jax.tree.map(jnp.ravel, tree_x)

        flat_embs, _ = jax.tree.flatten(tree_embs)
        return jnp.concatenate(flat_embs, axis=-1)


class TreeDecoder(Decoder):
    decoders: PyTree[Decoder]
    treedef: PyTreeDef = eqx.field(static=True)
    param_size: int = eqx.field(static=True)
    split_points: Tuple[int, ...] = eqx.field(static=True)
    is_leaf: callable = eqx.field(static=True)

    dist_cls: ClassVar[Type[TreeJointDistribution]] = TreeJointDistribution

    @classmethod
    def create(cls, decoder_config: PyTree[DecoderConfig], **kwargs):
        event_size = kwargs.pop("event_size", None)

        if event_size is None:
            raise ValueError("event_size cannot be None for creating decoders.")

        decoders = jax.tree.map(
            lambda config, size: Decoder.create(config, event_size=size),
            decoder_config() if isinstance(decoder_config, ComplexDecoderConfig) else decoder_config,
            event_size
        )

        is_leaf = lambda x: isinstance(x, Decoder)

        param_size_tree = jax.tree.map(
            lambda h: h.param_size,
            decoders,
            is_leaf=is_leaf
        )
        flat_param_size, treedef = jax.tree.flatten(param_size_tree)
        split_points = tuple(np.cumsum(np.array(flat_param_size))[:-1].tolist())
        param_size = int(sum(flat_param_size))

        return cls(
            decoders=decoders,
            treedef=treedef,
            param_size=param_size,
            split_points=split_points,
            is_leaf=is_leaf,
        )

    def __call__(self, x: jax.Array) -> TreeJointDistribution:
        params = jnp.split(x, self.split_points, axis=-1)
        params_tree = jax.tree.unflatten(self.treedef, params)
        dists = jax.tree.map(
            lambda h, p: h(p),
            self.decoders,
            params_tree,
            is_leaf=self.is_leaf
        )
        return self.dist_cls(dists=dists)


class HierarchicalDecoder(TreeDecoder):
    dist_cls: ClassVar[Type[HierarchicalJointDistribution]] = HierarchicalJointDistribution
