import abc
from typing import Any, Dict as PyDict, Tuple as PyTuple, TypeVar, Generic, Sequence, Union

import jax
import jax.numpy as jnp


T = TypeVar("T")


class Space(abc.ABC, Generic[T]):
    """
    Abstract base class for JAX-based spaces.
    Registered as a PyTree to allow passing into JIT-compiled functions.
    """

    def __init__(self, shape: PyTuple[int, ...], dtype: Any):
        self._validate_dtype(dtype)
        self.shape = shape
        self.dtype = dtype

    def _validate_dtype(self, dtype):
        """Raises ValueError if an abstract dtype is provided."""
        if dtype is None:
            return

        abstract_integers = (int, jnp.integer, jnp.signedinteger, jnp.unsignedinteger)
        abstract_floats   = (float, jnp.floating)

        if dtype in abstract_integers:
            raise ValueError(
                f"Abstract dtype '{dtype}' is not allowed. "
                f"Please use a concrete dtype like jnp.int32, jnp.int64, or jnp.uint8."
            )

        if dtype in abstract_floats:
            raise ValueError(
                f"Abstract dtype '{dtype}' is not allowed. "
                f"Please use a concrete dtype like jnp.float32."
            )

    @abc.abstractmethod
    def sample(self, key: jax.Array) -> T:
        """Sample a value from the space using a PRNG key."""
        pass

    @abc.abstractmethod
    def contains(self, x: T) -> bool:
        """Check if x is a valid member of this space."""
        pass

    def tree_flatten(self):
        return (), (self.shape, self.dtype)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


# Register the base class (and generic subclasses)
jax.tree_util.register_pytree_node(Space, Space.tree_flatten, Space.tree_unflatten)


@jax.tree_util.register_pytree_node_class
class Discrete(Space[jax.Array]):
    """
    A discrete space in {0, ..., n-1}.
    """
    def __init__(self, n: int, dtype=jnp.int32):
        super().__init__(shape=(), dtype=dtype)
        self.n = n

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.random.randint(key, shape=self.shape, minval=0, maxval=self.n, dtype=self.dtype)

    def contains(self, x: jax.Array) -> bool:
        is_int = jnp.issubdtype(x.dtype, jnp.integer)
        in_bounds = (x >= 0) & (x < self.n)
        return is_int & in_bounds & (x.shape == self.shape)

    def tree_flatten(self):
        return (), (self.n, self.dtype)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)


@jax.tree_util.register_pytree_node_class
class Box(Space[jax.Array]):
    """
    A continuous space in [low, high].
    """
    def __init__(self, low: Union[float, jax.Array], high: Union[float, jax.Array], shape: PyTuple[int, ...], dtype=jnp.float32):
        super().__init__(shape, dtype)
        self.low = jnp.broadcast_to(jnp.asarray(low, dtype=dtype), shape)
        self.high = jnp.broadcast_to(jnp.asarray(high, dtype=dtype), shape)

    @property
    def is_bounded(self) -> jax.Array:
        return jnp.logical_and(
            jnp.all(jnp.isfinite(self.low)),
            jnp.all(jnp.isfinite(self.high))
        )

    def sample(self, key: jax.Array) -> jax.Array:
        def bounded_sample(key):
            if jnp.issubdtype(self.dtype, jnp.integer):
                 return jax.random.randint(
                     key,
                     shape=self.shape,
                     minval=self.low,
                     maxval=self.high + 1,
                     dtype=self.dtype
                 )
            return jax.random.uniform(
                key,
                shape=self.shape,
                minval=self.low,
                maxval=self.high,
                dtype=self.dtype
            )

        def unbounded_sample(key):
            return jax.random.truncated_normal(
                key,
                shape=self.shape,
                lower=self.low,
                upper=self.high,
                dtype=self.dtype if not jnp.issubdtype(self.dtype, jnp.integer) else jnp.float32
            )

        return jax.lax.cond(
            self.is_bounded,
            bounded_sample,
            unbounded_sample,
            key
        )

    def contains(self, x: jax.Array) -> bool:
        in_bounds = (x >= self.low) & (x <= self.high)
        type_match = (x.dtype == self.dtype)
        shape_match = (x.shape == self.shape)
        return jnp.all(in_bounds) & type_match & shape_match

    def tree_flatten(self):
        return (self.low, self.high), (self.shape, self.dtype)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        low, high = children
        shape, dtype = aux_data
        return cls(low, high, shape, dtype)


@jax.tree_util.register_pytree_node_class
class MultiDiscrete(Space[jax.Array]):
    """
    Multiple discrete spaces with different number of categories per dimension.
    """
    def __init__(self, nvec: Sequence[int], dtype=jnp.int32):
        self.nvec = jnp.asarray(nvec, dtype=dtype)
        super().__init__(shape=self.nvec.shape, dtype=dtype)

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.random.randint(key, shape=self.shape, minval=0, maxval=self.nvec, dtype=self.dtype)

    def contains(self, x: jax.Array) -> bool:
        is_int = jnp.issubdtype(x.dtype, jnp.integer)
        in_bounds = (x >= 0) & (x < self.nvec)
        return is_int & jnp.all(in_bounds) & (x.shape == self.shape)

    def tree_flatten(self):
        return (self.nvec,), (self.dtype,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        nvec = children[0]
        dtype = aux_data[0]
        return cls(nvec, dtype)


@jax.tree_util.register_pytree_node_class
class Dict(Space[PyDict[str, Any]]):
    """
    A dictionary of spaces.
    """
    def __init__(self, spaces: PyDict[str, Space]):
        self.spaces = spaces
        super().__init__(shape=(), dtype=None)

    def sample(self, key: jax.Array) -> PyDict[str, Any]:
        keys = jax.random.split(key, len(self.spaces))
        return {
            k: space.sample(k_key)
            for (k, space), k_key in zip(self.spaces.items(), keys)
        }

    def contains(self, x: PyDict[str, Any]) -> bool:
        if not isinstance(x, dict) or len(x) != len(self.spaces):
            return False
        return all(k in x and self.spaces[k].contains(x[k]) for k in self.spaces)

    def tree_flatten(self):
        sorted_keys = sorted(self.spaces.keys())
        children = [self.spaces[k] for k in sorted_keys]
        return children, sorted_keys

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        keys = aux_data
        spaces = dict(zip(keys, children))
        return cls(spaces)
