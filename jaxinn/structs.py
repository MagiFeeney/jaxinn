import math
from typing import Union, Tuple, Any

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array, Bool, Float, PyTree, DTypeLike
import equinox as eqx

from jaxinn.agent.models.distributions import DistributionLike


class ArrayLikeOps:
    """Enables array-like behavior for a PyTree."""

    def __getitem__(self, index: Any):
        return jax.tree.map(lambda x: x[index], self)

    def __mul__(self, other):
        if isinstance(other, type(self)):
            return jax.tree.map(lambda x, y: x * y, self, other)
        return jax.tree.map(lambda x: x * other, self)

    def __rmul__(self, other):
        if isinstance(other, type(self)):
            return jax.tree.map(lambda x, y: x * y, other, self)
        return jax.tree.map(lambda x: other * x, self)

    def __add__(self, other):
        if isinstance(other, type(self)):
            return jax.tree.map(lambda x, y: x + y, self, other)
        return jax.tree.map(lambda x: x + other, self)

    def flatten(self):
        return jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), self)

    def narrow(self, axis: int, start: int, length: int):
        return jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, start, length, axis),
            self
        )

    def detach(self):
        return jax.lax.stop_gradient(self)


class Transition(eqx.Module, ArrayLikeOps):
    action: Float[Array, " action_dim"]
    next_obs: Float[Array, " obs_dim"]
    reward: Float[Array, ""]
    terminated: Bool[Array, ""]
    truncated: Bool[Array, ""]

    @classmethod
    def initialize(
            cls,
            capacity: Union[int, Tuple[int, ...]],
            obs_shape: PyTree[Tuple[int, ...]],
            obs_dtype: PyTree[DTypeLike],
            action_shape: PyTree[Tuple[int, ...]],
            action_dtype: PyTree[DTypeLike],
    ):
        capacity = (capacity,) if isinstance(capacity, int) else capacity
        action = jax.tree.map(
            lambda shape, dtype: jnp.zeros((*capacity, *shape), dtype=dtype),
            action_shape,
            action_dtype,
            is_leaf=lambda x: isinstance(x, tuple)
        )
        next_obs = jax.tree.map(
            lambda shape, dtype: jnp.zeros((*capacity, *shape), dtype=jnp.uint8 if len(shape) >= 3 else dtype),
            obs_shape,
            obs_dtype,
            is_leaf=lambda x: isinstance(x, tuple)
        )
        return cls(
            action=action,
            next_obs=next_obs,
            reward=jnp.zeros((*capacity, 1), dtype=jnp.float32),
            terminated=jnp.zeros((*capacity, 1), dtype=bool),
            truncated=jnp.zeros((*capacity, 1), dtype=bool),
        )


class Experience(eqx.Module):
    transition: Transition
    boundary_obs: jax.Array

    @classmethod
    def initialize(
            cls,
            capacity: Union[int, Tuple[int, ...]],
            obs_shape: PyTree[Tuple[int, ...]],
            obs_dtype: PyTree[DTypeLike],
            action_shape: PyTree[Tuple[int, ...]],
            action_dtype: PyTree[DTypeLike],
            needs_boundary_obs: bool,
    ):
        transition = Transition.initialize(capacity, obs_shape, obs_dtype, action_shape, action_dtype)
        boundary_obs = jnp.zeros_like(transition.next_obs) if needs_boundary_obs else None
        return cls(transition=transition, boundary_obs=boundary_obs)


class LatentState(eqx.Module, ArrayLikeOps):
    """
    Combine deterministic history encoding (belief) and the stochastic predictor (state) into a single state.
    """
    belief: jax.Array  # h_t
    state: jax.Array   # s_t

    @classmethod
    def initialize(
            cls,
            belief_size: int,
            state_size: Union[int, Tuple[int, ...]],
            random_init: bool = False,
            batch_shape: Tuple[int, ...] = (),
            *,
            key: PRNGKeyArray,
    ) -> "LatentState":
        key_belief, key_state = jax.random.split(key, 2)

        if isinstance(state_size, tuple):
            state_size = math.prod(state_size)

        mask = float(random_init)
        belief = jax.random.normal(key_belief, batch_shape + (belief_size,)) * mask
        state  = jax.random.normal(key_state,  batch_shape + (state_size,))  * mask

        return cls(belief=belief, state=state)

    @classmethod
    def concatenate(cls, states: list["LatentState"], axis: int = 0) -> "LatentState":
        return jax.tree.map(lambda *arrays: jnp.concatenate(arrays, axis=axis), *states)

    @property
    def shape(self):
        return self.feature.shape

    @property
    def batch_shape(self) -> tuple:
        return self.belief.shape[:-1]

    @property
    def feature(self) -> jax.Array:
        return jnp.concatenate([self.belief, self.state], axis=-1)


class LatentStateWithDist(eqx.Module):
    """
    Store the LatentState along with its dist
    """
    latent_state: LatentState
    fixed_dist: DistributionLike

    @property
    def dist(self):
        return self.fixed_dist.dist

    def __getattr__(self, name: str) -> Any:
        return getattr(self.fixed_dist, name)
