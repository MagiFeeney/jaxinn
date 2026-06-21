import math
from typing import Callable, Union, Dict, Tuple, Any

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array, Bool, Float
import equinox as eqx


class Transition(eqx.Module):
    action: Float[Array, " action_dim"]
    next_obs: Float[Array, " obs_dim"]
    reward: Float[Array, ""]
    terminated: Bool[Array, ""]
    truncated: Bool[Array, ""]

    @classmethod
    def initialize(
            cls,
            capacity: Union[int, Tuple[int, ...]],
            obs_shape: Tuple[int, ...],
            action_size: int,
    ):
        capacity = (capacity,) if isinstance(capacity, int) else capacity
        return cls(
            action=jnp.zeros((*capacity, action_size), dtype=jnp.float32),
            next_obs=jnp.zeros((*capacity, *obs_shape), dtype=jnp.uint8 if len(obs_shape) >= 3 else jnp.float32),
            reward=jnp.zeros((*capacity, 1), dtype=jnp.float32),
            terminated=jnp.zeros((*capacity, 1), dtype=bool),
            truncated=jnp.zeros((*capacity, 1), dtype=bool),
        )


class Experience(eqx.Module):
    transition: Transition
    terminal_observation: jax.Array

    @classmethod
    def initialize(
            cls,
            capacity: Union[int, Tuple[int, ...]],
            obs_shape: Tuple[int, ...],
            action_size: int,
    ):
        transition = Transition.initialize(capacity, obs_shape, action_size)
        terminal_observation = jnp.zeros_like(transition.next_obs)
        return cls(transition=transition, terminal_observation=terminal_observation)


class LatentState(eqx.Module):
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

    def __getitem__(self, index: Any) -> "LatentState":
        return jax.tree.map(lambda x: x[index], self)

    def __mul__(self, other):
        return jax.tree.map(lambda x: x * other, self)

    def __rmul__(self, other):
        return jax.tree.map(lambda x: other * x, self)

    def flatten(self) -> "LatentState":
        return jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), self)

    def narrow(self, axis: int, start: int, length: int) -> "LatentState":
        return jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, start, length, axis),
            self
        )

    def detach(self):
        return jax.tree.map(jax.lax.stop_gradient, self)


class LatentStateWithParams(eqx.Module):
    """
    Store the LatentState along with its parameters
    """
    latent_state: LatentState
    params: Dict[str, jax.Array]
    dist_cls: Callable[..., Any] = eqx.field(static=True)

    @property
    def dist(self):
        return self.dist_cls(**self.params).dist # FixedDistrax -> distrax.Distribution

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dist, name)
