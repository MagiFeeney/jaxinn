import abc
import jax
import jax.numpy as jnp
from typing import Tuple, Any, Dict
from jaxtyping import PRNGKeyArray, Array, Bool, Float
import equinox as eqx


class EnvInfo(eqx.Module):
    data: Dict[str, Any]

    def __init__(self, **kwargs):
        object.__setattr__(self, "data", kwargs)

    def __getattr__(self, item):
        # Check top level first
        if item in self.data:
            return self.data[item]

        # Look inside info otherwise
        info = self.data.get("info")
        if info is not None and isinstance(info, dict) and item in info:
            return info[item]

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{item}'")


class EnvState(eqx.Module):
    pass


class Transition(eqx.Module):
    action: Float[Array, " action_dim"]
    next_obs: Float[Array, " obs_dim"]
    reward: Float[Array, ""]
    done: Bool[Array, ""]


def process_obs(obs: jax.Array) -> jax.Array:
    """Process observation from NHWC to NCHW for the image input while turning the 2D input as gray image."""
    if obs.ndim == 3 and obs.shape[-1] in (1, 3, 4, 6):
        obs = jnp.moveaxis(obs, source=-1, destination=-3)
    elif obs.ndim == 2:
        obs = obs[None, ...]    # Convert to gray image

    return obs


def process_observation_space(space: Any) -> Any:
    """Align with and reflect the processed observation."""
    if hasattr(space, 'spaces') and isinstance(space.spaces, dict):
        new_spaces = {k: process_observation_space(v) for k, v in space.spaces.items()}
        return type(space)(new_spaces)

    if hasattr(space, 'shape') and hasattr(space, 'low') and hasattr(space, 'high'):
        if len(space.shape) == 3 and space.shape[-1] in (1, 3, 4, 6):
            new_shape = (space.shape[2], space.shape[0], space.shape[1])
            low = jnp.moveaxis(space.low, -1, -3) if getattr(space.low, 'ndim', 0) == 3 else space.low
            high = jnp.moveaxis(space.high, -1, -3) if getattr(space.high, 'ndim', 0) == 3 else space.high
            return type(space)(low=low, high=high, shape=new_shape, dtype=space.dtype)
        elif len(space.shape) == 2:
            new_shape = (1, space.shape[0], space.shape[1])
            low = space.low[None, ...] if getattr(space.low, 'ndim', 0) == 2 else space.low
            high = space.high[None, ...] if getattr(space.high, 'ndim', 0) == 2 else space.high
            return type(space)(low=low, high=high, shape=new_shape, dtype=space.dtype)
    return space


class Environment(eqx.Module):
    env: Any = eqx.field(static=True)
    env_params: Any = eqx.field(static=True)

    @abc.abstractmethod
    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, EnvState]:
        pass

    @abc.abstractmethod
    def step(self, key: PRNGKeyArray, env_state: EnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, EnvState]:
        pass

    @property
    @abc.abstractmethod
    def observation_space(self):
        pass

    @property
    @abc.abstractmethod
    def action_space(self):
        pass

    @property
    @abc.abstractmethod
    def action_size(self):
        pass

    @property
    def is_action_space_discrete(self) -> bool:
        return type(self.action_space).__name__ == "Discrete"
