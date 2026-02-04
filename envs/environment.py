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
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{item}'")


class EnvState(eqx.Module):
    pass


class Transition(eqx.Module):
    action: Float[Array, " action_dim"]
    next_obs: Float[Array, " obs_dim"]
    reward: Float[Array, ""]
    done: Bool[Array, ""]


class Environment(eqx.Module):
    env: Any = eqx.field(static=True)
    env_params: Any = eqx.field(static=True) # TODO: fix Playground env_params

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
