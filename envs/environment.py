import abc
import jax
import jax.numpy as jnp
from typing import Tuple, Any
from jaxtyping import PRNGKeyArray, Array, Bool, Float
import equinox as eqx


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

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, key: PRNGKeyArray, env_state: EnvState, action: jax.Array) -> Tuple[Transition, Any]:
        pass

    @property
    @abc.abstractmethod
    def observation_space(self):
        pass

    @property
    @abc.abstractmethod
    def reward_space(self):
        pass

    @property
    @abc.abstractmethod
    def action_space(self):
        pass


def create():
    pass
