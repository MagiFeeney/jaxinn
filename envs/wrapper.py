import math
import jax
import jax.numpy as jnp
from typing import Any, Callable
from jaxtyping import PRNGKeyArray
import equinox as eqx

from gymnax.environments.spaces import Discrete
from gymnax.environments.environment import Environment, TEnvParams, TEnvState


class Batched(eqx.Module):
    env: Environment = eqx.field(static=True)
    env_params: Any = eqx.field(static=True)
    num_envs: int = eqx.field(static=True)

    vmap_reset: Callable = eqx.field(static=True)
    vmap_step: Callable = eqx.field(static=True)

    def __init__(self, env: Any, env_params: TEnvParams, num_envs: int = 1):
        self.env = env
        self.env_params = env_params
        self.num_envs = num_envs # Default value; may be changed if given as input during reset

        self.vmap_reset = jax.vmap(self.env.reset, in_axes=(0, None))
        self.vmap_step = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))

    def reset(self, key: PRNGKeyArray, num_envs: int | None = None):
        num_keys = num_envs if num_envs is not None else self.num_envs
        keys = jax.random.split(key, num_keys)
        return self.vmap_reset(keys, self.env_params)

    def step(self, key: PRNGKeyArray, env_state: TEnvState, action: jax.Array):
        num_keys = action.shape[0] # Infer from action since we have already known that before step
        keys = jax.random.split(key, num_keys)
        return self.vmap_step(keys, env_state, action, self.env_params)

    @property
    def observation_space(self):
        return self.env.observation_space(self.env_params)

    @property
    def action_space(self):
        return self.env.action_space(self.env_params)

    @property
    def action_size(self):
        if isinstance(self.action_space, Discrete):
            return self.action_space.n
        return math.prod(self.action_space.shape)
