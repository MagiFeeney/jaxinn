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

        self.vmap_reset = jax.vmap(self.env.reset, in_axes=(0, None))
        self.vmap_step = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))

    def reset(self, key: PRNGKeyArray):
        keys = jax.random.split(key, self.num_envs)
        return self.vmap_reset(keys, self.env_params)

    def step(self, key: PRNGKeyArray, env_state: TEnvState, action: jax.Array):
        keys = jax.random.split(key, self.num_envs)
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
        return jnp.prod(jnp.array(self.action_space.shape))
