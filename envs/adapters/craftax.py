import math
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from envs.environment import Transition, Environment, EnvInfo

from gymnax.environments.spaces import Discrete
from gymnax.environments.environment import Environment as GymnaxEnvironment

from craftax.craftax.craftax_state import EnvParams as CraftaxEnvParams, EnvState as CraftaxEnvState
from craftax.craftax_env import make_craftax_env_from_name


class Craftax(Environment):     # TODO: subclass gymnax instead
    def __init__(
            self,
            env: GymnaxEnvironment, # craftax is built on top of gymnax
            env_params: Optional[CraftaxEnvParams] = None,
    ):
        super().__init__(env, env_params)

    @classmethod
    def create(cls, env_name: str, **kwargs) -> "Craftax":
        env = make_craftax_env_from_name(env_name, **kwargs)
        env_params = env.default_params
        return cls(env, env_params)

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, CraftaxEnvState]:
        obs, env_state = self.env.reset(key, self.env_params)
        transition = Transition(
            action=jnp.zeros(self.action_space.shape),
            next_obs=obs,
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo(
            info={},            # TODO: fix mismatched pytree
            reset=True,
        )
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: CraftaxEnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, CraftaxEnvState]:
        """Step the environment."""
        next_obs, next_env_state, reward, done, info = self.env.step(key, env_state, action, self.env_params)
        transition = Transition(
            action=action,
            next_obs=next_obs,
            reward=reward,
            done=done,
        )
        env_info = EnvInfo(
            info=info,
            reset=True,
        )
        return transition, env_info, next_env_state

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
