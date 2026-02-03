import math
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from gymnax import EnvParams as GymnaxEnvParams
from gymnax.environments.spaces import Discrete
from gymnax.environments.environment import Environment as GymnaxEnvironment
from gymnax.environments.environment import TEnvState as GymnaxEnvState

from envs.environment import Transition, Environment, EnvState, EnvInfo


class Gymnax(Environment):
    def __init__(
            self,
            env: GymnaxEnvironment,
            env_params: Optional[GymnaxEnvParams] = None,
    ):
        super().__init__(env, env_params)

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, GymnaxEnvState]:
        obs, env_state = self.env.reset(key, self.env_params)
        transition = Transition(
            action=jnp.zeros(self.action_size),
            next_obs=obs,
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo(
            info={},            # TODO: fix mismatched pytree
            reset=True,
        )
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: GymnaxEnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, GymnaxEnvState]:
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

    def reward_space(self):
        raise NotImplementedError
