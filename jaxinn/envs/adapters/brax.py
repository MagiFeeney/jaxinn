from typing import Any, Optional, Tuple, Dict

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import brax
from brax.envs import Env as BraxEnv
from brax.envs.base import State as BraxEnvState

from jaxinn.structs import Transition
from envs.environment import Environment, EnvInfo
from envs.spaces import Box


class Brax(Environment):
    def __init__(
            self,
            env: BraxEnv,
            env_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(env, env_params)

    @classmethod
    def create(cls, env_name: str, **kwargs) -> "Brax":
        env = brax.envs.get_environment(env_name.lower(), **kwargs)
        return cls(env, env_params=kwargs)

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, BraxEnvState]:
        env_state = self.env.reset(key)
        transition = Transition(
            action=jnp.zeros(self.action_space.shape, dtype=self.action_space.dtype),
            next_obs=env_state.obs,
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo(
            info=env_state.info,
            metrics=env_state.metrics,
            terminal_observation=jnp.zeros_like(transition.next_obs), # dummy
        )
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: BraxEnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, BraxEnvState]:
        next_env_state = self.env.step(env_state, action)
        transition = Transition(
            action=action,
            next_obs=next_env_state.obs,
            reward=next_env_state.reward,
            done=next_env_state.done.astype(bool),
        )
        env_info = EnvInfo(
            info=next_env_state.info,
            metrics=next_env_state.metrics,
        )
        return transition, env_info, next_env_state

    @property
    def observation_space(self):
        return Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.observation_size,)
        )

    @property
    def action_space(self):
        return Box(
            low=-1,
            high=1,
            shape=(self.action_size,)
        )

    @property
    def observation_size(self):
        return self.env.observation_size

    @property
    def action_size(self):
        return self.env.action_size
