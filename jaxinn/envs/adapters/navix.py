from typing import Any, Optional, Tuple, Dict

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import navix
from navix.environments import Environment as NavixEnvironment
from navix.environments import Timestep as NavixTimestep

from jaxinn.structs import Transition

from ..environment import Environment, EnvInfo
from ..spaces import Box, Discrete


class Navix(Environment):
    def __init__(
            self,
            env: NavixEnvironment,
            env_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(env, env_params)

    @classmethod
    def create(cls, env_name: str, **kwargs) -> "Navix":
        env = navix.make("Navix-" + env_name, **kwargs)
        return cls(env, env_params=kwargs)

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, NavixTimestep]:
        env_state = self.env.reset(key)
        transition = Transition(
            action=jnp.zeros(self.action_space.shape, dtype=self.action_space.dtype),
            next_obs=env_state.observation.astype(jnp.float32),
            reward=jnp.zeros(()),
            terminated=jnp.zeros((), dtype=bool),
            truncated=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo(
            info=env_state.info,
            terminal_observation=jnp.zeros_like(transition.next_obs), # dummy
        )
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: NavixTimestep, action: jax.Array) -> Tuple[Transition, EnvInfo, NavixTimestep]:
        next_env_state = self.env.step(env_state, action)
        transition = Transition(
            action=action,
            next_obs=next_env_state.observation.astype(jnp.float32),
            reward=next_env_state.reward,
            terminated=next_env_state.is_termination(),
            truncated=next_env_state.is_truncation(),
        )
        env_info = EnvInfo(info=next_env_state.info)
        return transition, env_info, next_env_state

    @property
    def observation_space(self):
        navix_obs_space = self.env.observation_space
        return Box(
            low=navix_obs_space.minimum,
            high=navix_obs_space.maximum,
            shape=navix_obs_space.shape,
            dtype=jnp.float32,
        )

    @property
    def action_space(self):
        return Discrete(self.action_size)

    @property
    def observation_size(self):
        return self.env.observation_size

    @property
    def action_size(self):
        return len(self.env.action_set)

    @property
    def max_episode_length(self) -> int:
        return self.max_steps
