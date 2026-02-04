from typing import Any, Optional, Tuple, Dict

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from envs.environment import Transition, Environment, EnvInfo
from envs.spaces import Box

import brax
from brax.envs import Env as BraxEnv
from brax.envs.base import State as BraxEnvState


class Brax(Environment):
    def __init__(
            self,
            env: BraxEnv,
            env_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(env, env_params)

    @classmethod
    def create(cls, env_name: str, **kwargs) -> "Brax":
        env = brax.envs.get_environment(env_name, **kwargs)
        return cls(env, env_params=None)

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, BraxEnvState]:
        env_state = self.env.reset(key)
        transition = Transition(
            action=jnp.zeros(self.action_size),
            next_obs=env_state.obs,
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo(
            info={},            # TODO: fix mismatched pytree
            reset=True,
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
            reset=True,
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
