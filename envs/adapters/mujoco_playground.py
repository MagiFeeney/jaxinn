import numpy as np
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from mujoco import mjx
from mujoco.mjx import Model as MjxModel
from mujoco_playground import MjxEnv
from mujoco_playground import State as MjxState

from envs.environment import Transition, Environment, EnvState, EnvInfo


class Playground(Environment):
    def __init__(
            self,
            env: MjxModel,
            env_params: Optional[Any] = None,
    ):
        super().__init__(env, env_params)

    def reset(self, key: PRNGKeyArray, env_params=None) -> Tuple[Transition, EnvInfo, MjxState]:
        env_state = self.env.reset(key)
        transition = Transition(
            action=jnp.zeros(self.action_size),
            next_obs=env_state.obs,
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo(
            info=env_state.info,
            metrics=env_state.metrics,
            reset=True,
        )
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: MjxState, action: jax.Array, env_params=None) -> Tuple[Transition, EnvInfo, MjxState]:
        """Step the environment."""
        next_env_state = self.env.step(env_state, action)
        transition = Transition(
            action=action,
            next_obs=next_env_state.obs,
            reward=next_env_state.reward,
            done=next_env_state.done,
        )
        env_info = EnvInfo(
            info=next_env_state.info,
            metrics=next_env_state.metrics,
            reset=False,
        )
        return transition, env_info, next_env_state

    def render(
            self,
            env_state: List[MjxState],
            height: int = 64,
            width: int: 64,
            camera: Optional[str] = None
    ) -> jax.Array:
        frames = self.env.render(env_state, height, width, camera) # returns a list of np.ndarray due to backbone renderer
        frames = np.stack(frames)
        return jnp.array(frames)

    @property
    def observation_space(self, env_params=None):
        raise NotImplementedError

    @property
    def reward_space(self, env_params=None):
        raise NotImplementedError

    @property
    def action_space(self, env_params=None):
        raise NotImplementedError

    @property
    def action_size(self):
        return self.env.action_size
