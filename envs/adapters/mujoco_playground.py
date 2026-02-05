import numpy as np
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from envs.environment import Transition, Environment, EnvInfo
from envs.spaces import Discrete, Box

from mujoco_playground import registry
from mujoco.mjx import Model as MjxModel
from mujoco_playground import MjxEnv
from mujoco_playground import State as MjxState


class Playground(Environment):
    def __init__(
            self,
            env: MjxModel,
            env_params: Optional[Any] = None,
    ):
        super().__init__(env, env_params)

    @classmethod
    def create(cls, env_name: str, **kwargs) -> "Playground":
        env = registry.load(env_name, **kwargs)
        env_params = registry.get_default_config(env_name)
        return cls(env, env_params)

    def reset(self, key: PRNGKeyArray, env_params=None) -> Tuple[Transition, EnvInfo, MjxState]:
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
    def observation_space(self, env_params=None) -> Box:
        if isinstance(self.observation_size, dict):
            return Dict(
                spaces={
                    key: Box(
                        low=-jnp.inf,
                        high=jnp.inf,
                        shape=(shape,) if isinstance(shape, int) else shape,
                        dtype=jnp.float32
                    )
                    for key, shape in self.observation_size.items()
                }
            )
        else:
            shape = self.observation_size
            return Box(
                low=-jnp.inf,
                high=jnp.inf,
                shape=(shape,) if isinstance(shape, int) else shape,
                dtype=jnp.float32
            )

    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    @property
    def action_space(self, env_params=None) -> Box:
        return Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_size,),
            dtype=jnp.float32
        )

    @property
    def action_size(self) -> int:
        return self.env.action_size
