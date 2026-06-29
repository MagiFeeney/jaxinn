from typing import Any, Optional, Tuple as PyTuple

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import jaxarc
from jaxarc.envs.environment import Environment as JaxARCEnvironment
from jaxarc.types import (
    EnvParams as JaxARCEnvParams,
    TimeStep as JaxARCTimeStep
)
import stoa.spaces as stoa_spaces

from jaxinn.structs import Transition

from ..environment import Environment, EnvInfo
from ..spaces import Discrete, MultiDiscrete, Box, Dict, Tuple


def stoa_space_to_jaxinn_space(space):
    if isinstance(space, stoa_spaces.DiscreteSpace):
        return Discrete(n=space.num_values, dtype=space.dtype)
    elif isinstance(space, stoa_spaces.MultiDiscreteSpace):
        return MultiDiscrete(nvec=space.num_values, dtype=space.dtype)
    elif isinstance(space, stoa_spaces.BoundedArraySpace):
        return Box(
            low=space.minimum,
            high=space.maximum,
            shape=space.shape,
            dtype=space.dtype,
        )
    elif isinstance(space, stoa_spaces.DictSpace):
        converted_spaces = {k: stoa_space_to_jaxinn_space(v) for k, v in space.spaces.items()}
        return Dict(converted_spaces)
    elif isinstance(space, stoa_spaces.TupleSpace):
        converted_spaces = tuple(stoa_space_to_jaxinn_space(s) for s in space.spaces)
        return Tuple(converted_spaces)
    else:
        raise TypeError(
            f"Unsupported Stoa space type for conversion to Jaxinn space: '{type(space).__name__}'."
        )


class JaxARC(Environment):
    def __init__(
            self,
            env: JaxARCEnvironment,
            env_params: Optional[JaxARCEnvParams] = None,
    ):
        super().__init__(env, env_params)

    @classmethod
    def create(cls, env_name: str, **kwargs) -> "JaxARC":
        try:
            env, env_params = jaxarc.make(env_name, auto_download=True, **kwargs)
            return cls(env, env_params=env_params)
        except Exception as e:
            subset = env_name.split("-", maxsplit=1)[0]

            try:
                tasks = jaxarc.registration.available_task_ids(subset, auto_download=True)
            except Exception:
                tasks = "Could not fetch available tasks."

            raise ValueError(
                f"Failed to load env '{env_name}'. "
                f"If the name is invalid, available tasks for '{subset}' are: {tasks}"
            ) from e

    def reset(self, key: PRNGKeyArray) -> PyTuple[Transition, EnvInfo, JaxARCTimeStep]:
        env_state, timestep = self.env.reset(key, self.env_params)
        transition = Transition(
            action = jax.tree.map(
                lambda shape, dtype: jnp.zeros(shape, dtype=dtype),
                self.action_space.shape,
                self.action_space.dtype,
                is_leaf=lambda x: isinstance(x, tuple)
            ),
            next_obs=timestep.observation.astype(jnp.float32),
            reward=jnp.zeros(()),
            terminated=jnp.zeros((), dtype=bool),
            truncated=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo(
            info=timestep.extras,
            terminal_observation=jnp.zeros_like(transition.next_obs), # dummy
        )
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: JaxARCTimeStep, action: jax.Array) -> PyTuple[Transition, EnvInfo, JaxARCTimeStep]:
        next_env_state, timestep = self.env.step(env_state, action, self.env_params)
        transition = Transition(
            action=action,
            next_obs=timestep.observation.astype(jnp.float32),
            reward=timestep.reward,
            terminated=timestep.terminated(),
            truncated=timestep.truncated(),
        )
        env_info = EnvInfo(info=timestep.extras)
        return transition, env_info, next_env_state

    @property
    def observation_space(self):
        space = self.env.observation_space(self.env_params)
        space = stoa_space_to_jaxinn_space(space)
        return space

    @property
    def action_space(self):
        space = self.env.action_space(self.env_params)
        space = stoa_space_to_jaxinn_space(space)
        return space

    @property
    def max_episode_length(self) -> int:
        return self.max_episode_steps
