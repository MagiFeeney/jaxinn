from typing import Any, Optional, Tuple as PyTuple
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import gymnax
import gymnax.environments.spaces as gymnax_spaces
from gymnax.environments.environment import Environment as GymnaxEnvironment
from gymnax.environments.environment import EnvState as GymnaxEnvState, EnvParams as GymnaxEnvParams

from jaxinn.structs import Transition

from ..environment import Environment, EnvInfo
from ..spaces import Discrete, Box, Dict, Tuple


def gymnax_space_to_jaxinn_space(space):
    if isinstance(space, gymnax_spaces.Discrete):
        return Discrete(n=space.n, dtype=space.dtype)
    elif isinstance(space, gymnax_spaces.Box):
        return Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=space.dtype,
        )
    elif isinstance(space, gymnax_spaces.Dict):
        converted_spaces = {k: gymnax_space_to_jaxinn_space(v) for k, v in space.spaces.items()}
        return Dict(converted_spaces)
    elif isinstance(space, gymnax_spaces.Tuple):
        converted_spaces = tuple(gymnax_space_to_jaxinn_space(s) for s in space.spaces)
        return Tuple(converted_spaces)
    else:
        raise TypeError(
            f"Unsupported Gymnax space type for conversion to Jaxinn space: '{type(space).__name__}'."
        )


class TerminalObservationWrapper:
    def __init__(self, env):
        self.env = env

    @partial(jax.jit, static_argnames=("self",))
    def step(
        self,
        key: jax.Array,
        state: GymnaxEnvState,
        action: int | float | jax.Array,
        params: GymnaxEnvParams | None = None,
    ) -> tuple[jax.Array, GymnaxEnvState, jax.Array, jax.Array, dict[Any, Any]]:
        if params is None:
            params = self.default_params

        key_step, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key_step, state, action, params
        )
        obs_re, state_re = self.reset_env(key_reset, params)

        # Auto-reset environment based on termination
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)

        # Get terminal obs
        info['boundary_obs'] = obs_st
        return obs, state, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class Gymnax(Environment):
    def __init__(
            self,
            env: GymnaxEnvironment,
            env_params: Optional[GymnaxEnvParams] = None,
    ):
        super().__init__(env, env_params)

    @classmethod
    def create(cls, env_name: str, **kwargs) -> "Gymnax":
        env, env_params = gymnax.make(env_name, **kwargs)
        env = TerminalObservationWrapper(env)
        return cls(env, env_params)

    def reset(self, key: PRNGKeyArray) -> PyTuple[Transition, EnvInfo, GymnaxEnvState]:
        obs, env_state = self.env.reset(key, self.env_params)
        transition = Transition(
            action=jnp.zeros(self.action_space.shape, dtype=self.action_space.dtype),
            next_obs=obs,
            reward=jnp.zeros(()),
            terminated=jnp.zeros((), dtype=bool),
            truncated=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo(boundary_obs=jnp.zeros_like(obs)) # dummy
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: GymnaxEnvState, action: jax.Array) -> PyTuple[Transition, EnvInfo, GymnaxEnvState]:
        next_obs, next_env_state, reward, done, info = self.env.step(key, env_state, action, self.env_params)
        truncated = next_env_state.time >= self.max_episode_length
        # Infer termination from the done and truncation flags since gymnax couples the two.
        # Note: Ambiguity remains when truncation is True. However, the probability of
        # false positives (i.e., actual termination coinciding exactly with truncation) is low.
        terminated = jnp.logical_and(done, jnp.logical_not(truncated))
        transition = Transition(
            action=action,
            next_obs=next_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        env_info = EnvInfo(info=info)
        return transition, env_info, next_env_state

    @property
    def observation_space(self):
        space = self.env.observation_space(self.env_params)
        space = gymnax_space_to_jaxinn_space(space)
        return space

    @property
    def action_space(self):
        space = self.env.action_space(self.env_params)
        space = gymnax_space_to_jaxinn_space(space)
        return space

    @property
    def max_episode_length(self) -> int:
        return self.max_steps_in_episode
