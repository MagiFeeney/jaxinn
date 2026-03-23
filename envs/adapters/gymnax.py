import math
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from functools import partial
from envs.environment import Transition, Environment, EnvInfo, process_obs, process_observation_space

import gymnax
from gymnax import EnvParams as GymnaxEnvParams
from gymnax.environments.spaces import Discrete, Box
from gymnax.environments.environment import Environment as GymnaxEnvironment
from gymnax.environments.environment import EnvState as GymnaxEnvState, EnvParams as GymnaxEnvParams


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
        """Performs step transitions in the environment."""
        if params is None:
            params = self.default_params

        # Step
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
        info['terminal_observation'] = obs_st
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

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, GymnaxEnvState]:
        obs, env_state = self.env.reset(key, self.env_params)
        obs = process_obs(obs)
        transition = Transition(
            action=jnp.zeros(self.action_space.shape, dtype=self.action_space.dtype),
            next_obs=obs,
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo(
            info={'discount': jnp.array(1.0)},
            reset=True,
        )
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: GymnaxEnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, GymnaxEnvState]:
        """Step the environment."""
        next_obs, next_env_state, reward, done, info = self.env.step(key, env_state, action, self.env_params)
        next_obs = process_obs(next_obs)
        if "terminal_observation" in info:
            info["terminal_observation"] = process_obs(info["terminal_observation"])
        transition = Transition(
            action=action,
            next_obs=next_obs,
            reward=reward,
            done=done,
        )
        env_info = EnvInfo(
            info=info,
            reset=False,
        )
        return transition, env_info, next_env_state

    @property
    def observation_space(self):
        space = self.env.observation_space(self.env_params)
        space = process_observation_space(space)
        return space

    @property
    def action_space(self):
        space = self.env.action_space(self.env_params)
        return space

    @property
    def action_size(self):
        if isinstance(self.action_space, Discrete):
            return self.action_space.n
        return math.prod(self.action_space.shape)
