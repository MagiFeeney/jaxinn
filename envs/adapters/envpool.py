import math
from typing import Any, Callable, Optional, Tuple, Dict

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from envs.environment import Transition, Environment, EnvInfo

import envpool
from envpool.python.env_pool import EnvPoolMixin


class EnvPool(Environment):
    def __init__(
            self,
            env: EnvPoolMixin,
            env_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(env, env_params)
        handle, recv, send, step = env.xla()
        self._handle: jax.Array = handle
        self._recv: Callable = recv
        self._send: Callable = send
        self._step: Callable = step

    @classmethod
    def create(cls, env_name: str, **kwargs) -> "EnvPool":
        env = envpool.make(env_name, env_type="gymnasium", **kwargs)
        return cls(env, env_params=None)

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, jax.Array]:
        env_state, (obs, _, _, _, info) = self._recv(self._handle)
        transition = Transition(
            action=jnp.zeros(self.action_space.shape, dtype=self.action_space.dtype),
            next_obs=obs,
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo(
            info=info,            # TODO: fix mismatched pytree
            reset=True,
        )
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: jax.Array, action: jax.Array) -> Tuple[Transition, EnvInfo, jax.Array]:
        """Step the environment."""
        next_env_state, (next_obs, reward, terminated, truncated, info) = self._step(env_state, action)
        done = terminated | truncated
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
        return self.env.observation_space # TODO: integrate with Jaxinn.envs.spaces to correct shape

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def action_size(self):
        return self.action_space.n
