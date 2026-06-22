import math
from typing import Any, Callable, Optional, Tuple, Dict

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import envpool
from envpool.python.envpool import EnvPoolMixin
import equinox as eqx

from jaxinn.structs import Transition

from ..environment import Environment, EnvInfo
from ..spaces import to_jax_space
from ..vmap import VmapTransformation


class EnvPoolVmapMixIn(VmapTransformation):
    @jax.custom_batching.custom_vmap
    def v_reset(self, key):
        data = self.env.reset()
        logical_shape = key.shape[:-1]
        data = jax.tree.map(self.make_get_logical(logical_shape), data)
        return self._handle, data

    @v_reset.def_vmap
    def v_reset_batch(axis_size, in_batched, self, key):
        flattened = key.ndim > 2
        inner = key.shape[1] if flattened else 1
        if flattened:
            flatten_key = key.reshape(axis_size * key.shape[1], *key.shape[2:])
        else:
            flatten_key = key
        new_handle, data = self.v_reset(self, flatten_key) # Recursion
        batched_handle = jnp.broadcast_to(new_handle, (axis_size,) + new_handle.shape)

        data = jax.tree.map(self.make_unflatten(axis_size, inner, flattened), data)
        results = (batched_handle, data)
        out_batched = jax.tree.map(lambda _: True, results)
        return results, out_batched

    @jax.custom_batching.custom_vmap
    def v_step(self, key, handle, action):
        logical_shape = key.shape[:-1]
        # Pad actions to satisfy the C++ primitive if logical_size < capacity
        padded_action = jax.tree.map(self.make_pad_leaf(logical_shape, self.capacity), action)

        target_dtype = self.action_space.dtype
        with jax.enable_x64(target_dtype == jnp.float64):
            new_handle, data = self._step(handle, padded_action.astype(target_dtype))

        data = jax.tree.map(self.make_get_logical(logical_shape), data)
        return new_handle, data

    @v_step.def_vmap
    def v_step_batch(axis_size, in_batched, self, key, handle, action):
        single_handle = handle[0] if in_batched[0] else handle
        flattened = key.ndim > 2
        inner = key.shape[1] if flattened else 1

        if flattened:
            flatten_action = action.reshape(axis_size * action.shape[1], *action.shape[2:])
            flatten_key = key.reshape(axis_size * key.shape[1], *key.shape[2:])
        else:
            flatten_action, flatten_key = action, key
        new_handle, data = self.v_step(self, flatten_key, single_handle, flatten_action)
        batched_handle = jnp.broadcast_to(new_handle, (axis_size,) + new_handle.shape)

        data = jax.tree.map(self.make_unflatten(axis_size, inner, flattened), data)
        results = (batched_handle, data)
        out_batched = jax.tree.map(lambda _: True, results)
        return results, out_batched


class EnvPool(Environment, EnvPoolVmapMixIn):
    _handle: jax.Array = eqx.field(static=True)
    _step: Callable = eqx.field(static=True)

    def __init__(
            self,
            env: EnvPoolMixin,
            env_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(env, env_params)
        handle, recv, send, step = env.xla()
        self._handle = handle
        self._step = step

    @classmethod
    def create(cls, env_name: str, num_envs: int, **kwargs) -> "EnvPool":
        env = envpool.make(env_name, env_type="gymnasium", num_envs=num_envs, **kwargs)
        return cls(env, env_params={"capacity": num_envs, **kwargs})

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, jax.Array]:
        env_state, (obs, info) = self.v_reset(self, key)
        transition = Transition(
            action=jnp.zeros(self.action_space.shape, dtype=self.action_space.dtype),
            next_obs=obs,
            reward=jnp.zeros(()),
            terminated=jnp.zeros((), dtype=bool),
            truncated=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo(
            info=info,
            terminal_observation=None,
        )
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: jax.Array, action: jax.Array) -> Tuple[Transition, EnvInfo, jax.Array]:
        next_env_state, (next_obs, reward, terminated, truncated, info) = self.v_step(self, key, env_state, action)
        transition = Transition(
            action=action,
            next_obs=next_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        env_info = EnvInfo(
            info=info,
            terminal_observation=None, # envpool uses next-step autoreset instead of same-step autoreset; same as gymnasium logic
        )
        return transition, env_info, next_env_state

    @property
    def observation_space(self):
        return to_jax_space(self.env.observation_space)

    @property
    def action_space(self):
        return to_jax_space(self.env.action_space)

    @property
    def action_size(self):
        if self.is_action_space_discrete:
            return self.action_space.n
        return math.prod(self.action_space.shape)

    @property
    def max_episode_length(self) -> int:
        return self.max_episode_steps
