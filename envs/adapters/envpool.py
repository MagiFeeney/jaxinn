import math
from typing import Any, Callable, Optional, Tuple, Dict

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx
from envs.environment import Transition, Environment, EnvInfo

import envpool
from envpool.python.envpool import EnvPoolMixin


class EnvPool(Environment):
    _handle: jax.Array = eqx.field(static=True)
    _reset: Callable = eqx.field(static=True)
    _step: Callable = eqx.field(static=True)

    def __init__(
            self,
            env: EnvPoolMixin,
            env_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(env, env_params)
        handle, recv, send, step = env.xla()
        self._handle = handle

        @jax.custom_batching.custom_vmap
        def v_reset(key, handle):
            data = self.env.reset()
            logical_size = key.shape[0]

            def get_logical(x):
                if x.ndim == 0: return x
                return x[:logical_size]

            data = jax.tree.map(get_logical, data)
            return handle, data

        @v_reset.def_vmap
        def v_reset_batch(axis_size, in_batched, key, handle):
            single_handle = handle[0] if in_batched[1] else handle
            flatten_key = key.reshape(axis_size * key.shape[1], *key.shape[2:])
            new_handle, data = v_reset(flatten_key, single_handle) # Recursion
            batched_handle = jnp.broadcast_to(new_handle, (axis_size,) + new_handle.shape)

            def unflatten(x):
                if x.ndim == 0: return x
                return x.reshape(axis_size, key.shape[1], *x.shape[1:])

            data = jax.tree.map(unflatten, data)
            results = (batched_handle, data)
            out_batched = jax.tree.map(lambda _: True, results)
            return results, out_batched

        @jax.custom_batching.custom_vmap
        def v_step(handle, action):
            logical_size = action.shape[0]

            # Pad actions to satisfy the C++ primitive if logical_size < capacity
            pad_size = env_params["capacity"] - logical_size
            pad_width = ((0, pad_size),) + ((0, 0),) * (action.ndim - 1)
            flatten_action = jnp.pad(action, pad_width, mode='constant', constant_values=0)
            new_handle, data = step(handle, flatten_action)

            def get_logical(x):
                if x.ndim == 0: return x
                return x[:logical_size]

            data = jax.tree.map(get_logical, data)
            return new_handle, data

        @v_step.def_vmap
        def v_step_batch(axis_size, in_batched, handle, action):
            single_handle = handle[0] if in_batched[0] else handle
            flatten_action = action.reshape(axis_size * action.shape[1], *action.shape[2:])
            new_handle, data = v_step(single_handle, flatten_action)
            batched_handle = jnp.broadcast_to(new_handle, (axis_size,) + new_handle.shape)

            def unflatten(x):
                if x.ndim == 0: return x
                return x.reshape(axis_size, action.shape[1], *x.shape[1:])

            data = jax.tree.map(unflatten, data)
            results = (batched_handle, data)
            out_batched = jax.tree.map(lambda _: True, results)
            return results, out_batched

        self._step = v_step
        self._reset = v_reset

    @classmethod
    def create(cls, env_name: str, num_envs: int, vmap_multiplier: int, **kwargs) -> "EnvPool":
        capacity = num_envs * vmap_multiplier
        env = envpool.make(env_name, env_type="gymnasium", num_envs=capacity, **kwargs)
        env_params = {"num_envs": num_envs, "capacity": capacity}
        env_params.update(kwargs)
        return cls(env, env_params=env_params)

    def reset(self, key: PRNGKeyArray, num_envs: int | None = None) -> Tuple[Transition, EnvInfo, jax.Array]:
        num_keys = num_envs if num_envs is not None else self.env_params["num_envs"]
        keys = jax.random.split(key, num_keys)
        env_state, (obs, info) = self._reset(keys, self._handle)
        transition = Transition(
            action=jnp.zeros((num_keys,) + self.action_space.shape, dtype=self.action_space.dtype),
            next_obs=obs,
            reward=jnp.zeros((num_keys)),
            done=jnp.zeros((num_keys), dtype=bool),
        )
        env_info = EnvInfo(
            info=info,
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
        info["terminal_observation"] = next_obs # envpool uses next-step auto-reset instead of same-step auto-reset
        env_info = EnvInfo(
            info=info,
            reset=False,
        )
        return transition, env_info, next_env_state

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def action_size(self):
        return self.action_space.n
