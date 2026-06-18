import math
import numpy as np
from typing import Any, Optional, Tuple, Dict

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import gymnasium as gym

from jaxinn.structs import Transition
from envs.environment import Environment, EnvInfo, EnvState
from envs.vmap import VmapTransformation
from envs.spaces import to_jax_space


class GymnasiumVmapMixIn(VmapTransformation):
    @jax.custom_batching.custom_vmap
    def v_reset(self, key):
        data = self._reset()
        logical_shape = key.shape[:-1]
        data = jax.tree.map(self.make_get_logical(logical_shape), data)
        return data

    @v_reset.def_vmap
    def v_reset_batch(axis_size, in_batched, self, key):
        flattened = key.ndim > 2
        inner = key.shape[1] if flattened else 1
        if flattened:
            flatten_key = key.reshape(axis_size * key.shape[1], *key.shape[2:])
        else:
            flatten_key = key
        data = self.v_reset(self, flatten_key) # Recursion
        data = jax.tree.map(self.make_unflatten(axis_size, inner, flattened), data)
        out_batched = jax.tree.map(lambda _: True, data)
        return data, out_batched

    @jax.custom_batching.custom_vmap
    def v_step(self, key, action):
        logical_shape = key.shape[:-1]
        # Pad actions to satisfy the C++ primitive if logical_size < capacity
        padded_action = jax.tree.map(self.make_pad_leaf(logical_shape, self.capacity), action)
        data = self._step(padded_action)
        data = jax.tree.map(self.make_get_logical(logical_shape), data)
        return data

    @v_step.def_vmap
    def v_step_batch(axis_size, in_batched, self, key, action):
        flattened = key.ndim > 2
        inner = key.shape[1] if flattened else 1

        if flattened:
            flatten_action = action.reshape(axis_size * action.shape[1], *action.shape[2:])
            flatten_key = key.reshape(axis_size * key.shape[1], *key.shape[2:])
        else:
            flatten_action, flatten_key = action, key
        data = self.v_step(self, flatten_key, flatten_action)
        data = jax.tree.map(self.make_unflatten(axis_size, inner, flattened), data)
        out_batched = jax.tree.map(lambda _: True, data)
        return data, out_batched


class JaxConverterMixIn:
    @property
    def obs_struct(self):
        obs_shape = self.observation_space.shape
        obs_dtype = jnp.uint8 if getattr(self, "from_pixels", False) else jnp.float32
        return jax.ShapeDtypeStruct((self.capacity,) + obs_shape, obs_dtype)

    @property
    def reward_struct(self):
        return jax.ShapeDtypeStruct((self.capacity,) + (), jnp.float32)

    @property
    def terminated_struct(self):
        return jax.ShapeDtypeStruct((self.capacity,) + (), jnp.bool_)

    @property
    def truncated_struct(self):
        return jax.ShapeDtypeStruct((self.capacity,) + (), jnp.bool_)

    def _python_reset(self):
        obs, _ = self.env.reset()
        if isinstance(obs, (tuple, list)):
            obs = np.stack(obs)
        obs = obs.astype(np.uint8) if getattr(self, "from_pixels", False) else obs.astype(np.float32)
        return obs

    def _python_step(self, action):
        action = np.array(action, dtype=np.float32)
        obs, reward, term, trunc, _ = self.env.step(action)
        if isinstance(obs, (tuple, list)):
            obs = np.stack(obs)
        obs = obs.astype(np.uint8) if getattr(self, "from_pixels", False) else obs.astype(np.float32)
        return obs, np.float32(reward), np.bool_(term), np.bool_(trunc)

    def _reset(self):
        def reset_fn():
            return self._python_reset()
        obs = jax.pure_callback(
            reset_fn,
            self.obs_struct,
        )
        return obs

    def _step(self, action):
        def step_fn(act):
            return self._python_step(act)
        return jax.pure_callback(
            step_fn,
            (self.obs_struct, self.reward_struct, self.terminated_struct, self.truncated_struct),
            action,
        )


class Gymnasium(JaxConverterMixIn, GymnasiumVmapMixIn, Environment):
    def __init__(
            self,
            env: gym.Env,
            env_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(env, env_params)

    @classmethod
    def create(cls, env_name: str, num_envs: int, **kwargs) -> "Gymnasium":
        env = gym.make_vec(env_name, num_envs=num_envs, **kwargs)
        return cls(env, env_params={"capacity": num_envs, **kwargs})

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, jax.Array]:
        obs = self.v_reset(self, key)
        transition = Transition(
            action=jnp.zeros(self.action_space.shape, dtype=self.action_space.dtype),
            next_obs=obs,
            reward=jnp.zeros(()),
            terminated=jnp.zeros((), dtype=bool),
            truncated=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo(terminal_observation=None)
        env_state = EnvState(last_done=jnp.zeros((), dtype=bool))
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: jax.Array, action: jax.Array) -> Tuple[Transition, EnvInfo, jax.Array]:
        # VmapMixIn → Jax callback → python env
        next_obs, reward, terminated, truncated = self.v_step(self, key, action)
        # Gymnasium reset observation is independent of the action
        action = jnp.where(
            env_state.last_done,
            jnp.zeros_like(action),
            action
        )
        transition = Transition(
            action=action,
            next_obs=next_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        env_info = EnvInfo(terminal_observation=None) # Gymnasium vectorized env autoresets at next step, resulting a dummy transition while the previous transition is preserved, so we don't have to manually extract the terminal observation
        next_env_state = EnvState(last_done=terminated | truncated)
        return transition, env_info, next_env_state

    @property
    def observation_space(self):
        observation_space = self.env.single_observation_space if self.capacity is not None and hasattr(self.env, "single_observation_space") else self.env.observation_space
        return to_jax_space(observation_space)

    @property
    def action_space(self):
        action_space = self.env.single_action_space if (self.capacity is not None) and hasattr(self.env, "single_action_space") else self.env.action_space
        return to_jax_space(action_space)

    @property
    def action_size(self):
        if self.is_action_space_discrete:
            return self.action_space.n
        return math.prod(self.action_space.shape)

    @property
    def max_episode_length(self) -> int:
        return self.max_episode_steps
