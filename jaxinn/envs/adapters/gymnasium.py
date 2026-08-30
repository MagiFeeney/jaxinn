import numpy as np
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import gymnasium as gym

from jaxinn.common.structs import Transition

from ..environment import Environment, EnvInfo, EnvState
from ..vmap import VmapTransformation
from ..spaces import Discrete, Box, Dict, Tuple


def gymnasium_space_to_jaxinn_space(space):
    if isinstance(space, gym.spaces.Discrete):
        return Discrete(n=int(space.n), dtype=jax.dtypes.canonicalize_dtype(space.dtype))
    elif isinstance(space, gym.spaces.Box):
        return Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=jax.dtypes.canonicalize_dtype(space.dtype),
        )
    elif isinstance(space, gym.spaces.Dict):
        converted_spaces = {k: gymnasium_space_to_jaxinn_space(v) for k, v in space.spaces.items()}
        return Dict(converted_spaces)
    elif isinstance(space, gym.spaces.Tuple):
        converted_spaces = tuple(gymnasium_space_to_jaxinn_space(s) for s in space.spaces)
        return Tuple(converted_spaces)
    else:
        raise TypeError(
            f"Unsupported Gymnasium space type for conversion to Jaxinn space: '{type(space).__name__}'."
        )


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
            flatten_action = jax.tree.map(lambda x: x.reshape(axis_size * x.shape[1], *x.shape[2:]), action)
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
        action = jax.device_get(action)
        obs, reward, term, trunc, _ = self.env.step(action) # TODO: return info
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
            env_params: dict[str, Any] | None = None,
    ):
        super().__init__(env, env_params)

    @classmethod
    def create(cls, env_name: str, num_envs: int, vectorization_mode: str, **kwargs) -> "Gymnasium":
        env_params = {"capacity": num_envs, **kwargs}

        if env_name.split("/")[0] == "ALE":
            import ale_py
            gym.register_envs(ale_py)

            max_episode_steps = kwargs.pop("max_episode_steps", 27000)
            env_params["max_episode_steps"] = max_episode_steps

            frame_skip = kwargs.pop("frame_skip", 4)
            noop_max = kwargs.pop("noop_max", 30)
            screen_size = kwargs.pop("screen_size", 64)
            terminal_on_life_loss = kwargs.pop("terminal_on_life_loss", False)
            grayscale_obs = kwargs.pop("grayscale_obs", True)

            wrappers = [
                lambda env: gym.wrappers.AtariPreprocessing(
                    env,
                    frame_skip=frame_skip,
                    noop_max=noop_max,
                    screen_size=screen_size,
                    terminal_on_life_loss=terminal_on_life_loss,
                    grayscale_obs=grayscale_obs,
                ),
                lambda env: gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps) if max_episode_steps is not None else env
            ]

            kwargs['wrappers'] = wrappers
            kwargs["frameskip"] = 1 # override as the wrapper takes care of it
            kwargs["max_episode_steps"] = None # same as the above

        env = gym.make_vec(env_name, num_envs=num_envs, vectorization_mode=vectorization_mode, **kwargs)
        return cls(env, env_params=env_params)

    def reset(self, key: PRNGKeyArray) -> tuple[Transition, EnvInfo, jax.Array]:
        obs = self.v_reset(self, key)
        transition = Transition(
            action = jax.tree.map(
                lambda shape, dtype: jnp.zeros(shape, dtype=dtype),
                self.action_space.shape,
                self.action_space.dtype,
                is_leaf=lambda x: isinstance(x, tuple)
            ),
            next_obs=obs,
            reward=jnp.zeros(()),
            terminated=jnp.zeros((), dtype=bool),
            truncated=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo()
        env_state = EnvState()
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: jax.Array, action: jax.Array) -> tuple[Transition, EnvInfo, jax.Array]:
        # VmapMixIn → Jax callback → python env
        next_obs, reward, terminated, truncated = self.v_step(self, key, action)
        transition = Transition(
            action=action,
            next_obs=next_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        env_info = EnvInfo()
        next_env_state = EnvState()
        return transition, env_info, next_env_state

    @property
    def observation_space(self):
        observation_space = self.env.single_observation_space if self.capacity is not None and hasattr(self.env, "single_observation_space") else self.env.observation_space
        return gymnasium_space_to_jaxinn_space(observation_space)

    @property
    def action_space(self):
        action_space = self.env.single_action_space if (self.capacity is not None) and hasattr(self.env, "single_action_space") else self.env.action_space
        return gymnasium_space_to_jaxinn_space(action_space)

    @property
    def max_episode_length(self) -> int:
        if getattr(self, "max_episode_steps", None) is not None:
            return self.max_episode_steps
        if getattr(self, "spec", None) is not None:
            if getattr(self.spec, "max_episode_steps", None) is not None:
                return self.spec.max_episode_steps
        raise AttributeError(
            f"Environment '{self.__class__.__name__}' does not define 'max_episode_steps' "
            f"directly or within its spec."
        )
