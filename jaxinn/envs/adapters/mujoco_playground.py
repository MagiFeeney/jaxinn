from ml_collections import config_dict
from collections.abc import Mapping
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from jax._src.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, SequenceKey
from mujoco_playground import registry
from mujoco_playground import MjxEnv
from mujoco_playground import State as MjxState
from mujoco_playground._src import dm_control_suite, locomotion, manipulation
from mujoco import mjx
from mujoco.mjx.warp import types as mjx_warp_types

from jaxinn.structs import Transition

from ..environment import Environment, EnvInfo
from ..spaces import Box, Dict
from ..vmap import VmapTransformation


def get_default_vision_config(env_name: str, num_envs: int = 1) -> config_dict.ConfigDict:
    # common
    config = config_dict.create(
        nworld=num_envs,
        cam_res=(64, 64),
        use_shadows=False,
        render_rgb=True,
        render_depth=False,
        enabled_geom_groups=[0, 1, 2],
    )

    if env_name in dm_control_suite.ALL_ENVS:
        config.use_textures = False
        config.cam_active = (True, False)  # [fixed, lookatcart]

    elif env_name in manipulation.ALL_ENVS:
        config.use_textures = False
        config.cam_active = None           # Use all cameras

    elif env_name in locomotion.ALL_ENVS:
        config.use_textures = True
        config.cam_active = (True,)        # Use primary camera

    else:
        raise ValueError(f"Env '{env_name}' not found in available suites.")

    return config


class MjxVisionWrapper:
    def __init__(
        self,
        env: MjxEnv,
        vision_config: Any,
        pixels_only: bool = True,
        obs_noise: Optional[Any] = None
    ):
        self._env = env
        self._pixels_only = pixels_only
        self._obs_noise = obs_noise

        if isinstance(vision_config, config_dict.ConfigDict):
            self._vision_config = vision_config
        elif isinstance(vision_config, dict):
            self._vision_config = config_dict.ConfigDict(vision_config)
        elif hasattr(vision_config, "__dict__"):
            self._vision_config = config_dict.ConfigDict(vars(vision_config))
        else:
            raise TypeError(f"Vision config must be convertible to ConfigDict, but got {type(vision_config)}.")

        vision_kwargs = self._vision_config.to_dict()

        if "cam_active" in vision_kwargs and vision_kwargs["cam_active"] is not None:
            cam_active = list(vision_kwargs["cam_active"])
            ncam = self._env.mj_model.ncam

            if len(cam_active) < ncam:   # Default to use the first camera
                cam_active.extend([False] * (ncam - len(cam_active)))
            elif len(cam_active) > ncam: # Truncate if have more than required
                cam_active = cam_active[:ncam]

            vision_kwargs["cam_active"] = tuple(cam_active)

        self._rc = mjx.create_render_context(
            mjm=self._env.mj_model,
            **vision_kwargs
        )
        self._rc_pytree = self._rc.pytree()

    @property
    def mj_model(self):
        return self._env.mj_model

    @property
    def mjx_model(self):
        return self._env.mjx_model

    @property
    def action_size(self):
        return self._env.action_size

    def reset(self, rng: jax.Array) -> MjxState:
        rng_reset, rng_noise = jax.random.split(rng, 2)
        state = self._env.reset(rng_reset)

        if self._obs_noise is not None and hasattr(self._obs_noise, "brightness"):
            brightness = jax.random.uniform(
                rng_noise,
                (1,),
                minval=self._obs_noise.brightness[0],
                maxval=self._obs_noise.brightness[1],
            )
            state.info['brightness'] = brightness

        state = state.replace(info=state.info)
        return self._render_and_update_obs(state)

    def step(self, state: MjxState, action: jax.Array) -> MjxState:
        state = self._env.step(state, action)
        return self._render_and_update_obs(state)

    @staticmethod
    def _adjust_brightness(img, scale):
        return jnp.clip(img * scale, 0, 1)

    def _render_and_update_obs(self, state: MjxState) -> MjxState:
        render_data = mjx.refit_bvh(self.mjx_model, state.data, self._rc_pytree)
        out = mjx.render(self.mjx_model, render_data, self._rc_pytree)
        rgb = mjx.get_rgb(self._rc_pytree, 0, out[0])

        if 'brightness' in state.info:
            rgb = self._adjust_brightness(rgb, state.info['brightness'])

        if self._pixels_only:
            obs = {"pixels/view_0": rgb}
        else:
            obs = {
                "state": state.obs,
                "pixels/view_0": rgb
            }
        return state.replace(obs=obs)


def _build_static_leaf_names() -> frozenset[str]:
    """
    Fields where _BATCH_DIM['Data'][k] is False are NOT batched over the
    world/env dimension — they should be treated as static leaves and
    never sliced, padded, or reshaped by PlaygroundVmapMixIn vmap logic.
    """
    batch_dim = mjx_warp_types._BATCH_DIM['Data']
    return frozenset(k for k, is_batched in batch_dim.items() if not is_batched)


def get_leaf_name(path: tuple) -> str | None:
    """Extract the leaf field name from a JAX tree path."""
    if not path:
        return None
    last = path[-1]
    match last:
        case DictKey(key=k):           return str(k)
        case GetAttrKey(name=a):       return str(a)   # NamedTuple / dataclass fields
        case SequenceKey(idx=i):       return str(i)
        case FlattenedIndexKey(key=k): return str(k)
    return None


_STATIC_LEAF_NAMES: frozenset[str] = _build_static_leaf_names()


def is_static_leaf(path, x) -> bool:
    return get_leaf_name(path) in _STATIC_LEAF_NAMES


class PlaygroundVmapMixIn(VmapTransformation):
    is_static_leaf = staticmethod(is_static_leaf)

    def _reset(self, key):
        keys = jax.random.split(key, self.capacity)
        env_state = jax.vmap(self.env.reset)(keys)
        return env_state

    @jax.custom_batching.custom_vmap
    def v_reset(self, key):
        logical_shape = key.shape[:-1]
        if key.ndim == 1:
            key, key_reset = jax.random.split(key, 2)
        else:
            key_replacement, key_reset = jax.random.split(key[0], 2)
            key = key.at[0].set(key_replacement)

        env_state = self._reset(key_reset)
        return jax.tree.map_with_path(self.make_get_logical(logical_shape), env_state)

    @v_reset.def_vmap
    def v_reset_batch(axis_size, in_batched, self, key):
        flattened = key.ndim > 2
        inner = key.shape[1] if flattened else 1
        flatten_key = key.reshape(axis_size * key.shape[1], *key.shape[2:]) if flattened else key

        env_state = self.v_reset(self, flatten_key) # Recursion

        new_env_state = jax.tree.map_with_path(self.make_unflatten(axis_size, inner, flattened), env_state)
        out_batched = jax.tree.map_with_path(self.make_out_batching(axis_size, inner, flattened), env_state)
        return new_env_state, out_batched

    @jax.custom_batching.custom_vmap
    def v_step(self, key, env_state, action):
        logical_shape = key.shape[:-1]

        # Pad actions to satisfy the C++ primitive if logical_size < capacity
        padded_env_state = jax.tree.map_with_path(self.make_pad_leaf(logical_shape, self.capacity), env_state)
        padded_action = jax.tree.map_with_path(self.make_pad_leaf(logical_shape, self.capacity), action)

        next_env_state = jax.vmap(self.env.step)(padded_env_state, padded_action)
        return jax.tree.map_with_path(self.make_get_logical(logical_shape), next_env_state)

    @v_step.def_vmap
    def v_step_batch(axis_size, in_batched, self, key, env_state, action):
        flattened = key.ndim > 2
        inner = key.shape[1] if flattened else 1

        if flattened:
            flatten_action = action.reshape(axis_size * inner, *action.shape[2:])
            flatten_key = key.reshape(axis_size * inner, *key.shape[2:])
            flatten_env_state = jax.tree.map_with_path(self.make_flatten(axis_size), env_state)
        else:
            flatten_action, flatten_key, flatten_env_state = action, key, env_state

        next_env_state = self.v_step(self, flatten_key, flatten_env_state, flatten_action)

        new_env_state = jax.tree.map_with_path(self.make_unflatten(axis_size, inner, flattened), next_env_state)
        out_batched = jax.tree.map_with_path(self.make_out_batching(axis_size, inner, flattened), next_env_state)
        return new_env_state, out_batched


class Playground(Environment, PlaygroundVmapMixIn):
    def __init__(
            self,
            env: MjxEnv,
            env_params: Optional[Any] = None,
    ):
        super().__init__(env, env_params)

    @classmethod
    def create(cls, env_name: str, num_envs: int = 1, vision: bool = True, **kwargs) -> "Playground":
        env_params = registry.get_default_config(env_name)

        has_native_vision = "vision_config" in env_params
        if vision:
            if "vision_config" in kwargs:
                vision_config = kwargs.pop("vision_config")
            elif hasattr(env_params, "vision_config"):
                vision_config = env_params.vision_config
            else:
                vision_config = get_default_vision_config(env_name, num_envs)
            vision_config.nworld = num_envs

            if has_native_vision: # Override in time
                env_params.vision = vision
                env_params.vision_config = vision_config

        env_params.update(kwargs)

        env = registry.load(env_name, config_overrides=env_params)

        if vision and not has_native_vision: # Post-hoc processing for non-native vision environments
            env = MjxVisionWrapper(
                env,
                vision_config,
                pixels_only=getattr(env_params, "pixels_only", True),
                obs_noise=getattr(env_params, "obs_noise", None)
            )
            env_params.vision = vision

        env_params["capacity"] = num_envs
        return cls(env, env_params)

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, MjxState]:
        env_state = self.v_reset(self, key)
        transition = Transition(
            action=jnp.zeros(self.action_space.shape, dtype=self.action_space.dtype),
            next_obs=env_state.obs["pixels/view_0"] if self.vision else env_state.obs,
            reward=jnp.zeros(()),
            terminated=jnp.zeros((), dtype=bool),
            truncated=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo(
            info=env_state.info,
            metrics=env_state.metrics,
            boundary_observation=jnp.zeros_like(transition.next_obs), # dummy
        )
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: MjxState, action: jax.Array) -> Tuple[Transition, EnvInfo, MjxState]:
        next_env_state = self.v_step(self, key, env_state, action)
        terminated = next_env_state.done.astype(bool)
        transition = Transition(
            action=action,
            next_obs=next_env_state.obs["pixels/view_0"] if self.vision else next_env_state.obs,
            reward=next_env_state.reward,
            terminated=terminated,
            truncated=jnp.zeros_like(terminated) # delegated to TimeLimit wrapper
        )
        env_info = EnvInfo(
            info=next_env_state.info,
            metrics=next_env_state.metrics,
        )
        return transition, env_info, next_env_state

    @property
    def observation_space(self) -> Box:
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
    def action_space(self) -> Box:
        return Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_size,),
            dtype=jnp.float32
        )

    @property
    def observation_size(self) -> int:
        key = jax.random.PRNGKey(0)

        if self.vision:
            keys = jax.random.split(key, self.capacity)
            abstract_state = jax.eval_shape(jax.vmap(self.env.reset), keys)
        else:
            abstract_state = jax.eval_shape(self.env.reset, key)

        obs = abstract_state.obs

        if isinstance(obs, Mapping):
            return jax.tree.map(lambda x: x.shape[1:] if self.vision else x.shape, obs)
        return obs.shape[-1]

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def max_episode_length(self) -> None:
        return None
