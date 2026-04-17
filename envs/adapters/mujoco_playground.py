import numpy as np
from typing import Any, Callable, Optional, Tuple, List

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from envs.environment import Transition, Environment, EnvInfo
from envs.spaces import Discrete, Box, Dict

from mujoco_playground import registry
from mujoco.mjx import Model as MjxModel
from mujoco_playground import MjxEnv
from mujoco_playground import State as MjxState
from mujoco.mjx.warp import types as mjx_warp_types


def _build_static_leaf_names() -> frozenset[str]:
    """
    Fields where _BATCH_DIM['Data'][k] is False are NOT batched over the
    world/env dimension — they should be treated as static leaves and
    never sliced, padded, or reshaped by PlaygroundVmapMixIn vmap logic.
    """
    batch_dim = mjx_warp_types._BATCH_DIM['Data']
    return frozenset(k for k, is_batched in batch_dim.items() if not is_batched)


from jax._src.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, SequenceKey


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


def make_get_logical(logical_shape):
    logical_size = 1 if len(logical_shape) == 0 else logical_shape[0]

    def body(path, x):
        if is_static_leaf(path, x):
            return x
        if not hasattr(x, 'ndim') or x.ndim == 0:
            return x
        if len(logical_shape) == 0:
            return x[0]
        return x[:logical_size]

    return body


def make_unflatten(axis_size, inner_size, flattened):
    expected = axis_size * inner_size

    def body(path, x):
        if is_static_leaf(path, x):
            return x
        if not hasattr(x, 'ndim') or x.ndim == 0:
            return x
        if flattened and x.shape[0] == expected:
            return x.reshape(axis_size, inner_size, *x.shape[1:])
        return x

    return body


def make_out_batching(axis_size, inner_size, flattened):
    expected = axis_size * inner_size

    def body(path, x):
        if is_static_leaf(path, x):
            return False
        if not hasattr(x, 'ndim') or x.ndim == 0:
            return False
        return x.shape[0] == expected

    return body


def make_pad_leaf(logical_shape, capacity):
    logical_size = 1 if len(logical_shape) == 0 else logical_shape[0]
    pad_size = capacity - logical_size

    def body(path, x):
        if is_static_leaf(path, x):
            return x
        if len(logical_shape) == 0:
            x = jnp.expand_dims(x, 0)
        if not hasattr(x, 'ndim') or x.ndim == 0:
            return x
        pw = ((0, pad_size),) + ((0, 0),) * (x.ndim - 1)
        return jnp.pad(x, pw, mode='constant', constant_values=0)

    return body


def make_flatten(axis_size):
    def body(path, x):
        if is_static_leaf(path, x):
            return x
        if not hasattr(x, 'ndim') or x.ndim < 2:
            return x
        return x.reshape(axis_size * x.shape[1], *x.shape[2:])
    return body


class PlaygroundVmapMixIn:
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
        return jax.tree.map_with_path(make_get_logical(logical_shape), env_state)

    @v_reset.def_vmap
    def v_reset_batch(axis_size, in_batched, self, key):
        flattened = key.ndim > 2
        inner = key.shape[1] if flattened else 1
        flatten_key = key.reshape(axis_size * key.shape[1], *key.shape[2:]) if flattened else key

        env_state = self.v_reset(self, flatten_key) # Recursion

        new_env_state = jax.tree.map_with_path(make_unflatten(axis_size, inner, flattened), env_state)
        out_batched = jax.tree.map_with_path(make_out_batching(axis_size, inner, flattened), env_state)
        return new_env_state, out_batched

    @jax.custom_batching.custom_vmap
    def v_step(self, key, env_state, action):
        logical_shape = key.shape[:-1]

        # Pad actions to satisfy the C++ primitive if logical_size < capacity
        padded_env_state = jax.tree.map_with_path(make_pad_leaf(logical_shape, self.capacity), env_state)
        padded_action = jax.tree.map_with_path(make_pad_leaf(logical_shape, self.capacity), action)

        next_env_state = jax.vmap(self.env.step)(padded_env_state, padded_action)
        return jax.tree.map_with_path(make_get_logical(logical_shape), next_env_state)

    @v_step.def_vmap
    def v_step_batch(axis_size, in_batched, self, key, env_state, action):
        flattened = key.ndim > 2
        inner = key.shape[1] if flattened else 1

        if flattened:
            flatten_action = action.reshape(axis_size * inner, *action.shape[2:])
            flatten_key = key.reshape(axis_size * inner, *key.shape[2:])
            flatten_env_state = jax.tree.map_with_path(make_flatten(axis_size), env_state)
        else:
            flatten_action, flatten_key, flatten_env_state = action, key, env_state

        next_env_state = self.v_step(self, flatten_key, flatten_env_state, flatten_action)

        new_env_state = jax.tree.map_with_path(make_unflatten(axis_size, inner, flattened), next_env_state)
        out_batched = jax.tree.map_with_path(make_out_batching(axis_size, inner, flattened), next_env_state)
        return new_env_state, out_batched


class Playground(Environment, PlaygroundVmapMixIn):
    def __init__(
            self,
            env: MjxModel,
            env_params: Optional[Any] = None,
    ):
        super().__init__(env, env_params)

    @classmethod
    def create(cls, env_name: str, num_envs: int = 1, vision: bool = True, **kwargs) -> "Playground":
        env_params = registry.get_default_config(env_name)
        env_params.vision = vision
        if vision:
            env_params.vision_config.nworld = num_envs
        env_params.update(kwargs)
        env = registry.load(env_name, config_overrides=env_params)
        env_params["capacity"] = num_envs
        return cls(env, env_params)

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, MjxState]:
        env_state = self.v_reset(self, key)
        transition = Transition(
            action=jnp.zeros(self.action_space.shape, dtype=self.action_space.dtype),
            next_obs=env_state.obs["pixels/view_0"] if self.vision else env_state.obs,
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo(
            info=env_state.info,
            metrics=env_state.metrics,
            reset=True,
        )
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: MjxState, action: jax.Array) -> Tuple[Transition, EnvInfo, MjxState]:
        """Step the environment."""
        next_env_state = self.v_step(self, key, env_state, action)
        transition = Transition(
            action=action,
            next_obs=next_env_state.obs["pixels/view_0"] if self.vision else next_env_state.obs,
            reward=next_env_state.reward,
            done=next_env_state.done.astype(bool),
        )
        env_info = EnvInfo(
            info=next_env_state.info,
            metrics=next_env_state.metrics,
            reset=False,
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
    def observation_size(self) -> int:
        return self.env.observation_size

    @property
    def action_space(self) -> Box:
        return Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_size,),
            dtype=jnp.float32
        )

    @property
    def action_size(self) -> int:
        return self.env.action_size
