from typing import Any, Callable, Tuple, Optional

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx

from jaxinn.structs import Transition
from .environment import Environment, EnvInfo, EnvState
from .spaces import OneHotDiscrete


class Wrapper(Environment):
    """Base class for JAX environment wrappers."""

    def __init__(self, env: Environment):
        self.env = env
        self.env_params = env.env_params # populate env_params

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, EnvState]:
        return self.env.reset(key)

    def step(self, key: PRNGKeyArray, env_state: EnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, EnvState]:
        return self.env.step(key, env_state, action)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def max_episode_length(self) -> Optional[int]:
        return self.env.max_episode_length

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError
        return getattr(self.env, name)


class Batched(Wrapper):
    num_envs: int = eqx.field(static=True)
    vmap_reset: Callable = eqx.field(static=True)
    vmap_step: Callable = eqx.field(static=True)

    def __init__(self, env: Environment, num_envs: int):
        super().__init__(env)
        self.num_envs = num_envs
        self.vmap_reset = jax.vmap(self.env.reset)
        self.vmap_step = jax.vmap(self.env.step)

    def reset(self, key: PRNGKeyArray, num_envs: int | None = None) -> Tuple[Transition, EnvInfo, EnvState]:
        num_keys = num_envs if num_envs is not None else self.num_envs
        keys = jax.random.split(key, num_keys)
        return self.vmap_reset(keys)

    def step(self, key: PRNGKeyArray, env_state: EnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, EnvState]:
        num_keys = jax.tree.leaves(action)[0].shape[0] # Infer from action since we have already known that at reset
        keys = jax.random.split(key, num_keys)
        return self.vmap_step(keys, env_state, action)


class TimeLimit(Wrapper):
    max_episode_length: int = eqx.field(static=True)

    def __init__(self, env: Environment, max_episode_length: int):
        super().__init__(env)
        self.max_episode_length = max_episode_length

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, EnvState]:
        transition, env_info, env_state = self.env.reset(key)

        state_with_time = EnvState(
            state=env_state,
            time=jnp.zeros((), dtype=jnp.int32),
        )
        return transition, env_info, state_with_time

    def step(self, key: PRNGKeyArray, env_state: EnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, EnvState]:
        transition, env_info, next_env_state = self.env.step(key, env_state.state, action)
        time = env_state.time + 1
        truncated = time >= self.max_episode_length
        new_transition = eqx.tree_at(lambda t: t.truncated, transition, truncated)
        state_with_time = EnvState(
            state=next_env_state,
            time=time,
        )
        return new_transition, env_info, state_with_time


class AutoReset(Wrapper):
    def __init__(self, env: Environment):
        super().__init__(env)

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, EnvState]:
        return self.env.reset(key)

    def step(self, key: PRNGKeyArray, env_state: EnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, EnvState]:
        key_step, key_reset = jax.random.split(key)
        step_transition, step_env_info, step_env_state = self.env.step(key_step, env_state, action)
        reset_transition, reset_env_info, reset_env_state = self.env.reset(key_reset)

        def select_fn(path, reset_val, step_val):
            if hasattr(self, "is_static_leaf") and callable(self.is_static_leaf) and self.is_static_leaf(path, step_val):
                return step_val
            done = step_transition.terminated | step_transition.truncated
            if done.ndim == 0:
                return jnp.where(done, reset_val, step_val)
            if done.shape[0] != reset_val.shape[0]:
                return step_val
            d = done.reshape((step_val.shape[0],) + (1,) * (step_val.ndim - 1))
            return jnp.where(d, reset_val, step_val)

        final_env_state = jax.tree.map_with_path(
            select_fn,
            reset_env_state,
            step_env_state
        )
        _next_obs = jax.tree.map_with_path(
            select_fn,
            reset_transition.next_obs,
            step_transition.next_obs
        )
        final_transition = eqx.tree_at(lambda t: t.next_obs, step_transition, _next_obs)
        final_info = EnvInfo(**step_env_info.data, terminal_observation=step_transition.next_obs)
        return final_transition, final_info, final_env_state


class ActionRepeat(Wrapper):
    action_repeat: int = eqx.field(static=True)

    def __init__(self, env: Environment, action_repeat: int):
        super().__init__(env)
        self.action_repeat = action_repeat

    def step(self, key: PRNGKeyArray, env_state: EnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, EnvState]:
        key, key_step = jax.random.split(key, 2)
        transition, env_info, next_env_state = self.env.step(key_step, env_state, action)
        first_reward = transition.reward

        def repeat_fn(carry, _):
            transition, env_info, env_state, key = carry
            key, key_step = jax.random.split(key, 2)
            prev_done = transition.terminated | transition.truncated

            step_transition, step_env_info, step_env_state = self.env.step(key_step, env_state, action)

            def select_fn(path, old_val, new_val):
                if hasattr(self, "is_static_leaf") and callable(self.is_static_leaf) and self.is_static_leaf(path, old_val):
                    return old_val
                return jnp.where(prev_done, old_val, new_val)

            transition = jax.tree.map_with_path(select_fn, transition, step_transition)
            env_info = jax.tree.map_with_path(select_fn, env_info, step_env_info)
            next_env_state = jax.tree.map_with_path(select_fn, env_state, step_env_state)

            reward = jnp.where(prev_done, 0.0, transition.reward)
            return (transition, env_info, next_env_state, key), reward

        if self.action_repeat > 1:
            (transition, env_info, next_env_state, _), rewards = jax.lax.scan(
                repeat_fn,
                (transition, env_info, next_env_state, key),
                None,
                self.action_repeat - 1
            )
            total_reward = first_reward + jnp.sum(rewards, axis=0)
            transition = eqx.tree_at(lambda t: t.reward, transition, total_reward)
        return transition, env_info, next_env_state


def walk_and_apply(node, transform_fn, target_key="terminal_observation"):
    """
    Recursively walks a dictionary and applies transform_fn to the value
    of any key matching target_key.
    """
    if not isinstance(node, dict):
        return node

    return {
        k: transform_fn(v) if k == target_key else walk_and_apply(v, transform_fn, target_key)
        for k, v in node.items()
    }


class ChannelFirst(Wrapper):
    def __init__(self, env: Environment):
        super().__init__(env)

    @staticmethod
    def process_obs(obs: jax.Array) -> jax.Array:
        """Process observation from NHWC to NCHW for the image input while turning the 2D input as gray image."""
        if obs.ndim in (3, 4) and obs.shape[-1] in (1, 3, 4, 6):
            obs = jnp.moveaxis(obs, source=-1, destination=-3)
        elif obs.ndim in (2, 3):
            obs = jnp.expand_dims(obs, axis=-3)    # Convert to gray image

        return obs

    def process_observation_space(self, space: Any) -> Any:
        """Align with and reflect the processed observation."""
        if hasattr(space, 'spaces') and isinstance(space.spaces, dict):
            new_spaces = {k: self.process_observation_space(v) for k, v in space.spaces.items()}
            return type(space)(new_spaces)

        if hasattr(space, 'shape') and hasattr(space, 'low') and hasattr(space, 'high'):
            if len(space.shape) == 3 and space.shape[-1] in (1, 3, 4, 6):
                new_shape = (space.shape[2], space.shape[0], space.shape[1])
                low = jnp.moveaxis(space.low, -1, -3) if getattr(space.low, 'ndim', 0) == 3 else space.low
                high = jnp.moveaxis(space.high, -1, -3) if getattr(space.high, 'ndim', 0) == 3 else space.high
                return type(space)(low=low, high=high, shape=new_shape, dtype=space.dtype)
            elif len(space.shape) == 2:
                new_shape = (1, space.shape[0], space.shape[1])
                low = space.low[None, ...] if getattr(space.low, 'ndim', 0) == 2 else space.low
                high = space.high[None, ...] if getattr(space.high, 'ndim', 0) == 2 else space.high
                return type(space)(low=low, high=high, shape=new_shape, dtype=space.dtype)
        return space

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, EnvState]:
        transition, info, state = self.env.reset(key)
        transition = eqx.tree_at(lambda t: t.next_obs, transition, self.process_obs(transition.next_obs))
        info = eqx.tree_at(lambda x: x.data, info, walk_and_apply(info.data, self.process_obs))
        return transition, info, state

    def step(self, key: PRNGKeyArray, env_state: EnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, EnvState]:
        transition, info, next_state = self.env.step(key, env_state, action)
        transition = eqx.tree_at(lambda t: t.next_obs, transition, self.process_obs(transition.next_obs))
        info = eqx.tree_at(lambda x: x.data, info, walk_and_apply(info.data, self.process_obs))
        return transition, info, next_state

    @property
    def observation_space(self):
        space = self.env.observation_space
        space = self.process_observation_space(space)
        return space


class UnsqueezeScalar(Wrapper):
    def __init__(self, env: Environment):
        super().__init__(env)

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, EnvState]:
        transition, info, state = self.env.reset(key)
        transition = eqx.tree_at(
            lambda t: (t.reward, t.terminated, t.truncated),
            transition,
            (
                jnp.atleast_1d(transition.reward),
                jnp.atleast_1d(transition.terminated),
                jnp.atleast_1d(transition.truncated),
            )
        )
        return transition, info, state

    def step(self, key: PRNGKeyArray, env_state: EnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, EnvState]:
        transition, info, next_state = self.env.step(key, env_state, action)
        transition = eqx.tree_at(
            lambda t: (t.reward, t.terminated, t.truncated),
            transition,
            (
                jnp.atleast_1d(transition.reward),
                jnp.atleast_1d(transition.terminated),
                jnp.atleast_1d(transition.truncated),
            )
        )
        return transition, info, next_state


class OneHotAction(Wrapper):
    def __init__(self, env: Environment):
        super().__init__(env)

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, EnvState]:
        transition, env_info, env_state = self.env.reset(key)
        batch_shape = transition.reward.shape
        target_shape = batch_shape + self.action_space.shape
        dummy_one_hot_action = jnp.zeros(target_shape, dtype=self.action_space.dtype)
        transition = eqx.tree_at(lambda t: t.action, transition, dummy_one_hot_action)
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: EnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, EnvState]:
        discrete_action = jnp.argmax(action, axis=-1)
        # Interact with actual scalar action
        transition, env_info, next_env_state = self.env.step(key, env_state, discrete_action)
        # Store as one-hot vector
        transition = eqx.tree_at(lambda t: t.action, transition, action)
        return transition, env_info, next_env_state

    @property
    def action_space(self):
        space = self.env.action_space
        if self.env.is_action_space_discrete:
            return OneHotDiscrete(n=space.n, dtype=jnp.float32)
        return space


class ResizeImage(Wrapper):
    """Resize image to a target shape."""
    target_shape: Tuple[int, int] = eqx.field(static=True, default=(64, 64))

    def __init__(self, env: Environment, target_shape: Tuple[int, int]):
        super().__init__(env)
        self.target_shape = target_shape

    def _upscale(self, obs: jax.Array) -> jax.Array:
        if obs.shape[-2:] != self.target_shape:
            target_shape = obs.shape[:-2] + self.target_shape
            obs = jax.image.resize(obs, shape=target_shape, method="nearest")
        return obs.astype(jnp.uint8)

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, EnvState]:
        transition, info, state = self.env.reset(key)
        transition = eqx.tree_at(lambda t: t.next_obs, transition, self._upscale(transition.next_obs))
        info = eqx.tree_at(lambda x: x.data, info, walk_and_scale(info.data, self._upscale))
        return transition, info, state

    def step(self, key: PRNGKeyArray, env_state: EnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, EnvState]:
        transition, info, next_state = self.env.step(key, env_state, action)
        transition = eqx.tree_at(lambda t: t.next_obs, transition, self._upscale(transition.next_obs))
        info = eqx.tree_at(lambda x: x.data, info, walk_and_scale(info.data, self._upscale))
        return transition, info, next_state

    @property
    def observation_space(self):
        space = self.env.observation_space
        new_shape = (space.shape[0], *self.target_shape)
        return type(space)(
            low=space.low,
            high=space.high,
            shape=new_shape,
            dtype=space.dtype
        )


class Branched:
    def __init__(self, env: Environment, separated: bool):
        self.env = env
        self.separated = separated

    def reset(self, key: PRNGKeyArray, *, mode: str, **kwargs) -> Tuple[Transition, EnvInfo, EnvState]:
        if self.separated:
            return self.env[mode].reset(key, **kwargs)
        else:
            return self.env.reset(key, **kwargs)

    def step(self, key: PRNGKeyArray, env_state: EnvState, action: jax.Array, *, mode: str) -> Tuple[Transition, EnvInfo, EnvState]:
        if self.separated:
            return self.env[mode].step(key, env_state, action)
        else:
            return self.env.step(key, env_state, action)

    def get(self, name: str, mode: str = "train"):
        if name == "env":
            raise AttributeError
        target_env = self.env[mode] if self.separated else self.env
        return getattr(target_env, name)
