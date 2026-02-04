import math
import jax
import jax.numpy as jnp
from typing import Any, Callable, Tuple
from jaxtyping import PRNGKeyArray
import equinox as eqx

from envs.environment import Transition, Environment, EnvInfo


class Wrapper(Environment):
    """Base class for JAX environment wrappers."""

    def __init__(self, env: Environment):
        self.env = env

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

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError
        return getattr(self.env, name)


class Batched(Wrapper):
    num_envs: int = eqx.field(static=True)
    vmap_reset: Callable = eqx.field(static=True) # TODO: StaticCallable?
    vmap_step: Callable = eqx.field(static=True)

    def __init__(self, env: Environment, num_envs: int):
        self.env = env

        self.num_envs = num_envs
        self.vmap_reset = jax.vmap(self.env.reset)
        self.vmap_step = jax.vmap(self.env.step)

    def reset(self, key: PRNGKeyArray, num_envs: int | None = None) -> Tuple[Transition, EnvInfo, EnvState]:
        num_keys = num_envs if num_envs is not None else self.num_envs
        keys = jax.random.split(key, num_keys)
        return self.vmap_reset(keys)

    def step(self, key: PRNGKeyArray, env_state: EnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, EnvState]:
        num_keys = action.shape[0] # Infer from action since we have already known that at reset
        keys = jax.random.split(key, num_keys)
        return self.vmap_step(keys, env_state, action)


class AutoReset(Wrapper):
    def __init__(self, env: Environment):
        self.env = env

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, EnvState]:
        return self.env.reset(key)

    def step(self, key: PRNGKeyArray, env_state: EnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, EnvState]:
        key_step, key_reset = jax.random.split(key)
        step_transition, step_env_info, step_env_state = self.env.step(key_step, env_state, action)
        done = step_transition.done

        def do_reset():
            return self.env.reset(key_reset)

        def no_reset():
            return step_transition, step_env_info, step_env_state

        # lax.cond executes only one branch (on single device) or handles masking (in vmap)
        reset_transition, reset_env_info, reset_env_state = jax.lax.cond(
            done,
            do_reset,
            no_reset
        )                       # potentially saves computation for rendering when done is False

        final_env_state = jax.tree_map(
            lambda r, s: jnp.where(done, r, s),
            reset_env_state,
            step_env_state
        )
        _next_obs = jax.tree_map(
            lambda r, s: jnp.where(done, r, s),
            reset_transition.next_obs,
            step_transition.next_obs
        )
        final_transition = eqx.tree_at(lambda t: t.next_obs, step_transition, _next_obs)
        final_info = EnvInfo(**info.data, terminal_observation=step_transition.next_obs)
        return final_transition, final_info, final_env_state
