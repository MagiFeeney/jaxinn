import abc
from typing import Any, Tuple, Dict

import jax
from jaxtyping import PRNGKeyArray
import equinox as eqx

from envs import Transition


class Experience(eqx.Module):
    transition: Transition
    terminal_observation: jax.Array


def replenish_and_flatten(experiences: Experience, source: int) -> Tuple[Transition, jax.Array]:
    def flatten_fn(x):
        # (T, B, ...) -> (B*T, ...)
        flattened = jnp.moveaxis(x, source=source, destination=source - 1).reshape(-1, *x.shape[source + 1:])
        # For storage
        if flattened.dtype == jnp.float32 and flattened.ndim > 3:
            is_normalized = flattened.max() <= 1.0
            return jax.lax.cond(
                is_normalized,
                lambda arr: (arr * 255.0).astype(jnp.uint8), # recover for storage
                lambda arr: arr.astype(jnp.uint8),
                flattened
            )
        return flattened

    # flatten and cast dtype in one go
    transitions_flatten = jax.tree.map(flatten_fn, experiences.transition)

    mask = transitions_flatten.done
    N = mask.shape[0]

    if experiences.terminal_observation is None:
        return transitions_flatten, None

    terminal_obs_flatten = flatten_fn(experiences.terminal_observation)

    # Indices for step transitions; we replenish ones at done = True with terminal_obs
    shifts = jnp.concatenate([
        jnp.zeros((1,) + mask.shape[1:], dtype=bool),
        mask[:-1]
    ], axis=0)
    step_indices = jnp.arange(N) + jnp.cumsum(shifts)

    # Indices for reset transitions
    reset_indices = step_indices + 1 # To keep shape static; only indices at mask are meaningful

    # Construct reset transitions
    reset_transitions = jax.tree.map(jnp.zeros_like, transitions_flatten)
    reset_transitions = eqx.tree_at(
        lambda x: x.next_obs,
        reset_transitions,
        transitions_flatten.next_obs
    )

    # Replenish terminal_obs
    mask_expanded = mask.reshape((N,) + (1,) * (terminal_obs_flatten.ndim - 1))
    new_next_obs = jnp.where(
        mask_expanded,
        terminal_obs_flatten,
        transitions_flatten.next_obs
    )
    step_transitions = eqx.tree_at(
        lambda x: x.next_obs,
        transitions_flatten,
        new_next_obs
    )

    padded_length = 2 * N
    # Create the merged empty array
    def merge_fn(step_leaf, reset_leaf):
        new_shape = (padded_length, *step_leaf.shape[1:]) # Fixed length for jit
        out = jnp.zeros(new_shape, dtype=step_leaf.dtype)
        out = out.at[step_indices].set(step_leaf)
        # Trick
        _reset_indices = jnp.where(mask, reset_indices, padded_length)
        out = out.at[_reset_indices].set(reset_leaf, mode='drop')
        return out

    valid_length = N + jnp.sum(mask) # Actual length
    return jax.tree.map(merge_fn, step_transitions, reset_transitions), valid_length


class Agent(eqx.Module):

    @abc.abstractmethod
    def init_state(self, key: PRNGKeyArray, batch_shape: Tuple[int, ...] = (), eval=False) -> Any:
        pass

    @abc.abstractmethod
    def act(self, last_latent_state: Any, last_action: jax.Array, obs: jax.Array, *, key: PRNGKeyArray, eval: bool = False) -> Tuple[Any, jax.Array]:
        pass

    @abc.abstractmethod
    def learn(self, key: PRNGKeyArray) -> Tuple["Agent", Dict[str, jax.Array]]:
        pass

    def add_experience(self, experiences: Experience, source: int = 1) -> "Agent":
        if self.memory is None:
            return self
        transitions_flatten, valid_length = replenish_and_flatten(experiences, source) # handle terminal obs; critical for world modeling e.g. predict reward
        new_memory = self.memory.add(transitions_flatten, valid_length)
        return eqx.tree_at(
            lambda x: x.memory,
            self,
            new_memory
        )
