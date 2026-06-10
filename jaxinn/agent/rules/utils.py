from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from jaxinn.structs import Transition, Experience


def transform(obs) -> jax.Array:
    if obs.dtype == jnp.uint8 and obs.ndim > 3:
        return obs.astype(jnp.float32) / 255.0 - 0.5
    return obs


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


def compute_adv_and_ret(
        rewards: jax.Array,
        values: jax.Array,
        baselines: jax.Array,
        dones: jax.Array,
        bootstrap: jax.Array,
        discount_factor: float = 0.99,
        uae_lambda: float = 0.95,
) -> Tuple[jax.Array, jax.Array]:

    def uae_step_fn(carry, inputs):
        """
        Unified advantage estimator (UAE): a generalized version of GAE.

        When baseline function is zero, it reduces to λ-return.

        Reference: https://arxiv.org/pdf/2302.00533
        """
        uae, next_value = carry
        reward, value, baseline, done = inputs

        delta = (
            reward
            + discount_factor * next_value * (1 - done)
            - baseline
        )
        z = value - baseline
        discounted_uae = discount_factor * uae_lambda * (1 - done) * uae
        advantage = delta + discounted_uae
        uae = (delta - z) + discounted_uae
        return (uae, value), advantage

    uae = jnp.zeros_like(bootstrap)
    input_carry = (uae, bootstrap)

    _, advantages = jax.lax.scan(
        uae_step_fn,
        input_carry,
        (rewards, values, baselines, dones),
        reverse=True,
    )
    return_predictions = advantages + baselines
    return advantages, return_predictions
