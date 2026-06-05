from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from envs import Transition
from .base import Experience


def transform(obs) -> jax.Array:
    if obs.dtype == jnp.uint8 and obs.ndim > 3:
        return obs.astype(jnp.float32) / 255.0 - 0.5
    return obs


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
