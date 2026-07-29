import jax
import jax.numpy as jnp
import equinox as eqx

from jaxinn.structs import Transition, Experience


def transform(obs: jax.Array) -> jax.Array:
    if obs.dtype == jnp.uint8 and obs.ndim > 3:
        return obs.astype(jnp.float32) / 255.0 - 0.5
    return obs


def flatten_time_major(
        x: jax.Array,
        source: int,
        target_ndim: int = 1,
        collapse: bool = False
) -> jax.Array:
    """
    Swaps the `source` axis with the preceding axis to ensure correct layout of trajectories, then flattens all leading dimensions according to `target_ndim`, with additional checks on RGB inputs for memory efficiency. Finally, rearranges the retained dimensions into a time-major ordering.

    If `collapse` is True, flattens the sequence/batch dims into a single time axis.
    """
    if collapse:
        x = x.reshape(-1, *x.shape[source:])
        source = 1

    num_leading_dims = source + 1

    if num_leading_dims < target_ndim:
        raise ValueError(
            f"Cannot flatten to {target_ndim} dims: source index {source} "
            f"(representing {num_leading_dims} dims) is too small."
        )

    end_dim = num_leading_dims - target_ndim + 1

    # Swap the source axis with the preceding axis
    # e.g., (T, E, ...) -> (E, T, ...) or (N, T, E, ...) -> (N, E, T, ...)
    swapped_x = jnp.moveaxis(x, source=source, destination=source - 1)

    flattened = swapped_x.reshape(-1, *swapped_x.shape[end_dim:])

    # For storage
    if flattened.dtype == jnp.float32 and flattened.ndim > 3:
        is_normalized = flattened.max() <= 1.0
        flattened = jax.lax.cond(
            is_normalized,
            lambda arr: (arr * 255.0).astype(jnp.uint8), # recover for storage
            lambda arr: arr.astype(jnp.uint8),
            flattened
        )

    # time-major
    time_axis = num_leading_dims - end_dim
    time_major_x = jnp.moveaxis(flattened, source=time_axis, destination=0)
    return time_major_x


def replenish_boundary_obs(experiences: Experience) -> tuple[Transition, jax.Array]:
    if experiences.boundary_obs is None:
        return experiences.transition, None

    mask = experiences.transition.terminated | experiences.transition.truncated
    N = mask.shape[0]

    # Indices for step transitions; we replenish ones at done = True with boundary_obs
    shifts = jnp.concatenate([
        jnp.zeros((1,) + mask.shape[1:], dtype=bool),
        mask[:-1]
    ], axis=0)
    step_indices = jnp.arange(N) + jnp.cumsum(shifts)

    # Indices for reset transitions
    reset_indices = step_indices + 1 # To keep shape static; only indices at mask are meaningful

    # Construct reset transitions
    reset_transitions = jax.tree.map(jnp.zeros_like, experiences.transition)
    reset_transitions = eqx.tree_at(
        lambda x: x.next_obs,
        reset_transitions,
        experiences.transition.next_obs
    )

    # Replenish boundary_obs
    mask_expanded = mask.reshape((N,) + (1,) * (experiences.boundary_obs.ndim - 1))
    new_next_obs = jnp.where(
        mask_expanded,
        experiences.boundary_obs,
        experiences.transition.next_obs
    )
    step_transitions = eqx.tree_at(
        lambda x: x.next_obs,
        experiences.transition,
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
        terminated: jax.Array,
        next_values: jax.Array,
        discount_factor: float = 0.99,
        uae_lambda: float = 0.95,
) -> tuple[jax.Array, jax.Array]:
    def uae_step_fn(uae, inputs):
        """
        Unified advantage estimator (UAE): a generalized version of GAE.

        When baseline function is zero, it reduces to λ-return.

        Reference: https://arxiv.org/pdf/2302.00533
        """
        reward, value, baseline, next_value, term = inputs

        delta = (
            reward
            + discount_factor * next_value * (1 - term)
            - baseline
        )
        z = value - baseline
        discounted_uae = discount_factor * uae_lambda * (1 - term) * uae
        advantage = delta + discounted_uae
        uae = advantage - z
        return uae, advantage

    uae = jnp.zeros_like(values[0])

    _, advantages = jax.lax.scan(
        uae_step_fn,
        uae,
        (rewards, values, baselines, next_values, terminated),
        reverse=True,
    )
    return_predictions = advantages + baselines
    return advantages, return_predictions


def staircase_lr_schedule(init_lr, num_iterations, updates_per_iteration):
    def schedule(count):
        current_iteration = count // updates_per_iteration
        frac = 1.0 - (current_iteration / num_iterations)
        return init_lr * jnp.maximum(frac, 0.0)
    return schedule
