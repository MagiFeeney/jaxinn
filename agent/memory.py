import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from jaxtyping import PRNGKeyArray
from ..train import Transition


class Memory(eqx.Module):
    data: Transition
    ptr: jax.Array
    size: jax.Array
    capacity: int = eqx.field(static=True)

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_size: int):
        self.capacity = capacity
        # Pre-allocate
        self.data = Transition(
            action=jnp.empty((capacity, action_size)),
            next_obs=jnp.empty((capacity, *obs_shape)).astype(jnp.uint8), # For memory efficiency
            reward=jnp.empty(capacity),
            done=jnp.empty(capacity),
        )
        self.ptr = jnp.array(0)
        self.size = jnp.array(0)

    def __getattr__(self, name):
        return getattr(self.data, name)

    @property
    def full(self):
        return self.size == self.capacity

    def add(self, transition: Transition):
        """Adds a single or a batch of transitions to the buffer."""
        batch_size = jax.tree.leaves(transition)[0].shape[0]
        indices = (self.ptr + jnp.arange(batch_size)) % self.capacity

        # Write new data
        new_data = jax.tree.map(
            lambda buf, batch: buf.at[indices].set(batch),
            self.data, transitions
        )

        # Update pointer and size
        new_ptr = (self.ptr + batch_size) % self.capacity
        new_size = jnp.minimum(self.size + batch_size, self.capacity)

        return eqx.tree_at(
            lambda b: (b.data, b.ptr, b.size),
            self,
            (new_data, new_ptr, new_size)
        )

    def sample(self, batch_size: int, key: PRNGKeyArray):
        """Samples a batch of transitions."""
        sample_index = jax.random.randint(key, (batch_size,), 0, self.size)

        batch = jax.tree.map(
            lambda x: x[sample_index],
            self.data
        )
        return batch

    def sample_trajectory(self, batch_size: int, chunk_size: int, key: PRNGKeyArray):
        """Samples a batch of trajectories with equal length."""
        # Prevent overshooting due to wrapping
        def _get_candidates_full():
            def _wrap_case():
                # ptr < chunk_size - 1: can't sample from [0, ptr) without going negative
                return jnp.arange(self.ptr, self.capacity - (chunk_size - 1 - self.ptr))

            def _normal_case():
                # ptr >= chunk_size - 1: can sample normally from [0, ptr - chunk_size + 1)
                # AND from [ptr, capacity) as additional candidates
                start_range = jnp.arange(0, self.ptr - chunk_size + 1) # Okay to be null
                extra_range = jnp.arange(self.ptr, self.capacity)
                return jnp.concatenate([start_range, extra_range])

            return jax.lax.cond(
                self.ptr < chunk_size - 1,
                _wrap_case,
                _normal_case
            )

        def _get_candidates_not_full():
            # Assumes ptr > chunk_size so have at least one index to sample from.
            # This holds as we append a trajectory whose size is far larger than required
            return jnp.arange(0, self.ptr - chunk_size + 1)

        batch_sample_candidates = jax.lax.cond(
            self.full,
            _get_candidates_full,
            _get_candidates_not_full
        )

        batch_index = jax.random.choice(key, batch_sample_candidates, shape=(batch_size,), replace=True)
        offset = jnp.arange(chunk_size)
        sample_index = offset[:, None] + batch_index[None, :] # T x B

        trajectories = jax.tree.map(
            lambda x: x[sample_index],
            self.data
        )
        return trajectories
