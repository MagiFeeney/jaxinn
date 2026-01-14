import abc
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from jaxtyping import PRNGKeyArray
from ..train import Transition


# Base class
class Memory(eqx.Module):
    data: Transition
    ptr: jax.Array
    size: jax.Array
    capacity: int = eqx.field(static=True)

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_dim: int):
        self.capacity = capacity
        # Pre-allocate
        self.data = Transition(
            action=jnp.empty((capacity, action_dim)),
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

    @abc.abstractmethod
    def add(self, transition: Transition):
        pass

    @abc.abstractmethod
    def sample(self, sample_shape: Tuple[int, ...], key: PRNGKeyArray):
        pass

    @abc.abstractmethod
    def sample_batch_index(self, batch_size: int, key: PRNGKeyArray):
        pass


# Memory with uniform sampling
class Uniform(Memory):
    def add(self, transition: Transition):
        """Adds a single or a batch of transitions to the buffer."""
        batch_size = jax.tree.leaves(transition)[0].shape[0]
        index = (self.ptr + jnp.arange(batch_size)) % self.capacity

        # Write new data
        new_data = jax.tree.map(
            lambda buf, batch: buf.at[index].set(batch),
            self.data, transition
        )

        # Update pointer and size
        new_ptr = (self.ptr + batch_size) % self.capacity
        new_size = jnp.minimum(self.size + batch_size, self.capacity)

        return eqx.tree_at(
            lambda m: (m.data, m.ptr, m.size),
            self,
            (new_data, new_ptr, new_size)
        )

    def sample(self, sample_shape: Tuple[int, ...], key: PRNGKeyArray):
        """Samples a batch of trajectories with equal length."""
        batch_size, chunk_size = sample_shape

        batch_index = self.sample_batch_index(batch_size, key, chunk_size=chunk_size)
        offset = jnp.arange(chunk_size)
        sample_index = offset[:, None] + batch_index[None, :] # T x B

        trajectories = jax.tree.map(
            lambda x: x[sample_index],
            self.data
        )
        return trajectories

    def sample_batch_index(self, batch_size: int, key: PRNGKeyArray, *, chunk_size: int):
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
        return batch_index


# Data structure for more efficient sampling w.r.t. priority
class SumTree(eqx.Module):
    tree: jax.Array
    active_leaves: int = eqx.field(static=True)   # Requested capacity; no need to be power of 2
    capacity: int = eqx.field(static=True)        # Real capacity of power of 2
    depth: int = eqx.field(static=True)

    def __init__(self, active_leaves: int):
        self.active_leaves = capacity
        self.depth = int(jnp.ceil(jnp.log2(capacity)))
        self.capacity = 2 ** self.depth

        self.tree = jnp.zeros(2 * self.capacity - 1)

    def update(self, start_index: int, priorities: jax.Array) -> "SumTree":
        batch_size = priorities.shape[0]
        leaf_indices = (start_index + jnp.arange(batch_size)) % self.active_leaves
        tree_indices = leaf_indices + self.capacity - 1

        deltas = priorities - self.tree[tree_indices]
        new_tree = self.tree.at[tree_indices].set(priorities)

        curr_index = tree_indices
        # For small depth, for loop is preferred
        for _ in range(self.depth):
            curr_index = (curr_index - 1) // 2
            new_tree = new_tree.at[curr_index].add(deltas)

        return eqx.tree_at(lambda t: t.tree, self, new_tree)

    def sample(self, batch_size: int, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
        total_priority = self.tree[0]
        queries = jax.random.uniform(key, shape=(batch_size,)) * total_priority
        return jax.vmap(self._retrieve)(queries)

    def _retrieve(self, s: float) -> Tuple[int, float]:
        idx = 0

        def body_fn(_, state):
            i, s_val = state
            left = 2 * i + 1
            right = left + 1

            left_val = self.tree[left]

            go_right = s_val > left_val

            next_i = jnp.where(go_right, right, left)
            next_s = jnp.where(go_right, s_val - left_val, s_val)
            return (next_i, next_s)

        final_idx, _ = jax.lax.fori_loop(0, self.depth, body_fn, (idx, s))
        leaf_idx = final_idx - (self.capacity - 1)
        val = self.tree[final_idx] # priorities For debugging
        return leaf_idx, val


# Memory with prioritized sampling
class Prioritized(Uniform):
    sumtree: SumTree
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_dim: int, alpha=0.6, beta=0.4):
        super().__init__(capacity, obs_shape, action_dim)
        self.alpha = alpha
        self.beta = beta
        self.sumtree = SumTree(capacity)

    def add(self, transition: Transition, priority: jax.Array):
        """Adds a single or a batch of transitions to the buffer."""
        # Add priority first to use old parameters
        new_tree = self.sumtree.update(self.ptr, priority ** self.alpha)
        new_memory = super().add(transition) # updated with new data, ptr and size

        # Update additional subtree of new priority
        return eqx.tree_at(
            lambda m: m.sumtree,
            new_memory,
            new_tree
        )

    def sample_batch_index(self, batch_size: int, key: PRNGKeyArray): # TODO: redefine
        return self.sumtree.sample(batch_size, key) # TODO: fix the overshooting
