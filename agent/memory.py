import abc
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Union
from jaxtyping import PRNGKeyArray
from envs import Transition


# Base class
class Memory(eqx.Module):
    data: Transition
    ptr: jax.Array
    size: jax.Array
    capacity: int = eqx.field(static=True)

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_size: int):
        self.capacity = capacity
        # Pre-allocate
        self.data = Transition( # TODO: if there is performance gap, replace with zeros
            action=jnp.empty((capacity, action_size)),
            next_obs=jnp.empty((capacity, *obs_shape), dtype=jnp.uint8), # For memory efficiency
            reward=jnp.empty(capacity),
            done=jnp.empty(capacity, dtype=bool),
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
    def add(self, transition: Transition, valid_length: Union[int, jax.Array] | None = None):
        """Adds a single or a batch of transitions to the buffer."""
        batch_size = jax.tree.leaves(transition)[0].shape[0]
        index = (self.ptr + jnp.arange(batch_size)) % self.capacity

        # Write new data
        new_data = jax.tree.map(
            lambda buf, batch: buf.at[index].set(batch),
            self.data, transition
        )

        # Get actual num. of data added
        if valid_length is not None:
            num_data = valid_length
        else:
            num_data = batch_size

        # Update pointer and size
        new_ptr = (self.ptr + num_data) % self.capacity
        new_size = jnp.minimum(self.size + num_data, self.capacity)

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
        sample_index = (offset[:, None] + batch_index[None, :]) % self.capacity # T x B

        trajectories = jax.tree.map(
            lambda x: x[sample_index],
            self.data
        )
        return trajectories

    def sample_batch_index(self, batch_size: int, key: PRNGKeyArray, *, chunk_size: int):
        start = jnp.where(
            self.full,
            self.ptr - self.capacity, # Prevent overshooting due to wrapping; if negative, turn non-contiguous intervals into a contiguous one, facilitating sampling efficiency
            0
        )
        end = self.ptr - chunk_size + 1

        batch_index = jax.random.randint(key, (batch_size,), start, end) % self.capacity # Equivalence between negative interval [-m, -1] to [N - m, N - 1] under modulo
        return batch_index


# Data structure for more efficient sampling w.r.t. priority
class SumTree(eqx.Module):
    tree: jax.Array
    masked_priorities: jax.Array                  # Priorities of last chunk (to prevent overshooting)
    active_leaves: int = eqx.field(static=True)   # Requested capacity; no need to be power of 2
    capacity: int = eqx.field(static=True)        # Real capacity of power of 2
    depth: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)

    def __init__(self, capacity: int, chunk_size: int):
        self.active_leaves = capacity
        self.depth = int(jnp.ceil(jnp.log2(capacity)))
        self.capacity = 2 ** self.depth
        self.tree = jnp.zeros(2 * self.capacity - 1)

        self.chunk_size = chunk_size
        self.masked_priorities = jnp.zeros(chunk_size - 1) # Storing priorities in last chunk [ptr - chunk_size + 1, ptr - 1]

    def update(self, start_index: int, priorities: jax.Array) -> "SumTree":
        batch_size = priorities.shape[0]
        trailing_size = self.chunk_size - 1

        # Include both masked and new priorities
        start = start_index - trailing_size
        offset = trailing_size + batch_size

        # To recover the last masked priorities
        leaf_indices = (start + jnp.arange(offset)) % self.active_leaves
        tree_indices = leaf_indices + self.capacity - 1

        merged_priorities = jnp.concatenate([self.masked_priorities, priorities])
        new_masked_priorities = merged_priorities[-trailing_size:] # Keep track of the new masked priorities
        merged_priorities = merged_priorities.at[-trailing_size:].set(0.) # Mask out the forbidden sampling regime
        deltas = merged_priorities - self.tree[tree_indices]
        new_tree = self.tree.at[tree_indices].set(merged_priorities)

        curr_index = tree_indices
        # For small depth, for loop is preferred
        for _ in range(self.depth):
            curr_index = (curr_index - 1) // 2
            new_tree = new_tree.at[curr_index].add(deltas)

        return eqx.tree_at(lambda t: (t.tree, t.masked_priorities), self, (new_tree, new_masked_priorities))

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
        val = self.tree[final_idx] # Priorities for debugging if needed
        return leaf_idx, val


# Memory with prioritized sampling
class Prioritized(Uniform):
    sumtree: SumTree
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)

    # Require chunk_size to be given because we want to handle the overshooting during settling priorities
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_size: int, chunk_size: int, alpha=0.6, beta=0.4):
        super().__init__(capacity, obs_shape, action_size)
        self.alpha = alpha
        self.beta = beta
        self.sumtree = SumTree(capacity, chunk_size)

    def add(self, transition: Transition, priority: jax.Array, valid_length: Union[int, jax.Array] | None = None):
        """Adds a single or a batch of transitions to the buffer."""
        # Add priority first to use old parameters
        # If valid_length is not None, priority of data after that should zero
        new_tree = self.sumtree.update(self.ptr, priority**self.alpha)
        new_memory = super().add(transition, valid_length) # updated with new data, ptr and size

        # Update additional subtree of new priority
        return eqx.tree_at(
            lambda m: m.sumtree,
            new_memory,
            new_tree
        )

    def sample_batch_index(self, batch_size: int, key: PRNGKeyArray, *, chunk_size: int): # chunk_size not used, just to align with base
        batch_index, _ = self.sumtree.sample(batch_size, key)
        return batch_index
