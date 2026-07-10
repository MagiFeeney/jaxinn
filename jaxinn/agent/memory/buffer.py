import abc
from typing import Tuple, Union

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree, DTypeLike
import equinox as eqx

from jaxinn.structs import Transition

from .storage import Storage, CPUStorage, GPUStorage


# Base class
class Memory(eqx.Module):
    storage: Storage
    seed_idx: jax.Array   # Unique id to anchor data for multiple seeds
    ptr: jax.Array
    size: jax.Array
    capacity: Union[int, Tuple[int, ...]] = eqx.field(static=True)

    def __init__(
            self,
            seed_idx: jax.Array,
            capacity: Union[int, Tuple[int, ...]],
            obs_shape: PyTree[Tuple[int, ...]],
            obs_dtype: PyTree[DTypeLike],
            action_shape: PyTree[Tuple[int, ...]],
            action_dtype: PyTree[DTypeLike],
            num_seeds: int | None = None
    ):
        self.seed_idx = jnp.array(seed_idx, dtype=jnp.int32)
        self.capacity = capacity
        # Initialize data on either the CPU or GPU, depending on the memory requirements of the task
        if num_seeds is not None:
            self.storage = CPUStorage(num_seeds, capacity, obs_shape, obs_dtype, action_shape, action_dtype)
        else:
            self.storage = GPUStorage(capacity, obs_shape, obs_dtype, action_shape, action_dtype) # vmap automatically handle multiple seeds
        self.ptr = jnp.array(0)
        self.size = jnp.array(0)

    def __getattr__(self, name):
        return getattr(self.storage, name)

    @property
    def length(self):
        return self.capacity if isinstance(self.capacity, int) else self.capacity[0]

    @property
    def full(self):
        return self.size == self.length

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
    def add(self, transition: Transition, valid_length: jax.Array | None = None):
        """Adds a single or a batch of transitions to the buffer."""
        batch_size = jax.tree.leaves(transition)[0].shape[0]

        # Get actual num. of data added
        num_data = batch_size if valid_length is None else valid_length

        # Get valid indices
        mask = jnp.arange(batch_size) < num_data
        index = (self.ptr + jnp.arange(batch_size)) % self.length
        valid_index = jnp.where(mask, index, self.length)

        # Write new data
        new_storage, token = self.storage.write(self.seed_idx, valid_index, transition)

        # Update pointer and size; token is used to prevent xla "Dead code elimination"
        new_ptr = (self.ptr + num_data + token) % self.length
        new_size = jnp.minimum(self.size + num_data, self.length)

        return eqx.tree_at(
            lambda m: (m.storage, m.ptr, m.size),
            self,
            (new_storage, new_ptr, new_size)
        )

    def sample(self, sample_shape: Tuple[int, ...], key: PRNGKeyArray):
        """Samples a batch of trajectories with equal length."""
        batch_size, chunk_size = sample_shape

        batch_index = self.sample_batch_index(batch_size, key, chunk_size=chunk_size)
        offset = jnp.arange(chunk_size)
        sample_index = (offset[:, None] + batch_index[None, :]) % self.length # T x B

        return self.storage.read(self.seed_idx, sample_index)

    def sample_batch_index(self, batch_size: int, key: PRNGKeyArray, *, chunk_size: int):
        start = jnp.where(
            self.full,
            self.ptr - self.length, # Prevent overshooting due to wrapping; if negative, turn non-contiguous intervals into a contiguous one, facilitating sampling efficiency
            0
        )
        end = self.ptr - chunk_size + 1

        batch_index = jax.random.randint(key, (batch_size,), start, end) % self.length # Equivalence between negative interval [-m, -1] to [N - m, N - 1] under modulo
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
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_size: int, chunk_size: int, num_seeds: int | None = None, alpha=0.6, beta=0.4):
        if not isinstance(capacity, int):
            raise TypeError(f"Capacity for Prioritized memory only supports integer types, but got {type(capacity).__name__}.")
        super().__init__(capacity, obs_shape, action_size, num_seeds)
        self.alpha = alpha
        self.beta = beta
        self.sumtree = SumTree(capacity, chunk_size)

    def add(self, transition: Transition, priority: jax.Array, valid_length: jax.Array | None = None):
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


class Batched(Uniform):
    def __init__(
            self,
            seed_idx: jax.Array,
            capacity: Union[int, Tuple[int, ...]],
            obs_shape: PyTree[Tuple[int, ...]],
            obs_dtype: PyTree[DTypeLike],
            action_shape: PyTree[Tuple[int, ...]],
            action_dtype: PyTree[DTypeLike],
            num_seeds: int | None = None
    ):
        super().__init__(seed_idx, capacity, obs_shape, obs_dtype, action_shape, action_dtype, num_seeds)

    def sample_batch_index(self, batch_size: int, key: PRNGKeyArray):
        return super().sample_batch_index(batch_size, key, chunk_size=1)

    def get_all(self) -> Transition:
        read_index = jnp.arange(self.length)
        batch = self.storage.read(self.seed_idx, read_index)
        return batch
