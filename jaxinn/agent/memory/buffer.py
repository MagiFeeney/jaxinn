import abc
from typing import ClassVar

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree, DTypeLike
import equinox as eqx

from jaxinn.common.structs import Transition
from jaxinn.agent.registry import Registrable
from jaxinn.configs.agent.memory import (
    UniformMemoryConfig,
    PrioritizedMemoryConfig,
    BatchedMemoryConfig,
    EpisodicMemoryConfig,
)

from .storage import Storage, CPUStorage, GPUStorage


# Base class
class Memory(Registrable, eqx.Module):
    storage: Storage
    seed_idx: jax.Array   # Unique id to anchor data for multiple seeds
    ptr: jax.Array
    size: jax.Array
    capacity: int | tuple[int, ...] = eqx.field(static=True)

    def __init__(
            self,
            seed_idx: jax.Array,
            capacity: int | tuple[int, ...],
            obs_shape: PyTree[tuple[int, ...]],
            obs_dtype: PyTree[DTypeLike],
            action_shape: PyTree[tuple[int, ...]],
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
    def sample(self, sample_shape: tuple[int, ...], key: PRNGKeyArray):
        pass

    @abc.abstractmethod
    def sample_batch_index(self, batch_size: int, key: PRNGKeyArray):
        pass


# Memory with uniform sampling
class Uniform(Memory):
    config_cls: ClassVar[type] = UniformMemoryConfig

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

    def sample(self, sample_shape: tuple[int, ...], key: PRNGKeyArray):
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

    def sample(self, batch_size: int, key: PRNGKeyArray) -> tuple[jax.Array, jax.Array]:
        total_priority = self.tree[0]
        queries = jax.random.uniform(key, shape=(batch_size,)) * total_priority
        return jax.vmap(self._retrieve)(queries)

    def _retrieve(self, s: float) -> tuple[int, float]:
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
    config_cls: ClassVar[type] = PrioritizedMemoryConfig

    sumtree: SumTree
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)

    # Require chunk_size to be given because we want to handle the overshooting during settling priorities
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], action_size: int, chunk_size: int, num_seeds: int | None = None, alpha=0.6, beta=0.4):
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
    config_cls: ClassVar[type] = BatchedMemoryConfig

    def __init__(
            self,
            seed_idx: jax.Array,
            capacity: int | tuple[int, ...],
            obs_shape: PyTree[tuple[int, ...]],
            obs_dtype: PyTree[DTypeLike],
            action_shape: PyTree[tuple[int, ...]],
            action_dtype: PyTree[DTypeLike],
            num_seeds: int | None = None
    ):
        super().__init__(seed_idx, capacity, obs_shape, obs_dtype, action_shape, action_dtype, num_seeds)

    def sample_batch_index(self, batch_size: int, key: PRNGKeyArray):
        return super().sample_batch_index(batch_size, key, chunk_size=1)[0]

    def get_all(self) -> Transition:
        read_index = jnp.arange(self.length)
        batch = self.storage.read(self.seed_idx, read_index)
        return batch


class Episodic(Memory):
    config_cls: ClassVar[type] = EpisodicMemoryConfig

    episode_ends: jax.Array
    episode_ptr: jax.Array
    episode_tail: jax.Array
    total_steps: jax.Array

    max_sequence_length: int = eqx.field(static=True, default=0) # Use full episode if 0, otherwise cap sequence by the value
    prioritize_ends: bool = eqx.field(static=True, default=True)

    def __init__(
            self,
            seed_idx: jax.Array,
            capacity: int | tuple[int, ...],
            obs_shape: PyTree[tuple[int, ...]],
            obs_dtype: PyTree[DTypeLike],
            action_shape: PyTree[tuple[int, ...]],
            action_dtype: PyTree[DTypeLike],
            num_seeds: int | None = None,
            max_sequence_length: int = 0,
            prioritize_ends: bool = True,
    ):
        super().__init__(seed_idx, capacity, obs_shape, obs_dtype, action_shape, action_dtype, num_seeds)
        self.episode_ends = jnp.full((self.length,), -1, dtype=jnp.int64)
        self.episode_ptr = jnp.array(0, dtype=jnp.int32)
        self.episode_tail = jnp.array(0, dtype=jnp.int32)
        self.total_steps = jnp.array(0, dtype=jnp.int64)

        self.max_sequence_length = max_sequence_length
        self.prioritize_ends = prioritize_ends

    def add(self, transition: Transition, valid_length: jax.Array | None = None):
        # Adding data, the same as Uniform
        batch_size = jax.tree.leaves(transition)[0].shape[0]
        num_data = batch_size if valid_length is None else valid_length
        mask = jnp.arange(batch_size) < num_data
        index = (self.ptr + jnp.arange(batch_size)) % self.length
        valid_index = jnp.where(mask, index, self.length)
        new_storage, token = self.storage.write(self.seed_idx, valid_index, transition)
        new_ptr = (self.ptr + num_data + token) % self.length
        new_size = jnp.minimum(self.size + num_data, self.length)

        # Get which step is end
        is_end = (jnp.ravel(transition.terminated) | jnp.ravel(transition.truncated)) & mask

        INF = jnp.iinfo(jnp.int32).max
        end_batch_idx = jnp.where(is_end, jnp.arange(batch_size), INF)
        sorted_ends = jnp.sort(end_batch_idx)
        num_episodes = is_end.sum()

        absolute_ends = self.total_steps + sorted_ends

        episode_index = (self.episode_ptr + jnp.arange(batch_size)) % self.length
        episode_mask = jnp.arange(batch_size) < num_episodes

        new_episode_ends = jnp.where(episode_mask, absolute_ends, self.episode_ends[episode_index]) # Replace INF with the placeholder value
        updated_episode_ends = self.episode_ends.at[episode_index].set(new_episode_ends)
        new_episode_ptr = (self.episode_ptr + num_episodes) % self.length

        # Prune corrupted episodes
        new_total_steps = self.total_steps + num_data
        oldest_valid_step = new_total_steps - new_size

        def cond_fn(tail):
            is_empty = tail == new_episode_ptr
            ep_start = updated_episode_ends[(tail - 1) % self.length] + 1
            is_valid = ep_start >= oldest_valid_step
            return ~(is_empty | is_valid)

        def body_fn(tail):
            return (tail + 1) % self.length

        new_episode_tail = jax.lax.while_loop(cond_fn, body_fn, self.episode_tail)

        return eqx.tree_at(
            lambda m: (m.storage, m.ptr, m.size, m.episode_ends, m.episode_ptr, m.episode_tail, m.total_steps),
            self,
            (new_storage, new_ptr, new_size, updated_episode_ends, new_episode_ptr, new_episode_tail, new_total_steps)
        )

    def sample(self, sample_shape: tuple[int, ...], key: PRNGKeyArray):
        batch_size, chunk_size = sample_shape
        sample_index, is_stitched = self.sample_batch_index(batch_size, key, chunk_size=chunk_size)
        data = self.storage.read(self.seed_idx, sample_index)

        is_stitched = is_stitched.reshape(data.truncated.shape)
        new_truncated = data.truncated | is_stitched

        return eqx.tree_at(lambda x: x.truncated, data, new_truncated)

    def sample_batch_index(self, batch_size: int, key: PRNGKeyArray, *, chunk_size: int):
        key, key_init = jax.random.split(key, 2)
        init_start, init_end = self.sample_sequence_bounds(key_init, chunk_size)
        init_offset = 0
        init_state = (init_start, init_end, init_offset, key)

        def step_fn(state, _):
            start, end, offset, key = state

            remaining = end - (start + offset)

            key, subkey = jax.random.split(key)
            new_start, new_end = self.sample_sequence_bounds(subkey, chunk_size)

            idx_range = jnp.arange(chunk_size)

            # Consume existing episode
            idx_current = start + offset + idx_range

            # Replenish from the new episode
            # Only effective when remaining < chunk_size
            idx_next = new_start + (idx_range - remaining)

            # Stitch two parts together
            mask = idx_range < remaining
            out_idx = jnp.where(mask, idx_current, idx_next)

            # Update states
            overshoot = remaining < chunk_size
            is_stitched = (idx_range == remaining - 1) & overshoot
            next_start = jnp.where(overshoot, new_start, start)
            next_end = jnp.where(overshoot, new_end, end)
            next_offset = jnp.where(overshoot, chunk_size - remaining, offset + chunk_size)

            next_state = (next_start, next_end, next_offset, key)

            return next_state, (out_idx, is_stitched)

        _, (chunk_indices, chunk_stitches) = jax.lax.scan(
            step_fn, init_state, None, length=batch_size
        )

        return chunk_indices.T % self.length, chunk_stitches.T # T x B

    def sample_episode_bounds(self, sample_shape: tuple[int, ...], key: PRNGKeyArray, use_absolute: bool = True):
        num_valid = (self.episode_ptr - self.episode_tail) % self.length
        num_valid = jnp.where((num_valid == 0) & (self.size > 0), self.length, num_valid)
        num_valid = jnp.maximum(1, num_valid)

        offsets = jax.random.randint(key, sample_shape, minval=0, maxval=num_valid)
        episode_indices = (self.episode_tail + offsets) % self.length

        absolute_starts = self.episode_ends[(episode_indices - 1) % self.length] + 1
        absolute_ends = self.episode_ends[episode_indices]

        if not use_absolute:
            return absolute_starts % self.length, absolute_ends % self.length
        return absolute_starts, absolute_ends

    def sample_sequence_bounds(self, key: PRNGKeyArray, chunk_size: int):
        key_episode, key_desync, key_offset = jax.random.split(key, 3)
        episode_start, episode_end = self.sample_episode_bounds(sample_shape=(), key=key_episode, use_absolute=True)
        episode_length = episode_end - episode_start + 1

        # Desynchronize episodes by randomizing length
        sequence_length = episode_length
        if self.max_sequence_length > 0:
            sequence_length = jnp.minimum(sequence_length, self.max_sequence_length)

        sequence_length -= jax.random.randint(key_desync, (), minval=0, maxval=chunk_size)
        sequence_length = jnp.maximum(chunk_size, sequence_length)

        # Prioritize sequences near end
        offset = jnp.maximum(0, episode_length - sequence_length)
        rand_maxval = offset + 1 + (chunk_size if self.prioritize_ends else 0)
        rand_idx = jax.random.randint(key_offset, (), minval=0, maxval=rand_maxval)
        offset = jnp.minimum(rand_idx, offset)

        seq_start = episode_start + offset
        seq_end = seq_start + jnp.minimum(sequence_length, episode_length)

        return seq_start, seq_end
