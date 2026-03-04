import abc
import numpy as np
from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from envs import Transition


class HostRAM:
    def __init__(self, num_seeds: int, capacity: int, obs_shape: Tuple[int, ...], action_size: int):
        self.data = Transition(
            action=np.zeros((num_seeds, capacity, action_size), dtype=np.float32),
            next_obs=np.zeros((num_seeds, capacity, *obs_shape), dtype=np.uint8),
            reward=np.zeros((num_seeds, capacity), dtype=np.float32),
            done=np.zeros((num_seeds, capacity), dtype=bool)
        )

    def write(self, index: np.ndarray, transition: Transition):
        num_seeds = index.shape[0]
        seed_idx = np.arange(num_seeds)[:, None]

        jax.tree.map(lambda arr, new: arr.__setitem__((seed_idx, index), new), self.data, transition)
        return np.zeros(num_seeds, dtype=np.int32) # dummy output

    def read(self, sample_index: np.ndarray) -> Transition:
        num_seeds = sample_index.shape[0]              # Shape: (num_seeds, batch_size, chunk_size)
        seed_idx = np.arange(num_seeds)[:, None, None] # Shape: (num_seeds, 1, 1)

        return jax.tree.map(lambda x: x[seed_idx, sample_index], self.data)


class Storage(eqx.Module):
    @abc.abstractmethod
    def write(self, index: jax.Array, transition: Transition) -> Tuple["Storage", jax.Array]:
        pass

    @abc.abstractmethod
    def read(self, sample_index: jax.Array) -> Transition:
        pass

    def __getattr__(self, name):
        return getattr(self.data, name)


class CPUStorage(Storage):
    host: HostRAM = eqx.field(static=True)
    num_seeds: int = eqx.field(static=True)

    def __init__(self, num_seeds: int, capacity: int, obs_shape: Tuple[int, ...], action_size: int):
        self.host = HostRAM(num_seeds, capacity, obs_shape, action_size)
        self.num_seeds = num_seeds

    @property
    def data(self):
        return self.host.data

    def write(self, index: jax.Array, transition: Transition):
        dummy_shape = jax.ShapeDtypeStruct((), jnp.int32)
        token = jax.pure_callback(
            self.host.write, dummy_shape,
            index, transition,
            vmap_method="broadcast_all"
        )
        return self, token

    def read(self, sample_index: jax.Array):
        batch_size, chunk_size = sample_index.shape
        expected_shape = (self.num_seeds, batch_size, chunk_size)

        expected_struct = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct((*expected_shape, *x.shape[2:]), x.dtype),
            self.host.data
        )

        return jax.pure_callback(
            self.host.read, expected_struct,
            sample_index,
            vmap_method="broadcast_all"
        )


class GPUStorage(Storage):
    data: Transition

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_size: int):
        self.data = Transition(
            action=jnp.zeros((capacity, action_size), dtype=jnp.float32),
            next_obs=jnp.zeros((capacity, *obs_shape), dtype=jnp.uint8),
            reward=jnp.zeros(capacity, dtype=jnp.float32),
            done=jnp.zeros(capacity, dtype=bool)
        )

    def write(self, index: jax.Array, transition: Transition):
        new_data = jax.tree.map(
            lambda buf, batch: buf.at[index].set(batch),
            self.data, transition
        )
        return eqx.tree_at(lambda s: s.data, self, new_data), jnp.int32(0)

    def read(self, sample_index: jax.Array):
        return jax.tree.map(
            lambda x: x[sample_index],
            self.data
        )
