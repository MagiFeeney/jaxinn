import abc
import numpy as np
from typing import Tuple, Any, Union

import jax
import jax.numpy as jnp
import equinox as eqx

from jaxinn.structs import Transition


class HostRAM:
    def __init__(self, num_seeds: int, capacity: Union[int, Tuple[int, ...]], obs_shape: Tuple[int, ...], action_size: int):
        capacity = (capacity,) if isinstance(capacity, int) else capacity
        self.data = Transition(
            action=np.zeros((num_seeds, *capacity, action_size), dtype=np.float32),
            next_obs=np.zeros((num_seeds, *capacity, *obs_shape), dtype=np.uint8 if len(obs_shape) >= 3 else np.float32),
            reward=np.zeros((num_seeds, *capacity, 1), dtype=np.float32),
            done=np.zeros((num_seeds, *capacity, 1), dtype=bool)
        )
        self.length = capacity[0]

    def write(self, seed_idx: jax.Array, index: np.ndarray, transition: Transition):
        token = np.zeros_like(seed_idx, dtype=np.int32) # dummy output
        mask = index != self.length
        valid_index = index[mask]
        seed_idx = seed_idx.reshape(-1, *(1,) * (valid_index.ndim - 1))
        jax.tree.map(lambda arr, new: arr.__setitem__((seed_idx, valid_index), new[mask]), self.data, transition)
        return token

    def read(self, seed_idx: jax.Array, sample_index: np.ndarray) -> Transition:
        seed_idx = seed_idx.reshape(-1, *(1,) * (sample_index.ndim - 1))
        return jax.tree.map(lambda x: x[seed_idx, sample_index], self.data)


class Storage(eqx.Module):
    @abc.abstractmethod
    def write(self, seed_idx: jax.Array, index: jax.Array, transition: Transition) -> Tuple["Storage", jax.Array]:
        pass

    @abc.abstractmethod
    def read(self, seed_idx: jax.Array, sample_index: jax.Array) -> Transition:
        pass

    def __getattr__(self, name):
        return getattr(self.data, name)


class CPUStorage(Storage):
    host: HostRAM = eqx.field(static=True)
    num_seeds: int = eqx.field(static=True)
    base_struct: Any = eqx.field(static=True)

    def __init__(self, num_seeds: int, capacity: int, obs_shape: Tuple[int, ...], action_size: int):
        self.host = HostRAM(num_seeds, capacity, obs_shape, action_size)
        self.num_seeds = num_seeds

        # Precompute to save compute
        self.base_struct = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[2:], x.dtype),
            self.host.data
        )

    @property
    def data(self):
        return self.host.data

    def write(self, seed_idx: jax.Array, index: jax.Array, transition: Transition):
        dummy_shape = jax.ShapeDtypeStruct((), jnp.int32)
        token = jax.pure_callback(
            self.host.write, dummy_shape,
            seed_idx, index, transition,
            vmap_method="broadcast_all"
        )
        return self, token

    def read(self, seed_idx: jax.Array, sample_index: jax.Array):
        expected_struct = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct((*sample_index.shape, *x.shape), x.dtype),
            self.base_struct
        )
        return jax.pure_callback(
            self.host.read, expected_struct,
            seed_idx, sample_index,
            vmap_method="broadcast_all"
        )


class GPUStorage(Storage):
    data: Transition

    def __init__(self, capacity: Union[int, Tuple[int, ...]], obs_shape: Tuple[int, ...], action_size: int):
        capacity = (capacity,) if isinstance(capacity, int) else capacity
        self.data = Transition(
            action=jnp.zeros((*capacity, action_size), dtype=jnp.float32),
            next_obs=jnp.zeros((*capacity, *obs_shape), dtype=jnp.uint8 if len(obs_shape) >= 3 else jnp.float32),
            reward=jnp.zeros((*capacity, 1), dtype=jnp.float32), # TODO: automatically decide whether to unsqueeze based on the env
            done=jnp.zeros((*capacity, 1), dtype=bool)
        )

    def write(self, seed_idx: jax.Array, index: jax.Array, transition: Transition):
        new_data = jax.tree.map(
            lambda buf, batch: buf.at[index].set(batch, mode="drop"),
            self.data, transition
        )
        return eqx.tree_at(lambda s: s.data, self, new_data), jnp.int32(0)

    def read(self, seed_idx: jax.Array, sample_index: jax.Array):
        return jax.tree.map(
            lambda x: x[sample_index],
            self.data
        )
