import abc
from typing import Any, Tuple, Dict, Optional

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx

from jaxinn.structs import Experience
from jaxinn.agent.memory import Memory
from jaxinn.agent.registry import Registrable

from .utils import flatten_time_major, replenish_terminal_obs


class Agent(Registrable, eqx.Module):

    @abc.abstractmethod
    def init_latent_state(self, key: PRNGKeyArray, batch_shape: Tuple[int, ...] = (), eval=False) -> Any:
        pass

    @abc.abstractmethod
    def act(self, last_latent_state: Any, last_action: jax.Array, obs: jax.Array, *, key: PRNGKeyArray, eval: bool = False) -> Tuple[Any, jax.Array]:
        pass

    @abc.abstractmethod
    def learn(self, key: PRNGKeyArray) -> Tuple["Agent", Dict[str, jax.Array]]:
        pass

    @abc.abstractmethod
    def make_batch_fn(self) -> callable:
        pass

    def add_experience(
            self,
            experiences: Experience,
            last_experience: Optional[Experience] = None,
            source: int = 1,
            is_serial: bool = True,
    ) -> "Agent":
        if self.memory is None:
            return self

        if isinstance(self.memory, Memory):
            capacity = self.memory.capacity
        elif isinstance(self.memory, Experience):
            capacity = self.memory.transition.reward.shape[:-1]
        else:
            raise NotImplementedError

        target_ndim = 1 if isinstance(capacity, int) else len(capacity)
        experiences_flatten = jax.tree.map(
            lambda x: flatten_time_major(x, source, target_ndim, collapse=is_serial),
            experiences
        )

        if target_ndim == 1:
            transitions_flatten, valid_length = replenish_terminal_obs(experiences_flatten) # handle terminal obs; critical for world modeling e.g. predict reward
            new_memory = self.memory.add(transitions_flatten, valid_length)
        else:
            if last_experience is not None:
                _source = 1 if is_serial else source
                last_experience_flatten = jax.tree.map(
                    lambda x: flatten_time_major(
                        jnp.expand_dims(x, axis=_source - 1),
                        _source,
                        target_ndim,
                        collapse=False
                    ),
                    last_experience
                )
                experiences_flatten = jax.tree.map(
                    lambda x, y: jnp.concatenate([x, y], axis=0),
                    experiences_flatten,
                    last_experience_flatten
                )

            new_memory = experiences_flatten

        return eqx.tree_at(
            lambda x: x.memory,
            self,
            new_memory
        )
