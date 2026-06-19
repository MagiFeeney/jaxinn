import abc
from typing import Any, Tuple, Dict

import jax
from jaxtyping import PRNGKeyArray
import equinox as eqx

from jaxinn.structs import Experience
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

    def add_experience(self, experiences: Experience, source: int = 1) -> "Agent":
        if self.memory is None:
            return self

        target_dim = 1 if isinstance(self.memory.capacity, int) else len(self.memory.capacity)
        experiences_flatten = jax.tree.map(
            lambda x: flatten_time_major(x, source, target_dim),
            experiences
        )
        transitions_flatten, valid_length = replenish_terminal_obs(experiences_flatten) # handle terminal obs; critical for world modeling e.g. predict reward

        new_memory = self.memory.add(transitions_flatten, valid_length)
        return eqx.tree_at(
            lambda x: x.memory,
            self,
            new_memory
        )
