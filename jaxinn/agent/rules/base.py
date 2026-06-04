import abc
from typing import Any, Tuple, Dict

import jax
import equinox as eqx

from envs import Transition


class Experience(eqx.Module):
    transition: Transition
    terminal_observation: jax.Array


class Agent(eqx.Module):
    @abc.abstractmethod
    def init_state(self, key: PRNGKeyArray, batch_shape: Tuple[int, ...] = (), eval=False) -> Any:
        pass

    @abc.abstractmethod
    def act(self, last_latent_state: LatentState, last_action: jax.Array, obs: jax.Array, *, key: PRNGKeyArray, eval: bool = False) -> Tuple[Any, jax.Array]:
        pass

    @abc.abstractmethod
    def add_experience(self, experiences: Experience, source: int = 1) -> "Agent":
        pass

    @abc.abstractmethod
    def learn(self, key: PRNGKeyArray) -> Tuple["Agent", Dict[str, jax.Array]]:
        pass
