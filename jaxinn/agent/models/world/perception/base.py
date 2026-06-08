import abc
from typing import Union

import jax
import equinox as eqx
import distrax

from jaxinn.agent.registry import Registrable

from ..primitives import LatentState


class Encoder(Registrable, eqx.Module):
    @abc.abstractmethod
    def __call__(self, obs: jax.Array) -> jax.Array:
        pass


class Decoder(Registrable, eqx.Module):
    @abc.abstractmethod
    def __call__(self, latent: Union[jax.Array, LatentState]) -> Union[distrax.Distribution, jax.Array]:
        pass
