import abc
from typing import Union

import jax
import equinox as eqx

from jaxinn.structs import LatentState
from jaxinn.agent.registry import Registrable

from ..distributions import DistributionLike


class Encoder(Registrable, eqx.Module):
    @abc.abstractmethod
    def __call__(self, obs: jax.Array) -> jax.Array:
        pass


class Decoder(Registrable, eqx.Module):
    @abc.abstractmethod
    def __call__(self, latent: Union[jax.Array, LatentState]) -> Union[DistributionLike, jax.Array]:
        pass
