import abc

import jax

from jaxinn.common.structs import LatentState
from jaxinn.agent.registry import Registrable

from ..base import Model
from ..distributions import DistributionLike


class Encoder(Registrable, Model):
    @abc.abstractmethod
    def __call__(self, obs: jax.Array) -> jax.Array:
        pass


class Decoder(Registrable, Model):
    @abc.abstractmethod
    def __call__(self, latent: jax.Array | LatentState) -> DistributionLike | jax.Array:
        pass
