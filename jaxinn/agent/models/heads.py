import abc
import math
from typing import Tuple, Optional, ClassVar, Type

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx
from equinox._module import Static
import distrax

from jaxinn.agent.registry import Registrable
from jaxinn.configs.head import (
    NormalHeadConfig,
    IsotropicNormalHeadConfig,
    ExpNormalHeadConfig,
    FreeStdNormalHeadConfig,
    TanhNormalHeadConfig,
    BetaHeadConfig,
    CategoricalHeadConfig,
    OneHotCategoricalHeadConfig,
)

from .distributions import SampleDist, TanhNormal, AffineBeta
from .utils import dx


class Head(Registrable, eqx.Module):
    param_size: eqx.AbstractVar[int]

    @abc.abstractmethod
    def __call__(self, x: jax.Array) -> distrax.Distribution:
        pass

    def sample(self, dist: distrax.Distribution, key: PRNGKeyArray, det: bool = False) -> jax.Array:
        if det:
            return dist.mode(seed=key)
        return dist.sample(seed=key)


class NormalHead(Head):
    config_cls: ClassVar[Tuple[Type, ...]] = (
        NormalHeadConfig,
        IsotropicNormalHeadConfig,
        ExpNormalHeadConfig,
        FreeStdNormalHeadConfig
    )

    log_std: Optional[jax.Array]

    param_size: int = eqx.field(static=True)
    state_dependent_std: bool = eqx.field(static=True)
    constant_std: bool = eqx.field(static=True)
    softplus_std: bool = eqx.field(static=True)
    min_std: float = eqx.field(static=True)

    def __init__(
        self,
        event_size: int,
        state_dependent_std: bool = True,
        constant_std: bool = False,
        softplus_std: bool = True,
        init_log_std: float = 0.0,
        min_std: float = 0.0,
    ):
        if state_dependent_std:
            self.param_size = 2 * event_size
            self.log_std = None
        else:
            self.param_size = event_size
            if constant_std:
                self.log_std = init_log_std
            else:
                self.log_std = jnp.full((event_size,), init_log_std)

        self.state_dependent_std = state_dependent_std
        self.constant_std = constant_std
        self.softplus_std = softplus_std
        self.min_std = min_std

    def __call__(self, x: jax.Array) -> dx.Distribution:
        if self.state_dependent_std:
            mean, log_std = jnp.split(x, 2, axis=-1)
        else:
            mean = x
            log_std = self.log_std

        if self.softplus_std:
            std = jax.nn.softplus(log_std) + self.min_std
        else:
            std = jnp.exp(log_std)

        dist = dx.Independent(dx.Normal(loc=mean, scale=std), reinterpreted_batch_ndims=Static(1))
        return dist

    def sample(self, dist: distrax.Distribution, key: PRNGKeyArray, det: bool = False) -> jax.Array:
        if det:
            return dist.mode()
        return dist.sample(seed=key)


class TanhNormalHead(Head):
    config_cls: ClassVar[Type] = TanhNormalHeadConfig

    param_size: int = eqx.field(static=True)
    min_std: float = eqx.field(static=True)
    mean_scale: float = eqx.field(static=True)
    raw_init_std: float = eqx.field(static=True)

    def __init__(
        self,
        event_size: int,
        min_std: float = 1e-4,
        init_std: float = 5.0,
        mean_scale: float = 5.0
    ):
        self.param_size = 2 * event_size

        self.min_std = min_std
        self.mean_scale = mean_scale
        self.raw_init_std = math.log(math.exp(init_std) - 1)

    def __call__(self, x: jax.Array) -> dx.Distribution:
        mean, log_std = jnp.split(x, 2, axis=-1)
        mean = self.mean_scale * jnp.tanh(mean / self.mean_scale)
        std = jax.nn.softplus(log_std + self.raw_init_std) + self.min_std

        dist = dx.Independent(dx(TanhNormal)(mean=mean, std=std), reinterpreted_batch_ndims=Static(1))
        return SampleDist(dist)


class BetaHead(Head):
    config_cls: ClassVar[Type] = BetaHeadConfig

    param_size: int = eqx.field(static=True)
    min_std: float = eqx.field(static=True)

    def __init__(
        self,
        event_size: int,
        min_std: float = 1e-4,
    ):
        self.param_size = 2 * event_size

        self.min_std = min_std

    def __call__(self, x: jax.Array) -> dx.Distribution:
        alpha_beta = jax.nn.softplus(x) + self.min_std
        alpha, beta = jnp.split(alpha_beta, 2, axis=-1)

        dist = dx.Independent(dx(AffineBeta)(alpha=alpha, beta=beta), reinterpreted_batch_ndims=Static(1))
        return SampleDist(dist)


class CategoricalHead(Head):
    config_cls: ClassVar[Type] = CategoricalHeadConfig

    param_size: int = eqx.field(static=True)
    event_size: tuple = eqx.field(static=True)

    def __init__(
        self,
        event_size: int | tuple,
    ):
        self.event_size = (event_size, ) if isinstance(event_size, int) else event_size

        self.param_size = math.prod(self.event_size)

    def __call__(self, x: jax.Array) -> dx.Distribution:
        logits = x.reshape(*x.shape[:-1], *self.event_size)
        dist = dx.Categorical(logits=logits)
        return dist

    def sample(self, dist: distrax.Distribution, key: PRNGKeyArray, det: bool = False) -> jax.Array:
        sample = dist.sample(seed=key)
        sample = sample.reshape(*sample.shape[:-len(self.event_size) + 1], -1)
        return sample


class OneHotCategoricalHead(CategoricalHead):
    config_cls: ClassVar[Type] = OneHotCategoricalHeadConfig

    def __call__(self, x: jax.Array) -> dx.Distribution:
        logits = x.reshape(*x.shape[:-1], *self.event_size)
        dist = dx.Independent(dx.OneHotCategorical(logits=logits), reinterpreted_batch_ndims=len(self.event_size) - 1) # TODO: whether reduce the dims before event
        return dist

    def sample(self, dist: distrax.Distribution, key: PRNGKeyArray, det: bool = False) -> jax.Array: # TODO: use mode for eval?
        sample = dist.sample(seed=key)
        sample = sample + dist.probs - jax.lax.stop_gradient(dist.probs)
        sample = sample.reshape(*sample.shape[:-len(self.event_size)], -1)
        return sample


class Orchestrator(eqx.Module):
    pass
