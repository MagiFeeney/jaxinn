import abc
import math
from typing import Tuple, Optional, ClassVar, Type

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree, PyTreeDef
import equinox as eqx
from equinox._module import Static
import distrax

from jaxinn.agent.registry import Registrable
from jaxinn.configs.head import (
    HeadConfig,
    NormalHeadConfig,
    IsotropicNormalHeadConfig,
    ExpNormalHeadConfig,
    FreeStdNormalHeadConfig,
    TanhNormalHeadConfig,
    BetaHeadConfig,
    CategoricalHeadConfig,
    OneHotCategoricalHeadConfig,
    MultiCategoricalHeadConfig,
)

from .distributions import (
    SampleDist,
    TanhNormal,
    AffineBeta,
    StraightThroughOneHotCategorical,
    FlattenSampleDist,
    IndependentJointDistribution,
    TreeJointDistribution,
)
from .utils import dx


class Head(Registrable, eqx.Module):
    param_size: eqx.AbstractVar[int]

    @abc.abstractmethod
    def __call__(self, x: jax.Array) -> distrax.Distribution:
        pass


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

    def __call__(self, x: jax.Array) -> distrax.Distribution:
        if self.state_dependent_std:
            mean, log_std = jnp.split(x, 2, axis=-1)
        else:
            mean = x
            log_std = jnp.broadcast_to(self.log_std, mean.shape)

        if self.softplus_std:
            std = jax.nn.softplus(log_std) + self.min_std
        else:
            std = jnp.exp(log_std)

        dist = dx.Independent(dx.Normal(loc=mean, scale=std), reinterpreted_batch_ndims=Static(1))
        return dist


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

    def __call__(self, x: jax.Array) -> distrax.Distribution:
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

    def __call__(self, x: jax.Array) -> distrax.Distribution:
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

    def __call__(self, x: jax.Array) -> distrax.Distribution:
        logits = x.reshape(*x.shape[:-1], *self.event_size)
        dist = dx.Independent(dx.Categorical(logits=logits), reinterpreted_batch_ndims=Static(len(self.event_size) - 1))
        return FlattenSampleDist(dist)


class OneHotCategoricalHead(CategoricalHead):
    config_cls: ClassVar[Type] = OneHotCategoricalHeadConfig

    def __call__(self, x: jax.Array) -> distrax.Distribution:
        logits = x.reshape(*x.shape[:-1], *self.event_size)
        dist = dx.Independent(dx(StraightThroughOneHotCategorical)(logits=logits), reinterpreted_batch_ndims=Static(len(self.event_size) - 1))
        return FlattenSampleDist(dist)


class MultiCategoricalHead(CategoricalHead):
    config_cls: ClassVar[Type] = MultiCategoricalHeadConfig

    heads: Tuple[CategoricalHead, ...]
    nvec_shape: Tuple[int, ...] = eqx.field(static=True)
    split_points: Tuple[int, ...] = eqx.field(static=True)
    variable_shape: Tuple[int, ...] = eqx.field(static=True)

    def __init__(self, event_size: int, nvec: jax.Array, variable_shape: Tuple[int, ...] = ()):
        super().__init__(event_size)

        self.nvec_shape = nvec.shape
        self.variable_shape = variable_shape

        np_nvec = np.array(nvec)
        flat_nvec = np.ravel(np_nvec)
        self.split_points = tuple(np.cumsum(flat_nvec)[:-1].tolist())

        self.heads = tuple(
            CategoricalHead(event_size=variable_shape + (int(n),))
            for n in flat_nvec
        )

    def __call__(self, x: jax.Array) -> IndependentJointDistribution:
        logits = jnp.split(x, self.split_points, axis=-1)
        dists = tuple(head(l) for head, l in zip(self.heads, logits))
        return IndependentJointDistribution(
            dists=dists,
            target_shape=self.nvec_shape
        )


class TreeHead(Head):
    heads_tree: PyTree[Head]
    heads_treedef: PyTreeDef = eqx.field(static=True)
    param_size: int = eqx.field(static=True)
    split_points: Tuple[int, ...] = eqx.field(static=True)
    is_leaf: callable = eqx.field(static=True)

    @classmethod
    def create(cls, head_config: PyTree[HeadConfig], **kwargs):
        event_size = kwargs.pop("event_size", None)

        if event_size is None:
            raise ValueError("event_size cannot be None for creating heads.")

        heads_tree = jax.tree.map(
            lambda config, size: Head.create(config, event_size=size),
            head_config,
            event_size
        )

        is_leaf = lambda x: isinstance(x, Head)

        param_size_tree = jax.tree.map(
            lambda h: h.param_size,
            heads_tree,
            is_leaf=is_leaf
        )
        flat_param_size, treedef = jax.tree.flatten(param_size_tree)
        split_points = tuple(np.cumsum(np.array(flat_param_size))[:-1].tolist())
        param_size = int(sum(flat_param_size))

        return cls(
            heads_tree=heads_tree,
            heads_treedef=treedef,
            param_size=param_size,
            split_points=split_points,
            is_leaf=is_leaf,
        )

    def __call__(self, x: jax.Array) -> TreeJointDistribution:
        params = jnp.split(x, self.split_points, axis=-1)
        params_tree = jax.tree.unflatten(self.heads_treedef, params)
        dists_tree = jax.tree.map(
            lambda h, p: h(p),
            self.heads_tree,
            params_tree,
            is_leaf=self.is_leaf
        )
        return TreeJointDistribution(dists_tree=dists_tree)

    def sample(self, dist: TreeJointDistribution, key: PRNGKeyArray, det: bool = False) -> PyTree[jax.Array]:
        num_leaves = self.heads_treedef.num_leaves
        keys = jax.random.split(key, num_leaves)
        keys_tree = jax.tree.unflatten(self.heads_treedef, keys)
        return jax.tree.map(
            lambda h, d, k: h.sample(d, key=k, det=det),
            self.heads_tree,
            dist.dists_tree,
            keys_tree,
            is_leaf=self.is_leaf
        )
