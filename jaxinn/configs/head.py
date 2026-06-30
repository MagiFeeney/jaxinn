from dataclasses import dataclass, field
from typing import Union, Optional

from jaxtyping import Float, Array

from .base import Base


@dataclass
class HeadConfig(Base):
    pass


class NormalHeadConfig(HeadConfig):
    state_dependent_std: bool = True
    constant_std: bool = False
    softplus_std: bool = True
    init_log_std: float = 0.0
    min_std: float = 0.0


@dataclass
class IsotropicNormalHeadConfig(HeadConfig):
    state_dependent_std: bool = False
    constant_std: bool = True
    softplus_std: bool = False
    init_log_std: float = 0.0
    min_std: float = 0.0


@dataclass
class ExpNormalHeadConfig(HeadConfig):
    state_dependent_std: bool = True
    constant_std: bool = False
    softplus_std: bool = False
    init_log_std: float = 0.0
    min_std: float = 0.0


@dataclass
class FreeStdNormalHeadConfig(HeadConfig):
    state_dependent_std: bool = False
    constant_std: bool = False
    softplus_std: bool = False
    init_log_std: float = 0.0
    min_std: float = 0.0


class TanhNormalHeadConfig(HeadConfig):
    min_std: float = 1e-4
    init_std: float = 5.0
    mean_scale: float = 5.0


class BetaHeadConfig(HeadConfig):
    min_std: float = 1e-4


class CategoricalHeadConfig(HeadConfig):
    pass


class OneHotCategoricalHeadConfig(CategoricalHeadConfig):
    pass


class MultiCategoricalHeadConfig(CategoricalHeadConfig):
    nvec: Optional[Float[Array]] = field(default=None, init=False)


HeadUnion = Union[
    NormalHeadConfig, IsotropicNormalHeadConfig, ExpNormalHeadConfig, FreeStdNormalHeadConfig, TanhNormalHeadConfig, BetaHeadConfig, CategoricalHeadConfig, OneHotCategoricalHeadConfig, MultiCategoricalHeadConfig
]

ContinuousHeadUnion = Union[
    NormalHeadConfig, IsotropicNormalHeadConfig, ExpNormalHeadConfig,
    FreeStdNormalHeadConfig, TanhNormalHeadConfig, BetaHeadConfig,
]

DiscreteHeadUnion = Union[
    CategoricalHeadConfig, OneHotCategoricalHeadConfig, MultiCategoricalHeadConfig
]
