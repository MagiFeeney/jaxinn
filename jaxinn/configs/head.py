from dataclasses import dataclass, field
from typing import Any

from jaxtyping import Float, Array

from .base import Base


@dataclass
class HeadConfig(Base):
    pass


@dataclass
class ComplexHeadConfig(HeadConfig):
    data: Any

    def __call__(self) -> Any:
        return self.data


@dataclass
class DictHeadConfig(ComplexHeadConfig):
    data: dict[str, HeadConfig]

    def __post_init__(self) -> None:
        if not isinstance(self.data, dict):
            raise TypeError(f"Expected dict for DictHeadConfig, got {type(self.data)}.")


@dataclass
class TupleHeadConfig(ComplexHeadConfig):
    data: tuple[HeadConfig, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.data, tuple):
            raise TypeError(f"Expected tuple for TupleHeadConfig, got {type(self.data)}.")


@dataclass
class HierarchicalHeadConfig(DictHeadConfig):
    data: dict[HeadConfig, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.data, dict):
            raise TypeError(f"Expected dict for HierarchicalHeadConfig, got {type(self.data)}.")

        if "option" not in self.data or "actions" not in self.data:
            raise ValueError("HierarchicalHeadConfig must exactly contain 'option' and 'actions' keys.")

        if not isinstance(self.data["option"], CategoricalHeadConfig):
            raise TypeError(f"The 'option' head config must be a CategoricalHeadConfig but got {type(self.data['option'])}.")

        if not isinstance(self.data["actions"], dict):
            raise TypeError(f"The 'actions' head config must be a dict but got {type(self.data['actions'])}.")


@dataclass
class DeterministicHeadConfig(HeadConfig):
    pass


@dataclass
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


@dataclass
class TanhNormalHeadConfig(HeadConfig):
    min_std: float = 1e-4
    init_std: float = 5.0
    mean_scale: float | None = 5.0
    log_std_range: tuple[int, int] | None = None


@dataclass
class BetaHeadConfig(HeadConfig):
    min_std: float = 1e-4


@dataclass
class CategoricalHeadConfig(HeadConfig):
    pass


@dataclass
class OneHotCategoricalHeadConfig(CategoricalHeadConfig):
    pass


@dataclass
class MultiCategoricalHeadConfig(CategoricalHeadConfig):
    nvec: Float[Array, "..."] | None = field(default=None)
    variable_shape: tuple[int, ...] = ()


HeadUnion = DeterministicHeadConfig | NormalHeadConfig | IsotropicNormalHeadConfig | ExpNormalHeadConfig | FreeStdNormalHeadConfig | TanhNormalHeadConfig | BetaHeadConfig | CategoricalHeadConfig | OneHotCategoricalHeadConfig | MultiCategoricalHeadConfig

ContinuousHeadUnion = DeterministicHeadConfig | NormalHeadConfig | IsotropicNormalHeadConfig | ExpNormalHeadConfig | FreeStdNormalHeadConfig | TanhNormalHeadConfig | BetaHeadConfig

DiscreteHeadUnion = CategoricalHeadConfig | OneHotCategoricalHeadConfig | MultiCategoricalHeadConfig
