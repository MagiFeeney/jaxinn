from dataclasses import dataclass

from .base import Base


@dataclass
class InitializerConfig(Base):
    pass


@dataclass
class Constant(InitializerConfig):
    value: float = 0.0


@dataclass
class OrthogonalConfig(InitializerConfig):
    scale: float = 1.0
    column_axis: int = -1


@dataclass
class DeltaOrthogonalConfig(OrthogonalConfig):
    pass


@dataclass
class TruncatedNormalConfig(InitializerConfig):
    stddev: float = 0.01
    lower: float = -2.0
    upper: float = 2.0


@dataclass
class AxesConfig(Base):
    in_axis: int = -2
    out_axis: int = -1
    batch_axis: tuple[int, ...] = ()


@dataclass
class LecunNormalConfig(AxesConfig):
    pass


@dataclass
class LecunUniformConfig(AxesConfig):
    pass


@dataclass
class XavierNormalConfig(AxesConfig):
    pass


@dataclass
class XavierUniformConfig(AxesConfig):
    pass


@dataclass
class HeNormalConfig(AxesConfig):
    pass


@dataclass
class HeUniformConfig(AxesConfig):
    pass
