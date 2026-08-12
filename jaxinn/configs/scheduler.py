from dataclasses import dataclass, field

from .base import Base


@dataclass
class LearningRateSchedulerConfig(Base):
    pass


@dataclass
class ConstantScheduleConfig(LearningRateSchedulerConfig):
    value: float


@dataclass
class LinearScheduleConfig(LearningRateSchedulerConfig):
    init_value: float
    end_value: float
    transition_steps: int


@dataclass
class StaircaseScheduleConfig(LearningRateSchedulerConfig):
    init_value: float
    num_iterations: int
    updates_per_iteration: int


@dataclass
class CosineDecayScheduleConfig(LearningRateSchedulerConfig):
    init_value: float
    decay_steps: int
    alpha: float = 0.0
    exponent: float = 1.0


@dataclass
class CosineOnecycleScheduleConfig(LearningRateSchedulerConfig):
    transition_steps: int
    peak_value: float
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 10000.0


@dataclass
class ExponentialDecayScheduleConfig(LearningRateSchedulerConfig):
    init_value: float
    transition_steps: int
    decay_rate: float
    transition_begin: int = 0
    staircase: bool = False
    end_value: float | None = None


@dataclass
class PolynomialScheduleConfig(LearningRateSchedulerConfig):
    init_value: float
    end_value: float
    power: float
    transition_steps: int
    transition_begin: int = 0


@dataclass
class WarmupCosineDecayScheduleConfig(LearningRateSchedulerConfig):
    init_value: float
    peak_value: float
    warmup_steps: int
    decay_steps: int
    end_value: float = 0.0
    exponent: float = 1.0


@dataclass
class PiecewiseConstantScheduleConfig(LearningRateSchedulerConfig):
    init_value: float
    boundaries_and_scales: dict[int, float] | None = None


@dataclass
class JoinSchedulesConfig(LearningRateSchedulerConfig):
    schedules: list[LearningRateSchedulerConfig, ...] = field(default_factory=list)
    boundaries: list[int, ...] = field(default_factory=list)


LearningRateSchedulerUnion = ConstantScheduleConfig | LinearScheduleConfig | StaircaseScheduleConfig | CosineDecayScheduleConfig | CosineOnecycleScheduleConfig | ExponentialDecayScheduleConfig | PolynomialScheduleConfig | WarmupCosineDecayScheduleConfig | PiecewiseConstantScheduleConfig | JoinSchedulesConfig
