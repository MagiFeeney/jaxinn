import abc
from typing import ClassVar

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from jaxinn.configs.scheduler import (
    LearningRateSchedulerConfig,
    ConstantScheduleConfig,
    LinearScheduleConfig,
    StaircaseScheduleConfig,
    CosineDecayScheduleConfig,
    CosineOnecycleScheduleConfig,
    ExponentialDecayScheduleConfig,
    PolynomialScheduleConfig,
    WarmupCosineDecayScheduleConfig,
    PiecewiseConstantScheduleConfig,
    JoinSchedulesConfig
)
from jaxinn.agent.registry import Registrable


class LearningRateScheduler(Registrable, eqx.Module):
    @abc.abstractmethod
    def __call__(self, step: int) -> jax.Array:
        pass


class ConstantSchedule(LearningRateScheduler):
    config_cls: ClassVar[type] = ConstantScheduleConfig

    value: float = eqx.field(static=True)

    def __call__(self, step: int) -> jax.Array:
        return optax.schedules.constant_schedule(self.value)(step)


class LinearSchedule(LearningRateScheduler):
    config_cls: ClassVar[type] = LinearScheduleConfig

    init_value: float = eqx.field(static=True)
    end_value: float = eqx.field(static=True)
    transition_steps: int = eqx.field(static=True)

    def __call__(self, step: int) -> jax.Array:
        schedule_fn = optax.schedules.linear_schedule(
            self.init_value, self.end_value, self.transition_steps
        )
        return schedule_fn(step)


class StaircaseSchedule(LearningRateScheduler):
    config_cls: ClassVar[type] = StaircaseScheduleConfig

    init_value: float = eqx.field(static=True)
    num_iterations: int = eqx.field(static=True)
    updates_per_iteration: int = eqx.field(static=True)

    def __call__(self, step: int) -> jax.Array:
        current_iteration = step // self.updates_per_iteration
        frac = 1.0 - (current_iteration / self.num_iterations)
        return self.init_value * jnp.maximum(frac, 0.0)


class CosineDecaySchedule(LearningRateScheduler):
    config_cls: ClassVar[type] = CosineDecayScheduleConfig

    init_value: float = eqx.field(static=True)
    decay_steps: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True, default=0.0)
    exponent: float = eqx.field(static=True, default=1.0)

    def __call__(self, step: int) -> jax.Array:
        schedule_fn = optax.schedules.cosine_decay_schedule(
            self.init_value, self.decay_steps, self.alpha, self.exponent
        )
        return schedule_fn(step)


class CosineOnecycleSchedule(LearningRateScheduler):
    config_cls: ClassVar[type] = CosineOnecycleScheduleConfig

    transition_steps: int = eqx.field(static=True)
    peak_value: float = eqx.field(static=True)
    pct_start: float = eqx.field(static=True, default=0.3)
    div_factor: float = eqx.field(static=True, default=25.0)
    final_div_factor: float = eqx.field(static=True, default=10000.0)

    def __call__(self, step: int) -> jax.Array:
        schedule_fn = optax.schedules.cosine_onecycle_schedule(
            self.transition_steps,
            self.peak_value,
            self.pct_start,
            self.div_factor,
            self.final_div_factor,
        )
        return schedule_fn(step)


class ExponentialDecaySchedule(LearningRateScheduler):
    config_cls: ClassVar[type] = ExponentialDecayScheduleConfig

    init_value: float = eqx.field(static=True)
    transition_steps: int = eqx.field(static=True)
    decay_rate: float = eqx.field(static=True)
    transition_begin: int = eqx.field(static=True, default=0)
    staircase: bool = eqx.field(static=True, default=False)
    end_value: float | None = eqx.field(static=True, default=None)

    def __call__(self, step: int) -> jax.Array:
        schedule_fn = optax.schedules.exponential_decay(
            self.init_value,
            self.transition_steps,
            self.decay_rate,
            self.transition_begin,
            self.staircase,
            self.end_value,
        )
        return schedule_fn(step)


class PolynomialSchedule(LearningRateScheduler):
    config_cls: ClassVar[type] = PolynomialScheduleConfig

    init_value: float = eqx.field(static=True)
    end_value: float = eqx.field(static=True)
    power: float = eqx.field(static=True)
    transition_steps: int = eqx.field(static=True)
    transition_begin: int = eqx.field(static=True, default=0)

    def __call__(self, step: int) -> jax.Array:
        schedule_fn = optax.schedules.polynomial_schedule(
            self.init_value,
            self.end_value,
            self.power,
            self.transition_steps,
            self.transition_begin,
        )
        return schedule_fn(step)


class WarmupCosineDecaySchedule(LearningRateScheduler):
    config_cls: ClassVar[type] = WarmupCosineDecayScheduleConfig

    init_value: float = eqx.field(static=True)
    peak_value: float = eqx.field(static=True)
    warmup_steps: int = eqx.field(static=True)
    decay_steps: int = eqx.field(static=True)
    end_value: float = eqx.field(static=True, default=0.0)
    exponent: float = eqx.field(static=True, default=1.0)

    def __call__(self, step: int) -> jax.Array:
        schedule_fn = optax.schedules.warmup_cosine_decay_schedule(
            self.init_value,
            self.peak_value,
            self.warmup_steps,
            self.decay_steps,
            self.end_value,
            self.exponent,
        )
        return schedule_fn(step)


class PiecewiseConstantSchedule(LearningRateScheduler):
    config_cls: ClassVar[type] = PiecewiseConstantScheduleConfig

    init_value: float = eqx.field(static=True)
    boundaries_and_scales: dict[int, float] | None = eqx.field(static=True, default=None)

    def __call__(self, step: int) -> jax.Array:
        schedule_fn = optax.schedules.piecewise_constant_schedule(
            self.init_value, self.boundaries_and_scales
        )
        return schedule_fn(step)


class JoinSchedules(LearningRateScheduler):
    config_cls: ClassVar[type] = JoinSchedulesConfig

    schedules: tuple[LearningRateScheduler, ...]
    boundaries: tuple[int, ...]

    def __init__(self, schedules: list[LearningRateSchedulerConfig], boundaries: list[int]):
        self.schedules = tuple(LearningRateScheduler.create(cfg) for cfg in schedules)
        self.boundaries = tuple(boundaries)

    def __call__(self, step: int) -> jax.Array:
        schedule_fn = optax.join_schedules(
            schedules=self.schedules,
            boundaries=self.boundaries
        )
        return schedule_fn(step)
