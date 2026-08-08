import abc
from typing import ClassVar, Literal, Any

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx

from jaxinn.configs.initializer import (
    ConstantConfig,
    OrthogonalConfig,
    DeltaOrthogonalConfig,
    HeNormalConfig,
    HeUniformConfig,
    LecunNormalConfig,
    LecunUniformConfig,
    XavierNormalConfig,
    XavierUniformConfig,
    TruncatedNormalConfig,
)

from .registry import Registrable


class Initializer(Registrable, eqx.Module):
    @abc.abstractmethod
    def __call__(self, key: PRNGKeyArray, shape: tuple[int, ...], dtype: Any = jnp.float32) -> jax.Array:
        pass


class Constant(Initializer):
    config_cls: ClassVar[type] = ConstantConfig

    value: float = eqx.field(static=True, default=0.0)

    def __call__(self, key: PRNGKeyArray, shape: tuple[int, ...], dtype: Any = jnp.float32) -> jax.Array:
        return jax.nn.initializers.constant(self.value)(key, shape, dtype)


class Orthogonal(Initializer):
    config_cls: ClassVar[type] = OrthogonalConfig

    scale: float = eqx.field(static=True, default=1.0)
    column_axis: int = eqx.field(static=True, default=-1)

    def __call__(self, key: PRNGKeyArray, shape: tuple[int, ...], dtype: Any = jnp.float32) -> jax.Array:
        return jax.nn.initializers.orthogonal(self.scale, self.column_axis)(key, shape, dtype)


class DeltaOrthogonal(Orthogonal):
    config_cls: ClassVar[type] = DeltaOrthogonalConfig

    def __call__(self, key: PRNGKeyArray, shape: tuple[int, ...], dtype: Any = jnp.float32) -> jax.Array:
        return jax.nn.initializers.delta_orthogonal(self.scale, self.column_axis)(key, shape, dtype)


class TruncatedNormal(Initializer):
    config_cls: ClassVar[type] = TruncatedNormalConfig

    stddev: float = eqx.field(static=True, default=0.01)
    lower: float = eqx.field(static=True, default=-2.0)
    upper: float = eqx.field(static=True, default=2.0)

    def __call__(self, key: PRNGKeyArray, shape: tuple[int, ...], dtype: Any = jnp.float32) -> jax.Array:
        init_fn = jax.nn.initializers.truncated_normal(
            stddev=self.stddev,
            lower=self.lower,
            upper=self.upper
        )
        return init_fn(key, shape, dtype)


class VarianceScaling(Initializer):
    scale: float = eqx.field(static=True)
    mode: Literal['fan_in', 'fan_out', 'fan_avg', 'fan_geo_avg'] = eqx.field(static=True)
    distribution: Literal['truncated_normal', 'normal', 'uniform'] = eqx.field(static=True)

    in_axis: int = eqx.field(static=True, default=-2)
    out_axis: int = eqx.field(static=True, default=-1)
    batch_axis: tuple[int, ...] = eqx.field(static=True, default=())

    def __call__(self, key: PRNGKeyArray, shape: tuple[int, ...], dtype: Any = jnp.float32) -> jax.Array:
        init_fn = jax.nn.initializers.variance_scaling(
            scale=self.scale,
            mode=self.mode,
            distribution=self.distribution,
            in_axis=self.in_axis,
            out_axis=self.out_axis,
            batch_axis=self.batch_axis
        )
        return init_fn(key, shape, dtype)


class LecunNormal(VarianceScaling):
    config_cls: ClassVar[type] = LecunNormalConfig

    scale: float = eqx.field(static=True, default=1.0)
    mode: str = eqx.field(static=True, default="fan_in")
    distribution: str = eqx.field(static=True, default="truncated_normal")


class LecunUniform(VarianceScaling):
    config_cls: ClassVar[type] = LecunUniformConfig

    scale: float = eqx.field(static=True, default=1.0)
    mode: str = eqx.field(static=True, default="fan_in")
    distribution: str = eqx.field(static=True, default="uniform")


class HeNormal(VarianceScaling):
    config_cls: ClassVar[type] = HeNormalConfig

    scale: float = eqx.field(static=True, default=2.0)
    mode: str = eqx.field(static=True, default="fan_in")
    distribution: str = eqx.field(static=True, default="truncated_normal")


class HeUniform(VarianceScaling):
    config_cls: ClassVar[type] = HeUniformConfig

    scale: float = eqx.field(static=True, default=2.0)
    mode: str = eqx.field(static=True, default="fan_in")
    distribution: str = eqx.field(static=True, default="uniform")


class XavierNormal(VarianceScaling):
    config_cls: ClassVar[type] = XavierNormalConfig

    scale: float = eqx.field(static=True, default=1.0)
    mode: str = eqx.field(static=True, default="fan_avg")
    distribution: str = eqx.field(static=True, default="truncated_normal")


class XavierUniform(VarianceScaling):
    config_cls: ClassVar[type] = XavierUniformConfig

    scale: float = eqx.field(static=True, default=1.0)
    mode: str = eqx.field(static=True, default="fan_avg")
    distribution: str = eqx.field(static=True, default="uniform")
