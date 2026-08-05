import abc
from typing import Any
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, DTypeLike
import equinox as eqx


class Transform(eqx.Module, abc.ABC):
    """Base for transformations over variables."""

    @abc.abstractmethod
    def __call__(self, x: Any) -> Any:
        """Applies the forward transformation given current state."""
        pass

    def update(self, x: Any) -> "Transform":
        """Updates internal state based on new data."""
        return self

    def inverse(self, x: Any) -> Any:
        """Applies the inverse transformation if mathematically defined."""
        return x


class Stateless(Transform):
    _forward: Callable = eqx.field(static=True)
    _inverse: Callable | None = eqx.field(static=True)

    def __init__(self, forward: Callable, inverse: Callable | None = None):
        self._forward = forward
        self._inverse = inverse

    def __call__(self, x: Any) -> Any:
        return jax.tree.map(self._forward, x)

    def update(self, x: Any) -> "Stateless":
        return self

    def inverse(self, x: Any) -> Any:
        if self._inverse is None:
            return x
        return jax.tree.map(self._inverse, x)


class Chain(Transform):
    """Composes multiple Transforms sequentially."""

    transforms: tuple[Transform, ...]

    def __call__(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)
        return x

    def update(self, x: Any) -> "Chain":
        new_transforms = []
        for t in self.transforms:
            new_t = t.update(x)
            new_transforms.append(new_t)
            x = new_t(x)

        return eqx.tree_at(lambda m: m.transforms, self, tuple(new_transforms))

    def inverse(self, x: Any) -> Any:
        for t in reversed(self.transforms):
            x = t.inverse(x)
        return x


class Scale(Transform):
    scale: float = eqx.field(static=True, default=1.0)

    def __call__(self, x): return x * self.scale
    def inverse(self, x): return x / self.scale


class Clip(Transform):
    low: float = eqx.field(static=True, default=-jnp.inf)
    high: float = eqx.field(static=True, default=jnp.inf)

    def __call__(self, x): return jnp.clip(x, self.low, self.high)


class Sign(Transform):
    def __call__(self, x): return jnp.sign(x)


class Tanh(Transform):
    def __call__(self, x): return jnp.tanh(x)
    def inverse(self, x): return jnp.atanh(x)


class SymLog(Transform):
    def __call__(self, x): return jnp.sign(x) * jnp.log(jnp.abs(x) + 1.0)
    def inverse(self, x): return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)


class ArcSinh(Transform):
    def __call__(self, x): return jnp.arcsinh(x)
    def inverse(self, x): return jnp.sinh(x)


class SignedSqrt(Transform):
    def __call__(self, x): return jnp.sign(x) * jnp.sqrt(jnp.abs(x))
    def inverse(self, x): return jnp.sign(x) * jnp.square(x)


class EMANormalizer(Transform):
    """Universal normalizer tracking an arbitrary dict of statistics via EMA."""

    emas: dict[str, jax.Array]

    shape: tuple = eqx.field(static=True)
    statistics: dict[str, Callable] = eqx.field(static=True)
    aggregation: Callable | None = eqx.field(static=True)
    momentum: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    center: bool = eqx.field(static=True)

    def __init__(
        self,
        shape: tuple,
        statistics: dict[str, Callable],
        aggregation: Callable | None = None,
        init_ema: dict[str, float] | None = None,
        momentum: float = 0.99,
        eps: float = 1e-8,
        center: bool = False,
    ):
        self.shape = tuple(shape)
        self.statistics = statistics
        self.momentum = momentum
        self.eps = eps
        self.center = center

        if self.center and "mean" not in self.statistics:
            raise ValueError("If center=True, `statistics` must contain a 'mean' key.")

        if aggregation is None:
            if len(statistics) != 1:
                raise ValueError("Must specify `aggregation` if tracking multiple statistics.")
            self.aggregation = lambda **kwargs: list(kwargs.values())[0] # Get the only statistic
        else:
            self.aggregation = aggregation

        if init_ema is None:
            init_ema = {k: 0.0 for k in statistics}
        elif set(init_ema.keys()) != set(statistics.keys()):
            raise ValueError("Keys in `init_ema` must exactly match keys in `statistics`.")

        self.emas = {
            k: jnp.full(self.shape, init_val, dtype=jnp.float32)
            for k, init_val in init_ema.items()
        }

    def __call__(self, x: jax.Array) -> jax.Array:
        scale = self.aggregation(**self.emas)

        if self.center:
            x = x - self.emas["mean"]

        x = x / (scale + self.eps)

        return x

    def update(self, x: jax.Array) -> "EMANormalizer":
        batch = x.reshape((-1,) + self.shape)

        new_emas = {}
        for k, stat_fn in self.statistics.items():
            batch_stat = stat_fn(batch)
            new_emas[k] = self.momentum * self.emas[k] + (1.0 - self.momentum) * batch_stat

        return eqx.tree_at(lambda m: m.emas, self, new_emas)

    def inverse(self, x: jax.Array) -> jax.Array:
        scale = self.aggregation(**self.emas)

        x = x * (scale + self.eps)
        if self.center:
            x = x + self.emas["mean"]

        return x


class RunningMeanStd(Transform):
    """Stateful transform tracking running mean and standard deviation."""

    mean: PyTree[jax.Array]
    var: PyTree[jax.Array]
    count: PyTree[jax.Array]

    eps: float = eqx.field(static=True)
    center: bool = eqx.field(static=True)

    def __init__(
            self,
            shape: PyTree[tuple[int, ...]],
            dtype: PyTree[DTypeLike],
            eps: float = 1e-8,
            center: bool = True,
    ):
        self.eps = eps

        self.mean = jax.tree.map(
            lambda s, d: jnp.zeros(s, dtype=d),
            shape,
            dtype,
            is_leaf=lambda x: isinstance(x, tuple)
        )
        self.var = jax.tree.map(
            lambda s, d: jnp.ones(s, dtype=d),
            shape,
            dtype,
            is_leaf=lambda x: isinstance(x, tuple)
        )
        self.count = jax.tree.map(
            lambda _: jnp.asarray(1e-4, dtype=jnp.float32),
            shape,
            is_leaf=lambda x: isinstance(x, tuple)
        )
        self.center = center

    def __call__(self, x: PyTree[jax.Array]) -> PyTree[jax.Array]:
        return jax.tree.map(
            lambda o, m, v: (o - m if self.center else o) / jnp.sqrt(v + self.eps),
            x, self.mean, self.var
        )

    def inverse(self, x: PyTree[jax.Array]) -> PyTree[jax.Array]:
        return jax.tree.map(
            lambda o, m, v: o * jnp.sqrt(v + self.eps) + (m if self.center else 0),
            x, self.mean, self.var
        )

    def update(self, x: PyTree[jax.Array]) -> "RunningMeanStd":
        """Welford's algorithm for calculating running mean and variance."""

        batch_size = jax.tree.map(lambda arr: jnp.asarray(arr.shape[0], dtype=jnp.float32), x)
        batch_mean = jax.tree.map(lambda arr: jnp.mean(arr, axis=0), x)
        batch_var = jax.tree.map(lambda arr: jnp.var(arr, axis=0), x)

        total_count = jax.tree.map(jnp.add, self.count, batch_size)
        delta = jax.tree.map(jnp.subtract, batch_mean, self.mean)

        def _update_mean(rms_mean, delta, batch_size, total_count):
            return rms_mean + delta * batch_size / total_count

        def _update_var(rms_var, count, var, delta, batch_size, total_count):
            m_a = rms_var * count
            m_b = var * batch_size
            m_2 = m_a + m_b + jnp.square(delta) * count * batch_size / total_count
            return m_2 / total_count

        new_mean = jax.tree.map(_update_mean, self.mean, delta, batch_size, total_count)
        new_var = jax.tree.map(_update_var, self.var, self.count, batch_var, delta, batch_size, total_count)

        return eqx.tree_at(
            lambda t: (t.mean, t.var, t.count),
            self,
            (new_mean, new_var, total_count)
        )
