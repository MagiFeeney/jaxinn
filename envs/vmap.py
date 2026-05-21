from typing import Callable, Optional
import jax.numpy as jnp


class VmapTransformation:
    """Transformations for custom vmap."""

    is_static_leaf: Optional[Callable] = None

    def _wrap_factory(self, core_fn, static_fallback_val=None):
        """build body function based on signature requirements."""
        if self.is_static_leaf is not None:
            def body(path, x):
                if self.is_static_leaf(path, x):
                    return x if static_fallback_val is None else static_fallback_val
                return core_fn(x)
        else:
            def body(x):
                return core_fn(x)

        return body

    def make_get_logical(self, logical_shape) -> Callable:
        logical_size = 1 if len(logical_shape) == 0 else logical_shape[0]

        def core(x):
            if not hasattr(x, "ndim") or x.ndim == 0:
                return x
            if len(logical_shape) == 0:
                return x[0]
            return x[:logical_size]

        return self._wrap_factory(core)

    def make_unflatten(self, axis_size, inner_size, flattened) -> Callable:
        expected = axis_size * inner_size

        def core(x):
            if not hasattr(x, "ndim") or x.ndim == 0:
                return x
            if flattened and x.shape[0] == expected:
                return x.reshape(axis_size, inner_size, *x.shape[1:])
            return x

        return self._wrap_factory(core)

    def make_out_batching(self, axis_size, inner_size, flattened) -> Callable:
        expected = axis_size * inner_size

        def core(x):
            if not hasattr(x, "ndim") or x.ndim == 0:
                return False
            return x.shape[0] == expected

        return self._wrap_factory(core, static_fallback_val=False)

    def make_pad_leaf(self, logical_shape, capacity) -> Callable:
        logical_size = 1 if len(logical_shape) == 0 else logical_shape[0]
        pad_size = capacity - logical_size

        def core(x):
            if len(logical_shape) == 0:
                x = jnp.expand_dims(x, 0)
            if not hasattr(x, "ndim") or x.ndim == 0:
                return x
            pw = ((0, pad_size),) + ((0, 0),) * (x.ndim - 1)
            return jnp.pad(x, pw, mode="constant", constant_values=0)

        return self._wrap_factory(core)

    def make_flatten(self, axis_size) -> Callable:
        def core(x):
            if not hasattr(x, "ndim") or x.ndim < 2:
                return x
            return x.reshape(axis_size * x.shape[1], *x.shape[2:])

        return self._wrap_factory(core)
