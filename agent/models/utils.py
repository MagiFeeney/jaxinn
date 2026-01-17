import jax
import jax.nn as jnn
from typing import Callable, Union, Any
from jaxtyping import PyTree
import equinox as eqx
import distrax


ACTIVATIONS = {
    "relu": jnn.relu,
    "elu": jnn.elu,
    "silu": jnn.silu,
    "gelu": jnn.gelu,
    "mish": jnn.mish,
    "tanh": jnn.tanh,
    "sigmoid": jnn.sigmoid,
}


Activation = Union[str, Callable]


def get_activation_fn(name_or_fn) -> Callable:
    if isinstance(name_or_fn, str):
        try:
            return ACTIVATIONS[name_or_fn.lower()]
        except KeyError:
            raise ValueError(f"Unknown activation: {name_or_fn}")
    elif callable(name_or_fn):
        return name_or_fn
    else:
        raise TypeError(f"Expected str or callable, got {type(name_or_fn)}")


class Static(eqx.Module):
    value: any = eqx.field(static=True)


class FixedDistrax(eqx.Module):
    cls: Callable = eqx.field(static=True)
    args: PyTree[Any]
    kwargs: PyTree[Any]

    def __init__(self, cls: Callable, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def _resolve(self, x):
        return jax.tree_util.tree_map(
            lambda leaf: (
                leaf.dist if isinstance(leaf, FixedDistrax)
                else leaf.value if isinstance(leaf, Static)
                else leaf
            ),
            x,
            is_leaf=lambda leaf: (
                isinstance(leaf, FixedDistrax)
                or isinstance(leaf, Static)
            ),
        )

    @property
    def dist(self):
        resolved_args = self._resolve(self.args)
        resolved_kwargs = self._resolve(self.kwargs)
        return self.cls(*resolved_args, **resolved_kwargs)

    def __getattr__(self, name):
        return getattr(self.dist, name)


class FixedFactory(eqx.Module):
    cls: Callable = eqx.field(static=True)

    def __call__(self, *args, **kwargs):
        return FixedDistrax(self.cls, *args, **kwargs)


class ProxyDistrax:
    def __call__(self, module):
        """
        Wrap a custom distrax-compatible callable (function or class).
        """
        if not callable(module):
            raise TypeError("ProxyDistrax can only wrap callables")

        return FixedFactory(module)

    def __getattr__(self, name):
        """
        Wrap a distrax method directly.
        """
        attr = getattr(distrax, name)

        if callable(attr):
            return FixedFactory(attr)

        return attr


dx = ProxyDistrax()
