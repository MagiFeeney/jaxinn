import jax
import jax.nn as jnn
import jax.numpy as jnp
from typing import Callable, Union, Dict, Any
from jaxtyping import PyTree, Array, PRNGKeyArray
import equinox as eqx
from equinox._module import Static
import distrax


RegisteredItem = Union[str, Callable]
Activation = RegisteredItem
Dtype = RegisteredItem


ACTIVATIONS = {
    "relu": jnn.relu,
    "elu": jnn.elu,
    "silu": jnn.silu,
    "gelu": jnn.gelu,
    "mish": jnn.mish,
    "tanh": jnn.tanh,
    "sigmoid": jnn.sigmoid,
}


DTYPES = {
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
    "float64": jnp.float64,
}


def _create_getter(registry: Dict[str, Any], entity_name: str) -> Callable[[RegisteredItem], Callable]:
    """Generates a getter function for a specific registry."""
    def getter(name_or_fn: RegisteredItem) -> Callable:
        if isinstance(name_or_fn, str):
            try:
                return registry[name_or_fn.lower()]
            except KeyError:
                raise ValueError(f"Unknown {entity_name}: {name_or_fn}")
        elif callable(name_or_fn):
            return name_or_fn
        else:
            raise TypeError(f"Expected str or callable, got {type(name_or_fn)}")
    return getter


get_activation_fn = _create_getter(ACTIVATIONS, "activation")
get_precision_fn = _create_getter(DTYPES, "dtype")


class StaticCallable(eqx.Module):
    fn: Callable[[Array], Array] = eqx.field(static=True)

    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array:
        return self.fn(x)


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
