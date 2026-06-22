from typing import Any, Callable, Dict, Union, Optional

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree
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
        return jax.tree.map(
            lambda leaf: (
                leaf.dist if isinstance(leaf, FixedDistrax)
                else leaf.value if isinstance(leaf, Static)
                else leaf
            ),
            x,
            is_leaf=lambda leaf: isinstance(leaf, (FixedDistrax, Static)),
        )

    @property
    def dist(self):
        resolved_args = self._resolve(self.args)
        resolved_kwargs = self._resolve(self.kwargs)
        return self.cls(*resolved_args, **resolved_kwargs)

    def __getattr__(self, name):
        if hasattr(self.dist, name):
            return getattr(self.dist, name)
        return getattr(self.dist.distribution, name)


class Composer(eqx.Module):
    cls: Callable = eqx.field(static=True)
    factory_args: tuple
    factory_kwargs: dict

    def __call__(self, *runtime_args, **runtime_kwargs):
        def _resolve_factory(leaf):
            if _is_factory(leaf):
                return leaf(*runtime_args, **runtime_kwargs)
            return leaf

        resolved_args, resolved_kwargs = jax.tree.map(
            _resolve_factory,
            (self.factory_args, self.factory_kwargs),
            is_leaf=_is_factory
        )
        return FixedDistrax(self.cls, *resolved_args, **resolved_kwargs)


class FixedFactory(eqx.Module):
    cls: Callable = eqx.field(static=True)

    def __call__(self, *args, **kwargs):
        leaves, _ = jax.tree.flatten((args, kwargs), is_leaf=_is_factory)
        if any(_is_factory(leaf) for leaf in leaves):
            return Composer(self.cls, args, kwargs)        # Delay instantiation
        else:
            return FixedDistrax(self.cls, *args, **kwargs) # Primitive fires with parameters


FactoryLike = Composer | FixedFactory


def _is_factory(x: Any) -> bool:
    return isinstance(x, FactoryLike)


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


def make_mlp(
        input_size: int,
        hidden_size: Union[int, list[int]],
        output_size: int,
        activation: Union[str, Callable, StaticCallable],
        num_layers: Optional[int] = None,
        *,
        key: PRNGKeyArray
) -> eqx.nn.Sequential:
    if isinstance(hidden_size, int):
        assert num_layers is not None and num_layers >= 0, (
            "When hidden_size is an integer, num_layers must be specified "
            "and non-negative."
        )
        hidden_size = [hidden_size] * num_layers

    sizes = [input_size] + hidden_size + [output_size]

    layers = []
    keys = jax.random.split(key, len(sizes) - 1)

    if isinstance(activation, str):
        activation = get_activation_fn(activation)
    if not isinstance(activation, StaticCallable):
        activation = StaticCallable(activation)

    for i in range(len(sizes) - 1):
        layers.append(eqx.nn.Linear(sizes[i], sizes[i+1], key=keys[i]))
        if i < len(sizes) - 2:
            layers.append(activation)

    return eqx.nn.Sequential(layers)
