import enum
import jax.nn as jnn
from typing import Callable, Union


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
