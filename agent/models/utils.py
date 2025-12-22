import enum
import jax.nn as jnn
from typing import Callable, Union


class ActivationKind(enum.Enum):
    relu = jnn.relu
    elu = jnn.elu
    silu = jnn.silu
    gelu = jnn.gelu
    mish = jnn.mish
    tanh = jnn.tanh
    sigmoid = jnn.sigmoid


Activation = Union[str, Callable, ActivationKind]


def get_activation_fn(activation: Activation) -> Callable:
    """
    Returns a JAX activation function.
    """
    if isinstance(activation, str):
        return ActivationKind[activation].value
    elif callable(activation):
        return activation
    else:
        assert isinstance(activation, ActivationKind)
        return activation.value
