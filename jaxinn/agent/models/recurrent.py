from collections.abc import Callable

import jax
import jax.numpy as jnp
import equinox as eqx

from .utils import get_activation_fn


class FusedGRUCell(eqx.Module):
    linear_and_gate: eqx.nn.Sequential

    update_gate_bias: float
    activation_function: Callable

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            use_layernorm: bool = True,
            activation_function: str | Callable = "tanh",
            update_gate_bias: float = -1.0,
            *,
            key: jax.random.PRNGKey
    ):
        self.update_gate_bias = update_gate_bias
        self.activation_function = get_activation_fn(activation_function)

        layers = [
            eqx.nn.Linear(
                input_size + hidden_size,
                3 * hidden_size,
                key=key
            )
        ]
        if use_layernorm:
            layers.append(eqx.nn.LayerNorm(3 * hidden_size))

        self.linear_and_gate = eqx.nn.Sequential(tuple(layers))

    def __call__(self, x: jax.Array, state: jax.Array) -> jax.Array:
        x = jnp.concatenate([x, state], axis=-1)
        parts = self.linear_and_gate(x)

        reset_gate, candidate, update_gate = jnp.split(parts, 3, axis=-1)

        reset_gate = jax.nn.sigmoid(reset_gate)
        candidate = self.activation_function(reset_gate * candidate) # Trade reset for parallelization
        update_gate = jax.nn.sigmoid(update_gate + self.update_gate_bias)

        out = update_gate * candidate + (1.0 - update_gate) * state
        return out
