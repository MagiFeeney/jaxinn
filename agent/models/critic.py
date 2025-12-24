import jax
import jax.numpy as jnp
import equinox as eqx
import distrax

from typing import Optional
from jaxtyping import Array, Float, PRNGKeyArray
from .utils import get_activation_fn


class ValueModel(eqx.Module):
    net: eqx.nn.Sequential
    action_size: Optional[int] = eqx.field(static=True)
    head_type: str = eqx.field(static=True)

    min_std: float

    def __init__(
            self,
            belief_size: int,
            state_size: int,
            hidden_size: int,
            activation_function="elu",
            action_size: Optional[int] = None,
            min_std: float = 0.0,
            head_type="Isotropic Normal",
            *,
            key: jax.random.PRNGKey,
    ):  # if action_size is not None, Q fn
        if head_type == "Isotropic Normal":
            output_size = 1
        elif head_type == "Normal":
            output_size = 2
        else:
            raise NotImplementedError

        activation = get_activation_fn(activation_function)

        keys = jax.random.split(key, 4)
        self.net = eqx.nn.Sequential([
            eqx.nn.Linear(
                belief_size + state_size + (
                    0 if action_size is None else int(action_size)
                ),
                hidden_size, key=keys[0]
            ),
            eqx.nn.Lambda(activation),
            eqx.nn.Linear(hidden_size, hidden_size, key=keys[1]),
            eqx.nn.Lambda(activation),
            eqx.nn.Linear(hidden_size, hidden_size, key=keys[2]),
            eqx.nn.Lambda(activation),
            eqx.nn.Linear(hidden_size, output_size, key=keys[3]),
        ])

        self.head_type = head_type
        self.action_size = action_size
        self.min_std = min_std

    def __call__(
        self,
        input_tensor: Float[Array, "... input_dim"],
        action: Optional[Float[Array, "... input_dim"]] = None,
    ) -> distrax.Distribution:
        assert (action is None) == (self.action_size is None)
        if action is not None:
            input_tensor = jnp.concatenate([input_tensor, action], axis=-1)
        out = self(input_tensor)

        if self.head_type == "Isotropic Normal":
            mean = out
            std = 1.0
            dist = distrax.Normal(mean, std)
        elif self.head_type == "Normal":
            mean, log_std = jax.split(out, 2, axis=-1)
            std = jax.nn.softplus(log_std) + self.min_std
            dist = distrax.Normal(mean, std)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

        return dist
