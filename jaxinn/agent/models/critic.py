import jax
import jax.numpy as jnp
import equinox as eqx
import distrax

from typing import Optional, Callable, Union
from jaxtyping import Array, Float, PRNGKeyArray
from .utils import get_activation_fn, dx, StaticCallable
from .world import LatentState


class Critic(eqx.Module):
    net: eqx.nn.Sequential
    action_size: Optional[int] = eqx.field(static=True)
    head_type: str = eqx.field(static=True)
    min_std: float = eqx.field(static=True)

    def __init__(
            self,
            belief_size: int,
            state_size: int,
            hidden_size: int,
            activation_function: Union[str, Callable] = "elu",
            action_size: Optional[int] = None,
            min_std: float = 0.0,
            head_type="Isotropic Normal",
            *,
            key: PRNGKeyArray,
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
            StaticCallable(activation),
            eqx.nn.Linear(hidden_size, hidden_size, key=keys[1]),
            StaticCallable(activation),
            eqx.nn.Linear(hidden_size, hidden_size, key=keys[2]),
            StaticCallable(activation),
            eqx.nn.Linear(hidden_size, output_size, key=keys[3]),
        ])

        self.head_type = head_type
        self.action_size = action_size
        self.min_std = min_std

    def __call__(
        self,
        latent_state: Union[Float[Array, "... input_size"], LatentState],
        action: Optional[Float[Array, "... input_size"]] = None,
    ) -> distrax.Distribution:
        if isinstance(latent_state, LatentState):
            latent_state = latent_state.feature

        assert (action is None) == (self.action_size is None)
        if action is not None:
            latent_state = jnp.concatenate([latent_state, action], axis=-1)

        out = self.net(latent_state)

        if self.head_type == "Isotropic Normal":
            mean = out
            std = jnp.ones_like(mean)
            dist = dx.Normal(mean, std)
        elif self.head_type == "Normal":
            mean, log_std = jax.split(out, 2, axis=-1)
            std = jax.nn.softplus(log_std) + self.min_std
            dist = dx.Normal(mean, std)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

        return dist
