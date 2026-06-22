import math
from typing import Optional, Callable, Union, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import equinox as eqx
import distrax

from jaxinn.structs import LatentState

from .utils import make_mlp, dx, StaticCallable


class Critic(eqx.Module):
    net: eqx.nn.Sequential
    action_size: Optional[int] = eqx.field(static=True)
    head_type: str = eqx.field(static=True)
    min_std: float = eqx.field(static=True)

    def __init__(
            self,
            belief_size: int,
            state_size: Union[int, Tuple[int, ...]],
            hidden_size: list[int],
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

        if isinstance(state_size, tuple):
            state_size = math.prod(state_size)

        input_size = belief_size + state_size + (
            0 if action_size is None else int(action_size)
        )

        self.net = make_mlp(
            input_size = input_size,
            hidden_size = hidden_size,
            output_size = output_size,
            activation = activation_function,
            key = key
        )

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
