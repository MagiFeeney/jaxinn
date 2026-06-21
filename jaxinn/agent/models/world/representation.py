from typing import Union, Dict, Tuple, Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import equinox as eqx

from jaxinn.agent.models.utils import get_activation_fn, dx, StaticCallable, FactoryLike


# Representation
class Representation(eqx.Module):
    """Representation learning of state, inferred from history and the latest observation: p(s_t | h_t, o_t)
    """
    net: eqx.nn.Sequential
    dist_cls: FactoryLike = eqx.field(static=True)
    head_type: str = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    num_categories: int = eqx.field(static=True)
    min_std: float = eqx.field(static=True)

    def __init__(
            self,
            belief_size: int,
            embedding_size: int,
            state_size: Union[int, Tuple[int, ...]],
            hidden_size: int,
            min_std: float = 0.1,
            activation_function="elu",
            head_type: str = "Normal",
            *,
            key: PRNGKeyArray,
    ):
        if head_type == "Normal":
            self.dist_cls = dx.Independent(dx.Normal, reinterpreted_batch_ndims=1) # Composed distribution
            self.num_variables, self.num_categories = state_size, 0
            output_size = 2 * state_size
        elif head_type == "Categorical":
            self.dist_cls = dx.Independent(dx.OneHotCategorical, reinterpreted_batch_ndims=len(state_size) - 1)
            assert isinstance(state_size, tuple) and len(state_size) == 2, (
                f"Expected `state_size` to be a 2-element tuple (representing a stack of "
                f"independent categorical distributions), but got {state_size!r}."
            )
            self.num_variables, self.num_categories = state_size # Unpack the tuple
            output_size = self.num_variables * self.num_categories # Flatten and concatenate all the categorical variables
        else:
            raise NotImplementedError(f"Unsupported head_type: {head_type}")

        activation = get_activation_fn(activation_function)

        keys = jax.random.split(key, 2)

        self.net = eqx.nn.Sequential([
            eqx.nn.Linear(belief_size + embedding_size, hidden_size, key=keys[0]),
            StaticCallable(activation),
            eqx.nn.Linear(hidden_size, output_size, key=keys[1]),
        ])

        self.min_std = min_std
        self.head_type = head_type

    def __call__(
            self,
            belief: Float[Array, "... belief_size"],
            obs: Float[Array, "... embedding_size"],
    ) -> Tuple[
        Dict[str, Float[Array, "..."]],
        Float[Array, "... belief_size"],
    ]:
        input_tensor = jnp.concatenate([belief, obs], axis=-1)
        out = self.net(input_tensor)

        if self.head_type == "Normal":
            mean, log_std = jnp.split(out, 2, axis=-1)
            std = jax.nn.softplus(log_std) + self.min_std
            params = {"loc": mean, "scale": std}
        elif self.head_type == "Categorical":
            logit = out.reshape(*out.shape[:-1], self.num_variables, self.num_categories)
            params = {"logits": logit}

        return params, belief

    def sample(
            self,
            params: Dict[str, Any],
            key: PRNGKeyArray,
    ) -> Float[Array, "... state_size"]:
        dist = self.dist_cls(**params)

        if self.head_type == "Normal":
            state = dist.sample(seed=key)
        elif self.head_type == "Categorical":
            state = dist.sample(seed=key)
            state = state + dist.probs - jax.lax.stop_gradient(dist.probs) # straight-through gradient
            state = state.reshape(*state.shape[:-2], -1) # flatten

        return state
