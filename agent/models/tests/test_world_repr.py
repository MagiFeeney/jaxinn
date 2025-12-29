import enum
import jax.nn as jnn

import jax
import jax.numpy as jnp
import equinox as eqx
import distrax

from typing import Tuple, Union, Any, Optional, Callable
from jaxtyping import Array, Float, PRNGKeyArray


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


class RepresentationModel(eqx.Module):
    """Representation learning of state, inferred from history and the latest observation: p(s_t | h_t, o_t)
    """
    net: eqx.nn.Sequential
    head_type: str = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    num_categories: int = eqx.field(static=True)
    min_std: float

    def __init__(
            self,
            belief_size: int,
            embedding_size: int,
            state_size: Union[int, tuple],
            hidden_size: int,
            min_std: float = 0.1,
            activation_function="elu",
            head_type: str = "Normal",
            *,
            key: jax.random.PRNGKey,
    ):
        if head_type == "Normal":
            self.num_variables, self.num_categories = state_size, 0
            output_size = 2 * state_size
        elif head_type == "Categorical":
            self.num_variables, self.num_categories = state_size # Unpack the tuple
            output_size = self.num_variables * self.num_categories # Flatten and concatenate all the categorical variables
        else:
            raise NotImplementedError(f"Unsupported head_type: {head_type}")

        activation = get_activation_fn(activation_function)

        keys = jax.random.split(key, 2)

        self.net = eqx.nn.Sequential([
            eqx.nn.Linear(belief_size + embedding_size, hidden_size, key=keys[0]),
            eqx.nn.Lambda(activation),
            eqx.nn.Linear(hidden_size, output_size, key=keys[1]),
        ])

        self.min_std = min_std
        self.head_type = head_type

    def __call__(
            self,
            belief: Float[Array, "... belief_size"],
            obs: Float[Array, "... embedding_size"],
    ) -> distrax.Distribution:
        input_tensor = jnp.concatenate([belief, obs], axis=-1)
        out = self.net(input_tensor)

        if self.head_type == "Normal":
            mean, log_std = jnp.split(out, 2, axis=-1)
            std = jax.nn.softplus(log_std) + self.min_std
            dist = dx.Normal(mean, std)
        elif self.head_type == "Categorical":
            logit = out.reshape(*out.shape[:-1], self.num_variables, self.num_categories)
            dist = dx.OneHotCategorical(logits=logit)

        return dist


belief_size = 20
embedding_size = 100
state_size = 50
hidden_size = 64

batch_size = 10

key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key, 2)

# Normal
model = RepresentationModel(belief_size, embedding_size, state_size, hidden_size, head_type="Normal", key=subkey)

belief = jnp.ones((batch_size, belief_size))
obs = jnp.ones((batch_size, embedding_size))

dist = jax.vmap(model)(belief, obs)

sample = dist.sample(seed=0)
print(f"sample.shape {sample.shape}")

logp = dist.log_prob(sample)
print(f"logp.shape {logp.shape}")


key, subkey = jax.random.split(key, 2)

state_size = (64, 32)

# Categorical
model = RepresentationModel(belief_size, embedding_size, state_size, hidden_size, head_type="Categorical", key=subkey)

dist = jax.vmap(model)(belief, obs)

sample = dist.sample(seed=0)
print(f"sample.shape {sample.shape}")

logp = dist.log_prob(sample)
print(f"logp.shape {logp.shape}")
