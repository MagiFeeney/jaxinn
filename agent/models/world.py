import jax
import jax.numpy as jnp
import equinox as eqx
import distrax

from typing import Optional, Callable, Union
from jaxtyping import Array, Float, PRNGKeyArray
from .utils import get_activation_fn, dx


class TransitionModel(jit.ScriptModule):
    __constants__ = [
        "min_stddev", "action_size",
        "belief_size", "state_size",
        "embedding_size",
    ]

    belief_size: int
    state_size: int
    embedding_size: int
    action_size: int

    def __init__(
        self,
        belief_size: int,  # h_t
        state_size: int,   # s_t
        action_size: int,
        hidden_size: int,
        embedding_size: int,  # enc(o_t)
        activation_function: Union[str, Callable] = "elu",
        min_stddev: float = 0.1,
    ):
        super().__init__()
        self.min_stddev = min_stddev
        self.belief_size = belief_size
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.action_size = action_size

        assert (belief_size > 0) and (state_size > 0)

        # x
        self.state_action_pre_rnn = BottledModule(nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            get_activation_module(activation_function),
        ))                                                  # p(c_{t - 1} | s_{t - 1}, a_{t - 1})
        self.rnn = nn.GRUCell(hidden_size, belief_size) # p(h_t | c_{t - 1}, h_{t - 1})

        self.belief_to_state_prior = BottledModule(nn.Sequential(
            nn.Linear(belief_size, hidden_size),
            get_activation_module(activation_function),
            nn.Linear(hidden_size, 2 * state_size),
        ))                      # state prior: p(s_t | h_t)

        # posterior

        self.xy_belief_obs_to_state_posterior = BottledModule(nn.Sequential(
            nn.Linear(
                belief_size + embedding_size,
                hidden_size,
            ),
            get_activation_module(activation_function),
            nn.Linear(
                hidden_size,
                2 * state_size,
            ),
        ))                      # state posterior: p(s_t | h_t, o_t)


class RepresentationModel(eqx.Module):
    """Representation learning of state, inferred from history and the latest observation: p(s_t | h_t, o_t)
    """
    net: eqx.nn.Sequential
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
    ):
        activation = get_activation_fn(activation_function)

        if head_type == "Normal":
            output_size = 2 * state_size
        elif head_type == "Categorical":
            self.num_variables, self.num_categories = state_size
            output_size = self.num_variables * self.num_categories # Flatten and concatenate all the categorical variables
        else:
            raise NotImplementedError

        self.net = eqx.nn.Sequential([
            eqx.nn.Linear(belief_size + embedding_size, hidden_size),
            eqx.nn.Lambda(activation),
            eqx.nn.Linear(hidden_size, output_size),
        ])

        self.min_std = min_std

    def __call__(self, belief, obs):
        input_tensor = jnp.concatenate([belief, obs], axis=-1)
        out = self.net(input_tensor)
        if self.head_type == "Normal":
            mean, log_std = jax.split(out, 2, axis=-1)
            std = jax.nn.softplus(log_std) + self.min_std
            dist = dx.Normal(mean, std)
        elif self.head_type == "Categorical":
            logit = out.reshape(*out.shape[:-1], self.num_categories, self.num_variables) # TODO: determine order of K and N
            dist = dx.OneHotCategorical(logits=logit)
        return dist


class RewardModel(eqx.Module):
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
        out = self.net(input_tensor)

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
