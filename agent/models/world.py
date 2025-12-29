import jax
import jax.numpy as jnp
import equinox as eqx
import distrax

from typing import Optional, Callable, Union, Dict
from jaxtyping import Array, Float, PRNGKeyArray
from .utils import get_activation_fn, dx


class LatentState(eqx.Module):
    """
    Combine deterministic history encoding (belief) and the stochastic predictor (state) into a single state.
    """
    belief: jax.Array  # h_t
    state: jax.Array   # s_t

    @property
    def batch_shape(self) -> tuple:
        return self.belief.shape[:-1]

    @property
    def feature(self) -> jax.Array:
        return jnp.concatenate([self.belief, self.state], axis=-1)

    def __getitem__(self, index: Any) -> "LatentState":
        return jax.tree.map(lambda x: x[index], self)

    def flatten(self) -> "LatentState":
        return jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), self)

    def narrow(self, axis: int, start: int, length: int) -> "LatentState": # TODO: delete if not used
        return jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, start, length, axis),
            self
        )


class LatentStateWithParams(eqx.Module):
    """
    Store the LatentState along with its parameters
    """
    latent_state: LatentState
    params: Dict[str, jax.Array]
    dist_cls: Callable[..., Any] = eqx.field(static=True)

    @property
    def dist(self):
        return self.dist_cls(**self.params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dist, name)


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
            latent_state: LatentState,
            obs: Float[Array, "... embedding_size"],
            key: PRNGKeyArray,
    ) -> LatentStateWithParams:
        input_tensor = jnp.concatenate([latent_state.belief, obs], axis=-1)
        out = self.net(input_tensor)

        if self.head_type == "Normal":
            mean, log_std = jnp.split(out, 2, axis=-1)
            std = jax.nn.softplus(log_std) + self.min_std
            params = {"loc": mean, "scale": std}
            dist_cls = dx.Normal
            dist = dist_cls(mean, std)
            state = dist.sample(seed=key)
        elif self.head_type == "Categorical":
            logit = out.reshape(*out.shape[:-1], self.num_variables, self.num_categories)
            params = {"logits": logit}
            dist_cls = dx.OneHotCategorical
            dist = dist_cls(logits=logit)
            state = dist.sample(seed=key)
            state = state + dist.probs - jax.lax.stop_gradient(dist.probs) # straight-through gradient
            state = state.reshape(*state.shape[:2], -1) # flatten

        return LatentStateWithParams(
            latent_state=LatentState(belief=latent_state.belief, state=state),
            params=params,
            dist_cls=dist_cls
        )


class TransitionModel(eqx.Module):
    encoder: eqx.nn.Sequential
    body: eqx.nn.GRUCell
    head: eqx.nn.Sequential
    head_type: str = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    num_categories: int = eqx.field(static=True)
    min_std: float

    def __init__(
            self,
            belief_size: int,
            state_size: Union[int, tuple],
            action_size: int,
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
            input_size = state_size + action_size
        elif head_type == "Categorical":
            self.num_variables, self.num_categories = state_size # Unpack the tuple
            output_size = self.num_variables * self.num_categories # Flatten and concatenate all the categorical variables
            input_size = output_size + action_size
        else:
            raise NotImplementedError(f"Unsupported head_type: {head_type}")

        activation = get_activation_fn(activation_function)

        keys = jax.random.split(key, 4)

        # p(c_{t - 1} | s_{t - 1}, a_{t - 1})
        self.encoder = eqx.nn.Sequential([
            eqx.nn.Linear(input_size, hidden_size, key=keys[0]),
            eqx.nn.Lambda(activation),
        ])

        # p(h_t | c_{t - 1}, h_{t - 1})
        self.body = nn.GRUCell(hidden_size, belief_size, key=keys[1])

        # p(s_t | h_t)
        self.head = eqx.nn.Sequential([
            eqx.nn.Linear(belief_size, hidden_size, key=keys[2]),
            eqx.nn.Lambda(activation),
            eqx.nn.Linear(hidden_size, output_size, key=keys[3]),
        ])

        self.min_std = min_std
        self.head_type = head_type

    def __call__(
            self,
            latent_state: LatentState,
            action: Float[Array, "... action_size"],
            key: PRNGKeyArray,
    ) -> LatentStateWithParams:
        input_tensor = jnp.concatenate([latent_state.state, action], axis=-1)
        embedding = self.encoder(input_tensor)
        belief = self.body(embedding, latent_state.belief)
        out = self.head(belief)

        if self.head_type == "Normal":
            mean, log_std = jnp.split(out, 2, axis=-1)
            std = jax.nn.softplus(log_std) + self.min_std
            params = {"loc": mean, "scale": std}
            dist_cls = dx.Normal
            dist = dist_cls(mean, std)
            state = dist.sample(seed=key)
        elif self.head_type == "Categorical":
            logit = out.reshape(*out.shape[:-1], self.num_variables, self.num_categories)
            params = {"logits": logit}
            dist_cls = dx.OneHotCategorical
            dist = dist_cls(logits=logit)
            state = dist.sample(seed=key)
            state = state + dist.probs - jax.lax.stop_gradient(dist.probs) # straight-through gradient
            state = state.reshape(*state.shape[:2], -1) # flatten

        return LatentStateWithParams(
            latent_state=LatentState(belief=belief, state=state),
            params=params,
            dist_cls=dist_cls
        )


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
