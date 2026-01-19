import jax
import jax.numpy as jnp
import equinox as eqx
import distrax

from typing import Optional, Callable, Union, Dict, Tuple, Any
from jaxtyping import Array, Float, PRNGKeyArray
from .utils import get_activation_fn, dx


class LatentState(eqx.Module):
    """
    Combine deterministic history encoding (belief) and the stochastic predictor (state) into a single state.
    """
    belief: jax.Array  # h_t
    state: jax.Array   # s_t

    @classmethod
    def initialize(
            cls,
            belief_size: int,
            state_size: int,
            random_init: bool = False,
            batch_shape: Tuple[int, ...] = (),
            *,
            key: PRNGKeyArray,
    ) -> "LatentState":
        key_belief, key_state = jax.random.split(key, 2)

        mask = float(random_init)
        belief = jax.random.normal(key_belief, batch_shape + (belief_size,)) * mask
        state  = jax.random.normal(key_state,  batch_shape + (state_size,))  * mask

        return cls(belief=belief, state=state)

    @property
    def batch_shape(self) -> tuple:
        return self.belief.shape[:-1]

    @property
    def feature(self) -> jax.Array:
        return jnp.concatenate([self.belief, self.state], axis=-1)

    def __getitem__(self, index: Any) -> "LatentState":
        return jax.tree.map(lambda x: x[index], self)

    def __mul__(self, other):
        return jax.tree.map(lambda x: x * other, self)

    def __rmul__(self, other):
        return jax.tree.map(lambda x: other * x, self)

    def flatten(self) -> "LatentState":
        return jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), self)

    def narrow(self, axis: int, start: int, length: int) -> "LatentState":
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


# Perception
class Encoder(eqx.Module):
    body: eqx.nn.Sequential
    head: Union[eqx.nn.Linear, eqx.nn.Identity]
    shape: Tuple[int, int, int] = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    embedding_size: Optional[int] = eqx.field(static=True)

    def __init__(
            self,
            shape: Tuple[int, int, int],
            kernel_size: int = 4,
            depth: int = 48,
            stride: int = 2,
            embedding_size: Optional[int] = 1024,
            activation_function: Union[str, Callable] = "elu",
            *,
            key: PRNGKeyArray
    ):
        activation = get_activation_fn(activation_function)

        if embedding_size is not None:
            keys = jax.random.split(key, 5)
        else:
            keys = jax.random.split(key, 4)

        self.body = eqx.nn.Sequential([
            eqx.nn.Conv2d(shape[0], 1 * depth, kernel_size=kernel_size, stride=stride, key=keys[0]),
            eqx.nn.Lambda(activation),
            eqx.nn.Conv2d(1 * depth, 2 * depth, kernel_size=kernel_size, stride=stride, key=keys[1]),
            eqx.nn.Lambda(activation),
            eqx.nn.Conv2d(2 * depth, 4 * depth, kernel_size=kernel_size, stride=stride, key=keys[2]),
            eqx.nn.Lambda(activation),
            eqx.nn.Conv2d(4 * depth, 8 * depth, kernel_size=kernel_size, stride=stride, key=keys[3]),
            eqx.nn.Lambda(activation),
            eqx.nn.Lambda(jnp.ravel),
        ])

        feature_map_shape = self.get_feature_map_shape(shape)
        feature_map_size = int(jnp.prod(feature_map_shape)) # flattened
        if embedding_size is not None:
            self.embedding_size = embedding_size
            self.head = eqx.nn.Linear(feature_map_size, embedding_size, key=keys[4])
        else:
            self.embedding_size = feature_map_size
            self.head = eqx.nn.Identity()

        self.shape = shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.stride = stride

    def __call__(
            self,
            obs: Float[Array, "... obs_size"]
    ) -> Float[Array, "... output_size"]:
        feature = self.body(obs)
        out = self.head(feature)
        return out

    def get_feature_map_shape(self, shape) -> Tuple[int, int, int]:
        dummy_input = jnp.zeros(shape)
        out_shape_struct = jax.eval_shape(self.body, dummy_input)
        out_shape = jnp.array(out_shape_struct.shape)
        return out_shape


class Decoder(eqx.Module):
    embedding: eqx.nn.Linear
    body: eqx.nn.Sequential
    shape: Tuple[int, int, int] = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    embedding_size: int = eqx.field(static=True)

    def __init__(
            self,
            belief_size: int,
            state_size: int,
            shape: Tuple[int, int, int],
            kernel_size: int = 4,
            depth: int = 48,
            stride: int = 2,
            activation_function: Union[str, Callable] = "elu",
            embedding_size: int = 1024,
            *,
            key: PRNGKeyArray
    ):
        activation = get_activation_fn(activation_function)

        keys = jax.random.split(key, 5)

        self.embedding = eqx.nn.Linear(belief_size + state_size, embedding_size, key=keys[0])

        self.body = eqx.nn.Sequential([
            eqx.nn.Lambda(activation),
            eqx.nn.ConvTranspose2d(embedding_size, 4 * depth, kernel_size=5, stride=stride, key=keys[1]),
            eqx.nn.Lambda(activation),
            eqx.nn.ConvTranspose2d(4 * depth, 2 * depth, kernel_size=5, stride=stride, key=keys[2]),
            eqx.nn.Lambda(activation),
            eqx.nn.ConvTranspose2d(2 * depth, 1 * depth, kernel_size=6, stride=stride, key=keys[3]),
            eqx.nn.Lambda(activation),
            eqx.nn.ConvTranspose2d(1 * depth, shape[0], kernel_size=6, stride=stride, key=keys[4]),
        ])

        self.shape = shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.stride = stride
        self.embedding_size = embedding_size

    def __call__(
            self,
            latent_state: Union[Float[Array, "... input_size"], LatentState],
    ) -> distrax.Distribution:
        if isinstance(latent_state, LatentState):
            latent_state = latent_state.feature
        embedding = self.embedding(latent_state)
        embedding = embedding[..., None, None] # Reshape the vector to BCHW
        out = self.body(embedding)
        dist = dx.Normal(out, jnp.ones_like(out))
        return dx.Independent(dist, reinterpreted_batch_ndims=len(self.shape))


# Representation
class Representation(eqx.Module):
    """Representation learning of state, inferred from history and the latest observation: p(s_t | h_t, o_t)
    """
    net: eqx.nn.Sequential
    dist_cls: str = eqx.field(static=True)
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
            key: PRNGKeyArray,
    ):
        if head_type == "Normal":
            self.dist_cls = dx.Normal
            self.num_variables, self.num_categories = state_size, 0
            output_size = 2 * state_size
        elif head_type == "Categorical":
            self.dist_cls = dx.OneHotCategorical
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
            state = state.reshape(*state.shape[:2], -1) # flatten

        return state


# Transition
class Transition(eqx.Module):
    encoder: eqx.nn.Sequential
    body: eqx.nn.GRUCell
    head: eqx.nn.Sequential
    dist_cls: str = eqx.field(static=True)
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
            key: PRNGKeyArray,
    ):
        if head_type == "Normal":
            self.dist_cls = dx.Normal
            self.num_variables, self.num_categories = state_size, 0
            output_size = 2 * state_size
            input_size = state_size + action_size
        elif head_type == "Categorical":
            self.dist_cls = dx.OneHotCategorical
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
        self.body = eqx.nn.GRUCell(hidden_size, belief_size, key=keys[1])

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
    ) -> Tuple[
        Dict[str, Float[Array, "..."]],
        Float[Array, "... belief_size"],
    ]:
        input_tensor = jnp.concatenate([latent_state.state, action], axis=-1)
        embedding = self.encoder(input_tensor)
        belief = self.body(embedding, latent_state.belief)
        out = self.head(belief)

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
            params: Dict[str, Float[Array, "..."]],
            key: PRNGKeyArray,
    ) -> Float[Array, "... state_size"]:
        dist = self.dist_cls(**params)

        if self.head_type == "Normal":
            state = dist.sample(seed=key)
        elif self.head_type == "Categorical":
            state = dist.sample(seed=key)
            state = state + dist.probs - jax.lax.stop_gradient(dist.probs) # straight-through gradient
            state = state.reshape(*state.shape[:2], -1) # flatten

        return state


# Motivation
class Reward(eqx.Module):
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
        latent_state: Union[Float[Array, "... input_size"], LatentState],
        action: Optional[Float[Array, "... action_size"]] = None,
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


class Perception(eqx.Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(self, encoder, decoder, *, key: PRNGKeyArray):
        key_encoder, key_decoder = jax.random.split(key, 2)
        self.encoder = Encoder(**encoder(), key=key_encoder)
        self.decoder = Decoder(**decoder(), key=key_decoder)


class World(eqx.Module):
    perception: Perception
    representation: Representation
    transition: Transition
    reward: Reward

    def __init__(self, perception, representation, transition, reward, *, key: PRNGKeyArray):
        key_perception, key_representation, key_transition, key_reward = jax.random.split(key, 4)
        self.perception = Perception(**perception(), key=key_perception)
        self.representation = Representation(**representation(), key=key_representation)
        self.transition = Transition(**transition(), key=key_transition)
        self.reward = Reward(**reward(), key=key_reward)
