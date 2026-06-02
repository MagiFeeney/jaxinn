import math
from typing import Optional, Callable, Union, Dict, Tuple, Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import equinox as eqx
from equinox._module import Static
import distrax

from .utils import get_activation_fn, get_precision_fn, dx, StaticCallable, make_mlp


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
            state_size: Union[int, Tuple[int, ...]],
            random_init: bool = False,
            batch_shape: Tuple[int, ...] = (),
            *,
            key: PRNGKeyArray,
    ) -> "LatentState":
        key_belief, key_state = jax.random.split(key, 2)

        if isinstance(state_size, tuple):
            state_size = math.prod(state_size)

        mask = float(random_init)
        belief = jax.random.normal(key_belief, batch_shape + (belief_size,)) * mask
        state  = jax.random.normal(key_state,  batch_shape + (state_size,))  * mask

        return cls(belief=belief, state=state)

    @classmethod
    def concatenate(cls, states: list["LatentState"], axis: int = 0) -> "LatentState":
        return jax.tree.map(lambda *arrays: jnp.concatenate(arrays, axis=axis), *states)

    @property
    def shape(self):
        return self.feature.shape

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

    def detach(self):
        return jax.tree.map(lambda x: jax.lax.stop_gradient(x), self)


class LatentStateWithParams(eqx.Module):
    """
    Store the LatentState along with its parameters
    """
    latent_state: LatentState
    params: Dict[str, jax.Array]
    dist_cls: Callable[..., Any] = eqx.field(static=True)

    @property
    def dist(self):
        return self.dist_cls(**self.params).dist # FixedDistrax -> distrax.Distribution

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dist, name)


# Perception
class CNNEncoder(eqx.Module):
    body: eqx.nn.Sequential
    head: Union[eqx.nn.Linear, eqx.nn.Identity]
    shape: Tuple[int, int, int] = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    embedding_size: int = eqx.field(static=True)
    dtype: str = eqx.field(static=True)

    feature_map_shape: Tuple[int, int, int] = eqx.field(static=True)

    def __init__(
            self,
            shape: Tuple[int, int, int],
            embedding_size: int,
            kernel_size: int = 4,
            depth: int = 32,
            stride: int = 2,
            activation_function: Union[str, Callable] = "elu",
            dtype: str = "float32",
            *,
            key: PRNGKeyArray
    ):
        activation = get_activation_fn(activation_function)
        self.dtype = get_precision_fn(dtype)

        keys = jax.random.split(key, 5)

        self.body = eqx.nn.Sequential([
            eqx.nn.Conv2d(shape[0], 1 * depth, kernel_size=kernel_size, stride=stride, padding="SAME", key=keys[0], dtype=self.dtype),
            StaticCallable(activation),
            eqx.nn.Conv2d(1 * depth, 2 * depth, kernel_size=kernel_size, stride=stride, padding="SAME", key=keys[1], dtype=self.dtype),
            StaticCallable(activation),
            eqx.nn.Conv2d(2 * depth, 4 * depth, kernel_size=kernel_size, stride=stride, padding="SAME", key=keys[2], dtype=self.dtype),
            StaticCallable(activation),
            eqx.nn.Conv2d(4 * depth, 8 * depth, kernel_size=kernel_size, stride=stride, padding="SAME", key=keys[3], dtype=self.dtype),
            StaticCallable(activation),
            StaticCallable(jnp.ravel),
        ])

        self.feature_map_shape = self.get_feature_map_shape(shape)
        feature_map_size = math.prod(self.feature_map_shape)
        self.head = eqx.nn.Linear(feature_map_size, embedding_size, key=keys[4])

        self.shape = shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.stride = stride
        self.embedding_size = embedding_size

    def __call__(
            self,
            obs: Float[Array, "... obs_size"]
    ) -> Float[Array, "... output_size"]:
        obs = obs.astype(self.dtype)
        feature = eqx.filter_checkpoint(self.body)(obs)
        feature = feature.astype(jnp.float32) # upcast for stability
        out = self.head(feature)
        return out

    def get_feature_map_shape(self, shape) -> int:
        dummy_input = jnp.zeros(shape, dtype=self.dtype)
        out = jax.eval_shape(self.body[:-1], dummy_input) # exclude ravel
        return out.shape


class CNNDecoder(eqx.Module):
    embedding: eqx.nn.Linear
    body: eqx.nn.Sequential
    shape: Tuple[int, int, int] = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    dtype: str = eqx.field(static=True)

    feature_map_shape: Tuple[int, int, int] = eqx.field(static=True)

    def __init__(
            self,
            belief_size: int,
            state_size: Union[int, Tuple[int, ...]],
            shape: Tuple[int, int, int],
            feature_map_shape: Tuple[int, int, int],
            kernel_size: int = 4,
            depth: int = 32,
            stride: int = 2,
            activation_function: Union[str, Callable] = "elu",
            dtype: str = "float32",
            *,
            key: PRNGKeyArray
    ):
        activation = get_activation_fn(activation_function)
        self.dtype = get_precision_fn(dtype)
        self.feature_map_shape = feature_map_shape
        feature_map_size = math.prod(feature_map_shape)

        keys = jax.random.split(key, 5)

        if isinstance(state_size, tuple):
            state_size = math.prod(state_size)

        self.embedding = eqx.nn.Linear(belief_size + state_size, feature_map_size, key=keys[0])
        self.body = eqx.nn.Sequential([
            StaticCallable(activation),
            eqx.nn.ConvTranspose2d(feature_map_shape[0], 4 * depth, kernel_size=kernel_size, stride=stride, padding="SAME", key=keys[1], dtype=self.dtype),
            StaticCallable(activation),
            eqx.nn.ConvTranspose2d(4 * depth, 2 * depth, kernel_size=kernel_size, stride=stride, padding="SAME", key=keys[2], dtype=self.dtype),
            StaticCallable(activation),
            eqx.nn.ConvTranspose2d(2 * depth, 1 * depth, kernel_size=kernel_size, stride=stride, padding="SAME", key=keys[3], dtype=self.dtype),
            StaticCallable(activation),
            eqx.nn.ConvTranspose2d(1 * depth, shape[0], kernel_size=kernel_size, stride=stride, padding="SAME", key=keys[4], dtype=self.dtype),
        ])

        self.shape = shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.stride = stride

    def __call__(
            self,
            latent_state: Union[Float[Array, "... input_size"], LatentState],
    ) -> distrax.Distribution:
        if isinstance(latent_state, LatentState):
            latent_state = latent_state.feature
        embedding = self.embedding(latent_state)
        embedding = embedding.reshape(embedding.shape[:-1] + self.feature_map_shape).astype(self.dtype)
        out = eqx.filter_checkpoint(self.body)(embedding)
        out = out.astype(jnp.float32)

        if out.shape[-3:] != self.shape:
            out = jax.image.resize(out, shape=out.shape[:-3] + self.shape, method="bilinear")

        dist = dx.Normal(out, jnp.ones_like(out))
        return dx.Independent(dist, reinterpreted_batch_ndims=Static(len(self.shape)))


class LinearEncoder(eqx.Module):
    net: eqx.Module

    def __init__(
            self,
            shape: Tuple[int, ...],
            hidden_size: Optional[int] = None,
            embedding_size: Optional[int] = None,
            num_layers: Optional[int] = None,
            activation_function: Union[str, Callable] = "elu",
            *,
            key: PRNGKeyArray
    ):
        assert len(shape) == 1, (
            f"Expected a 1D shape, but got shape {shape} with {len(shape)} dimensions "
            f"in {self.__class__.__name__}."
        )
        activation = get_activation_fn(activation_function)

        if hidden_size is not None and \
           embedding_size is not None and \
           num_layers is not None:
            self.net = make_mlp(
                input_size = shape[0],
                hidden_size = hidden_size,
                output_size = embedding_size,
                num_layers = num_layers,
                activation = StaticCallable(activation),
                key = key
            )
        else:
            self.net = eqx.nn.Identity()

    def __call__(
            self,
            obs: Float[Array, "... obs_size"]
    ) -> Float[Array, "... output_size"]:
        return self.net(obs)


class LinearDecoder(eqx.Module):
    net: eqx.Module

    def __init__(
            self,
            shape: Tuple[int, ...],
            belief_size: int,
            state_size: Union[int, Tuple[int, ...]],
            hidden_size: int,
            num_layers: int,
            activation_function: Union[str, Callable] = "elu",
            *,
            key: PRNGKeyArray
    ):
        assert len(shape) == 1, (
            f"Expected a 1D shape, but got shape {shape} with {len(shape)} dimensions "
            f"in {self.__class__.__name__}."
        )
        activation = get_activation_fn(activation_function)

        if isinstance(state_size, tuple):
            state_size = math.prod(state_size)

        self.net = make_mlp(
            input_size = belief_size + state_size,
            hidden_size = hidden_size,
            output_size = shape[0],
            num_layers = num_layers,
            activation = StaticCallable(activation),
            key = key
        )

    def __call__(
            self,
            latent_state: Union[Float[Array, "... input_size"], LatentState],
    ) -> distrax.Distribution:
        if isinstance(latent_state, LatentState):
            latent_state = latent_state.feature
        out = self.net(latent_state)
        dist = dx.Normal(out, jnp.ones_like(out))
        return dx.Independent(dist, reinterpreted_batch_ndims=Static(1))


# Representation
class Representation(eqx.Module):
    """Representation learning of state, inferred from history and the latest observation: p(s_t | h_t, o_t)
    """
    net: eqx.nn.Sequential
    dist_cls: str = eqx.field(static=True)
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
            self.dist_cls = dx.Normal
            self.num_variables, self.num_categories = state_size, 0
            output_size = 2 * state_size
        elif head_type == "Categorical":
            self.dist_cls = dx.OneHotCategorical
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
    min_std: float = eqx.field(static=True)

    def __init__(
            self,
            belief_size: int,
            state_size: Union[int, Tuple[int, ...]],
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
            assert isinstance(state_size, tuple) and len(state_size) == 2, (
                f"Expected `state_size` to be a 2-element tuple (representing a stack of "
                f"independent categorical distributions), but got {state_size!r}."
            )
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
            StaticCallable(activation),
        ])

        # p(h_t | c_{t - 1}, h_{t - 1})
        self.body = eqx.nn.GRUCell(hidden_size, belief_size, key=keys[1])

        # p(s_t | h_t)
        self.head = eqx.nn.Sequential([
            eqx.nn.Linear(belief_size, hidden_size, key=keys[2]),
            StaticCallable(activation),
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
    min_std: float = eqx.field(static=True)

    def __init__(
            self,
            belief_size: int,
            state_size: Union[int, Tuple[int, ...]],
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

        if isinstance(state_size, tuple):
            state_size = math.prod(state_size)

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


# Add your own registries!
PERCEPTION_REGISTRY = {
    "cnn": (CNNEncoder, CNNDecoder),
    "linear": (LinearEncoder, LinearDecoder)
}


EncoderType = Union[CNNEncoder, LinearEncoder]
DecoderType = Union[CNNDecoder, LinearDecoder]


class Perception(eqx.Module):
    encoder: EncoderType
    decoder: DecoderType

    def __init__(self, type, domain, encoder, decoder, *, key: PRNGKeyArray):
        key_encoder, key_decoder = jax.random.split(key, 2)

        encoder_cls, decoder_cls = PERCEPTION_REGISTRY[type]

        if len(encoder.shape) == 1:
            assert domain == "state" and domain == "state" , \
                f"State-based task (1D) requires a state module. Got {type}."
        else:
            assert domain == "pixel" and domain == "pixel" , \
                f"Pixel-based task (3D+) requires a spatial module. Got {type}."

        self.encoder = encoder_cls(**encoder(), key=key_encoder)
        if type == "cnn":
            self.decoder = decoder_cls(feature_map_shape=self.encoder.feature_map_shape, **decoder(), key=key_decoder)
        else:
            self.decoder = decoder_cls(**decoder(), key=key_decoder)


class World(eqx.Module):
    perception: Perception
    representation: Representation
    transition: Transition
    reward: Reward

    def __init__(self, perception, representation, transition, reward, *, key: PRNGKeyArray):
        key_perception, key_representation, key_transition, key_reward = jax.random.split(key, 4)
        self.perception = Perception(perception.type, perception.domain, **perception(), key=key_perception)
        self.representation = Representation(**representation(), key=key_representation)
        self.transition = Transition(**transition(), key=key_transition)
        self.reward = Reward(**reward(), key=key_reward)
