import jax
import jax.numpy as jnp
import equinox as eqx
from equinox._module import Static
import distrax

from typing import Tuple, Union, Any, Optional, Callable, Dict
from jaxtyping import Array, Float, PRNGKeyArray
from .utils import get_activation_fn, dx
from .world import LatentState


class TanhNormal(distrax.Transformed):
    """Normal distribution transformed by an Tanh transformation: X ↦ tanh(X)."""

    def __init__(self, mean, std):
        _distribution = distrax.Normal(mean, std)
        transform = distrax.Tanh()
        super().__init__(_distribution, transform)

    # Approximate mean after Tanh transformation as tanh(base_mean)
    def mean(self) -> jnp.ndarray:
        return jnp.tanh(self.distribution.mean())

    def sample(self, seed: PRNGKeyArray, sample_shape=()):
        return super().sample(seed=seed, sample_shape=sample_shape)


# Patched version: correct the batch_shape when used with vmap
class PatchedBeta(distrax.Beta):
    @property
    def batch_shape(self) -> Tuple[int, ...]:
      """Shape of batch of distribution samples."""
      return jax.lax.broadcast_shapes(self._alpha.shape, self._beta.shape)


class AffineBeta(distrax.Transformed):
    """Beta distribution with an affine transformation: X ↦ loc + scale · X.

    Attributes:
        loc (jnp.ndarray): Location parameter.
        scale (jnp.ndarray): Scale parameter.
    """

    def __init__(
            self,
            alpha: jnp.ndarray,
            beta: jnp.ndarray,
            loc: jnp.ndarray = 0.0,
            scale: jnp.ndarray = 1.0,
    ):
        _distribution = PatchedBeta(alpha=alpha, beta=beta)

        loc = jnp.broadcast_to(loc, alpha.shape)
        scale = jnp.broadcast_to(scale, alpha.shape)
        transform = distrax.ScalarAffine(shift=loc, scale=scale)
        super().__init__(_distribution, transform)

    def mode(self) -> Optional[jnp.ndarray]:
        return super().mode()

    def sample(self, seed: PRNGKeyArray, sample_shape=()):
        return super().sample(seed=seed, sample_shape=sample_shape)


class SampleDist(eqx.Module):
    dist: distrax.Distribution
    num_samples: int = eqx.field(static=True)

    def __init__(self, dist, num_samples=100):
        self.dist = dist
        self.num_samples = num_samples

    def mean(self, seed: PRNGKeyArray) -> Float[Array, "..."]:
        samples = self.dist.sample(
            seed=seed, sample_shape=(self.num_samples,)
        )
        return jnp.mean(samples, axis=0)

    def mode(self, seed: PRNGKeyArray) -> Float[Array, "..."]:
        samples = self.dist.sample(
            seed=seed, sample_shape=(self.num_samples,)
        )
        logprobs = self.dist.log_prob(samples)

        indices = jnp.argmax(logprobs, axis=0, keepdims=True)
        mode = jnp.take_along_axis(
            samples, indices[..., None], axis=0
        )
        return jnp.squeeze(mode, axis=0)

    def entropy(self, seed: PRNGKeyArray) -> Float[Array, "..."]:
        samples = self.dist.sample(
            seed=seed, sample_shape=(self.num_samples,)
        )
        logprobs = self.dist.log_prob(samples)
        return -jnp.mean(logprobs, axis=0)

    def sample(self, seed: PRNGKeyArray) -> Float[Array, "..."]:
        return self.dist.sample(seed=seed)

    def log_prob(self, value: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.dist.log_prob(value)


class Actor(eqx.Module):
    net: eqx.nn.Sequential
    action_size: int = eqx.field(static=True)
    head_type: str = eqx.field(static=True)
    init_std: float = eqx.field(static=True)
    min_std: float
    mean_scale: float
    raw_init_std: float

    def __init__(
        self,
        belief_size: int,
        state_size: int,
        hidden_size: int,
        action_size: int,
        head_type: str = "Tanh Normal",
        activation_function: Union[str, Callable] = "elu",
        min_std: float = 1e-4,
        init_std: float = 5.0,
        mean_scale: float = 5.0,
        *,
        key: PRNGKeyArray,
    ):
        activation = get_activation_fn(activation_function)

        keys = jax.random.split(key, 5)
        # Build network
        self.net = eqx.nn.Sequential([
            eqx.nn.Linear(belief_size + state_size, hidden_size, key=keys[0]),
            eqx.nn.Lambda(activation),
            eqx.nn.Linear(hidden_size, hidden_size, key=keys[1]),
            eqx.nn.Lambda(activation),
            eqx.nn.Linear(hidden_size, hidden_size, key=keys[2]),
            eqx.nn.Lambda(activation),
            eqx.nn.Linear(hidden_size, hidden_size, key=keys[3]),
            eqx.nn.Lambda(activation),
            eqx.nn.Linear(hidden_size, 2 * action_size, key=keys[4]),
        ])

        self.head_type = head_type
        self.action_size = action_size
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.raw_init_std = jnp.log(jnp.exp(init_std) - 1)

    def __call__(
        self,
        latent_state: Union[Float[Array, "... input_size"], LatentState],
    ) -> distrax.Distribution:
        if isinstance(latent_state, LatentState):
            latent_state = latent_state.feature

        out = self.net(latent_state)

        if self.head_type == "Beta":
            alpha_beta = jax.nn.softplus(out) + self.min_std
            alpha, beta = jnp.split(alpha_beta, 2, axis=-1)
            params = {"alpha": alpha, "beta": beta}
        elif self.head_type == "Tanh Normal":
            mean, log_std = jnp.split(out, 2, axis=-1)
            mean = self.mean_scale * jnp.tanh(mean / self.mean_scale)
            std = jax.nn.softplus(log_std + self.raw_init_std) + self.min_std
            params = {"mean": mean, "std": std}

        return params

    def sample(
            self,
            params: Dict[str, Any],
            key: PRNGKeyArray,
            det: bool = False,      # default to training
    ) -> Float[Array, "... action_size"]:
        if self.head_type == "Beta":
            dist = dx(AffineBeta)(**params)
        elif self.head_type == "Tanh Normal":
            dist = dx(TanhNormal)(**params)

        dist = dx.Independent(dist, reinterpreted_batch_ndims=Static(1)) # Explicitly make non jax array as static to prevent being vmapped

        sample_dist = SampleDist(dist)

        return jax.lax.cond(
            det,
            lambda: sample_dist.mode(seed=key),
            lambda: sample_dist.sample(seed=key), # TODO: add action_noise
        )
