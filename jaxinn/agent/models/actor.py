import math
from typing import Tuple, Union, Any, Optional, Callable, Dict

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import equinox as eqx
from equinox._module import Static
import distrax

from jaxinn.structs import LatentState

from .utils import make_mlp, dx, StaticCallable, FactoryLike


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
    dist_cls: FactoryLike = eqx.field(static=True)
    action_size: int = eqx.field(static=True)
    head_type: str = eqx.field(static=True)
    init_std: float = eqx.field(static=True)
    min_std: float = eqx.field(static=True)
    mean_scale: float = eqx.field(static=True)
    raw_init_std: float = eqx.field(static=True)

    def __init__(
        self,
        belief_size: int,
        state_size: Union[int, Tuple[int, ...]],
        hidden_size: list[int],
        action_size: int,
        head_type: str = "Tanh Normal",
        activation_function: Union[str, Callable] = "elu",
        min_std: float = 1e-4,
        init_std: float = 5.0,
        mean_scale: float = 5.0,
        *,
        key: PRNGKeyArray,
    ):
        output_size = 2 * action_size

        if head_type == "Beta":
            self.dist_cls = dx.Independent(dx(AffineBeta), reinterpreted_batch_ndims=Static(1)) # Explicitly make non jax array as static to prevent being vmapped
        elif head_type == "Tanh Normal":
            self.dist_cls = dx.Independent(dx(TanhNormal), reinterpreted_batch_ndims=Static(1))
        elif head_type == "Categorical":
            self.dist_cls = dx.OneHotCategorical
            output_size = action_size

        if isinstance(state_size, tuple):
            state_size = math.prod(state_size)

        # Build network
        self.net = make_mlp(
            input_size = belief_size + state_size,
            hidden_size = hidden_size,
            output_size = output_size,
            activation = activation_function,
            key = key
        )

        self.head_type = head_type
        self.action_size = action_size
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.raw_init_std = math.log(math.exp(init_std) - 1)

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
        elif self.head_type == "Categorical":
            logit = out
            params = {"logits": logit}

        return params

    def get_dist(
            self,
            params: Dict[str, Any]
    ) -> distrax.Distribution:
        dist = self.dist_cls(**params)
        if self.head_type == "Categorical":
            return dist
        return SampleDist(dist)

    def sample(
            self,
            params: Dict[str, Any],
            key: PRNGKeyArray,
            det: bool = False,      # default to training
    ) -> Float[Array, "... action_size"]:
        dist = self.get_dist(params)

        if self.head_type == "Categorical":
            # Same for both train and eval
            action = dist.sample(seed=key)
            action = action + dist.probs - jax.lax.stop_gradient(dist.probs) # straight-through gradient
            return action

        if det:
            return dist.mode(seed=key)
        else:
            return dist.sample(seed=key)
