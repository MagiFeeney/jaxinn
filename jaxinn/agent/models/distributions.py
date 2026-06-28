from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx
import distrax


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

    def mean(self, seed: PRNGKeyArray) -> jax.Array:
        samples = self.dist.sample(
            seed=seed, sample_shape=(self.num_samples,)
        )
        return jnp.mean(samples, axis=0)

    def mode(self, seed: PRNGKeyArray) -> jax.Array:
        samples = self.dist.sample(
            seed=seed, sample_shape=(self.num_samples,)
        )
        logprobs = self.dist.log_prob(samples)

        indices = jnp.argmax(logprobs, axis=0, keepdims=True)
        mode = jnp.take_along_axis(
            samples, indices[..., None], axis=0
        )
        return jnp.squeeze(mode, axis=0)

    def entropy(self, seed: PRNGKeyArray) -> jax.Array:
        samples = self.dist.sample(
            seed=seed, sample_shape=(self.num_samples,)
        )
        logprobs = self.dist.log_prob(samples)
        return -jnp.mean(logprobs, axis=0)

    def sample(self, seed: PRNGKeyArray) -> jax.Array:
        return self.dist.sample(seed=seed)

    def log_prob(self, value: jax.Array) -> jax.Array:
        return self.dist.log_prob(value)
