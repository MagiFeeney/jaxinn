import math
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree, PyTreeDef
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


class IndependentJointDistribution(eqx.Module):
    """Wraps any sequence of independent distributions into a single joint distribution."""

    dists: Tuple[distrax.Distribution, ...]
    target_shape: Optional[Tuple[int, ...]] = eqx.field(static=True)

    def __init__(
            self,
            dists: Tuple[distrax.Distribution, ...],
            target_shape: Optional[Tuple[int, ...]] = None
    ):
        if target_shape is not None and math.prod(target_shape) != len(dists):
            raise ValueError(
                f"target_shape product must match the number of flattened dists. "
                f"Got shape {target_shape} of product {math.prod(target_shape)} "
                f"for {len(dists)} dists."
            )

        self.target_shape = target_shape if target_shape is not None else (len(dists),)
        self.dists = dists

    def sample(self, *, seed: PRNGKeyArray, sample_shape=()) -> jax.Array:
        keys = jax.random.split(seed, len(self.dists))
        samples = [d.sample(seed=k, sample_shape=sample_shape) for d, k in zip(self.dists, keys)]
        stacked = jnp.stack(samples, axis=-1)
        return stacked.reshape(*stacked.shape[:-1], *self.target_shape)

    def log_prob(self, x: jax.Array) -> jax.Array:
        x = x.reshape(*x.shape[:-len(self.target_shape)], -1)
        log_probs = [d.log_prob(x[..., i]) for i, d in enumerate(self.dists)]
        return sum(log_probs)

    def entropy(self) -> jax.Array:
        return sum(d.entropy() for d in self.dists)

    def mode(self) -> jax.Array:
        modes = [d.mode() for d in self.dists]
        stacked = jnp.stack(modes, axis=-1)
        return stacked.reshape(*stacked.shape[:-1], *self.target_shape)


class TreeJointDistribution(eqx.Module):
    """Wraps a PyTree of independent distributions into a single joint distribution."""

    dists_tree: PyTree
    dists_treedef: PyTreeDef

    def __init__(self, dists_tree: PyTree[distrax.Distribution]):
        self.dists_tree = dists_tree
        self.dists_treedef = jax.tree.structure(dists_tree)

    def sample(self, *, seed: PRNGKeyArray, sample_shape=()) -> PyTree[jax.Array]:
        num_leaves = self.dists_treedef.num_leaves
        keys = jax.random.split(seed, num_leaves)
        keys_tree = jax.tree.unflatten(self.dists_treedef, keys)
        return jax.tree.map(
            lambda d, k: d.sample(seed=k, sample_shape=sample_shape),
            self.dists_tree,
            keys_tree
        )

    def log_prob(self, x: PyTree[jax.Array]) -> jax.Array:
        log_probs_tree = jax.tree.map(lambda d, v: d.log_prob(v), self.dists_tree, x)
        log_probs_leaves = jax.tree.leaves(log_probs_tree)
        return jnp.sum(jnp.stack(log_probs_leaves), axis=0)

    def entropy(self) -> jax.Array:
        entropies_tree = jax.tree.map(lambda d: d.entropy(), self.dists_tree)
        entropies_leaves = jax.tree.leaves(entropies_tree)
        return jnp.sum(jnp.stack(entropies_leaves), axis=0)

    def mode(self, seed: PRNGKeyArray) -> PyTree[jax.Array]:
        return jax.tree.map(lambda d: d.mode(), self.dists_tree)
