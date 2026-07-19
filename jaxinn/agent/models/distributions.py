import abc
import math
from typing import Tuple, Optional, Any, TypeAlias, Dict
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree, PyTreeDef
import equinox as eqx
import distrax

from .utils import FixedDistrax


class Distribution(eqx.Module):
    dist: "DistributionLike"

    def __getattr__(self, name):
        return getattr(self.dist, name)


class JointDistribution(eqx.Module):
    dists: PyTree["DistributionLike"]

    @abc.abstractmethod
    def sample(self, *, seed: PRNGKeyArray, sample_shape=()) -> PyTree[jax.Array]:
        pass

    @abc.abstractmethod
    def log_prob(self, x: PyTree[jax.Array]) -> jax.Array:
        pass

    @abc.abstractmethod
    def mode(self) -> PyTree[jax.Array]:
        pass


Primitive: TypeAlias = FixedDistrax | distrax.Distribution
DistributionLike: TypeAlias =  Primitive | Distribution | JointDistribution


class TanhNormal(distrax.Transformed):
    """Normal distribution transformed by a Tanh transformation: X ↦ tanh(X)."""

    def __init__(self, mean, std):
        _distribution = distrax.Normal(mean, std)
        transform = distrax.Tanh()
        super().__init__(_distribution, transform)

    # Approximate mean after Tanh transformation as tanh(mean)
    def mean(self) -> jax.Array:
        return jnp.tanh(self.distribution.mean())

    def log_prob(self, value: jax.Array) -> jax.Array:
        # Clip value to avoid NaNs at the boundaries of tanh (-1 and 1) during inverse of it by arctanh(x)
        safe_value = jnp.clip(value, -1.0 + 1e-7, 1.0 - 1e-7)
        return super().log_prob(safe_value)


class PatchedBeta(distrax.Beta):
    """Patched Beta to correct the batch_shape when used with vmap."""

    @property
    def batch_shape(self) -> Tuple[int, ...]:
      return jax.lax.broadcast_shapes(self._alpha.shape, self._beta.shape)


class AffineBeta(distrax.Transformed):
    """Beta distribution with an affine transformation: X ↦ loc + scale · X.

    Attributes:
        loc (jax.Array): Location parameter.
        scale (jax.Array): Scale parameter.
    """

    def __init__(
            self,
            alpha: jax.Array,
            beta: jax.Array,
            loc: jax.Array = 0.0,
            scale: jax.Array = 1.0,
    ):
        _distribution = PatchedBeta(alpha=alpha, beta=beta)

        loc = jnp.broadcast_to(loc, alpha.shape)
        scale = jnp.broadcast_to(scale, alpha.shape)
        transform = distrax.ScalarAffine(shift=loc, scale=scale)
        super().__init__(_distribution, transform)


class StraightThroughOneHotCategorical(distrax.OneHotCategorical):
    """A differentiable OneHotCategorical distribution using the straight-through gradient estimator."""

    def sample(self, *, seed: PRNGKeyArray, sample_shape=()) -> jax.Array:
        sample = super().sample(seed=seed, sample_shape=sample_shape)
        sample = sample + self.probs - jax.lax.stop_gradient(self.probs)
        return sample

    def mode(self) -> jax.Array:
        mode = super().mode
        return mode.astype(jnp.float32) # align with sample's dtype


class SampleDist(Distribution):
    """Provides sample-based estimates of distribution attributes when closed-form expressions are unavailable.

    Works for either pure or nested distributions.
    """

    num_samples: int = eqx.field(static=True, default=100)

    def mean(self, seed: PRNGKeyArray) -> jax.Array:
        samples = self.dist.sample(
            seed=seed, sample_shape=(self.num_samples,)
        )
        return jax.tree.map(lambda x: jnp.mean(x, axis=0), samples)

    def mode(self, seed: PRNGKeyArray) -> jax.Array:
        samples = self.dist.sample(
            seed=seed, sample_shape=(self.num_samples,)
        )
        logprobs = self.dist.log_prob(samples)
        indices = jnp.argmax(logprobs, axis=0, keepdims=True)

        def _extract_mode(x):
            event_dims = x.ndim - indices.ndim
            idx = indices.reshape(indices.shape + (1,) * event_dims)

            mode = jnp.take_along_axis(x, idx, axis=0)
            return jnp.squeeze(mode, axis=0)

        return jax.tree.map(_extract_mode, samples)

    def entropy(self, seed: PRNGKeyArray) -> jax.Array:
        samples = self.dist.sample(
            seed=seed, sample_shape=(self.num_samples,)
        )
        logprobs = self.dist.log_prob(samples)
        return -jnp.mean(logprobs, axis=0)

    def sample(self, *, seed: PRNGKeyArray, sample_shape=()) -> jax.Array:
        return self.dist.sample(seed=seed)

    def log_prob(self, value: jax.Array) -> jax.Array:
        return self.dist.log_prob(value)


class FlattenDist(Distribution):
    """Flatten the samples into 1D vector and inflates them back for log-prob.

    Useful for network input.
    """

    def sample(self, *, seed: PRNGKeyArray, sample_shape=()) -> jax.Array:
        sample = self.dist.sample(seed=seed, sample_shape=sample_shape)
        event_shape = self.dist.event_shape
        if len(event_shape) == 0:
            return sample
        sample = sample.reshape(*sample.shape[:-len(event_shape)], -1)
        return sample

    def log_prob(self, x: jax.Array) -> jax.Array:
        event_shape = self.dist.event_shape
        if len(event_shape) > 0:
            x = x.reshape(*x.shape[:-1], *event_shape)
        return self.dist.log_prob(x)

    def mode(self) -> jax.Array:
        mode = self.dist.mode()
        event_shape = self.dist.event_shape
        if len(event_shape) == 0:
            return mode
        return mode.reshape(*mode.shape[:-len(event_shape)], -1)


class IndependentJointDistribution(JointDistribution):
    """Wraps any sequence of independent distributions into a single joint distribution."""

    dists: Tuple[DistributionLike, ...]
    target_shape: Optional[Tuple[int, ...]] = eqx.field(static=True)

    def __init__(
            self,
            dists: Tuple[DistributionLike, ...],
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


class TreeJointDistribution(JointDistribution):
    """Wraps a PyTree of independent distributions into a single joint distribution."""

    dists: PyTree[Any]
    treedef: PyTreeDef = eqx.field(static=True)
    is_leaf: callable = eqx.field(static=True)

    def __init__(self, dists: PyTree[DistributionLike]):
        self.dists = dists
        self.is_leaf = lambda x: isinstance(x, DistributionLike)
        self.treedef = jax.tree.structure(dists, is_leaf=self.is_leaf)

    def sample(self, *, seed: PRNGKeyArray, sample_shape=()) -> PyTree[jax.Array]:
        num_leaves = self.treedef.num_leaves
        keys = jax.random.split(seed, num_leaves)
        keys_tree = jax.tree.unflatten(self.treedef, keys)
        return jax.tree.map(
            lambda d, k: d.sample(seed=k, sample_shape=sample_shape),
            self.dists,
            keys_tree,
            is_leaf=self.is_leaf
        )

    def log_prob(self, x: PyTree[jax.Array]) -> jax.Array:
        log_probs_tree = jax.tree.map(lambda d, v: d.log_prob(v), self.dists, x, is_leaf=self.is_leaf)
        return jax.tree.reduce(jnp.add, log_probs_tree)

    def entropy(self) -> jax.Array:
        entropies_tree = jax.tree.map(lambda d: d.entropy(), self.dists, is_leaf=self.is_leaf)
        return jax.tree.reduce(jnp.add, entropies_tree)

    def mode(self) -> PyTree[jax.Array]:
        return jax.tree.map(lambda d: d.mode(), self.dists, is_leaf=self.is_leaf)


class HierarchicalJointDistribution(TreeJointDistribution):
    """A hierarchical distribution with exactly one active branch at a time.

    Expects a PyTree dictionary containing the keys 'option' and 'actions'.
    """

    def _mask_branches(self, option: jax.Array, branch_tree: Dict[str, Any], reduce_to_scalar: bool) -> Any:
        """Helper function to mask inactive branches.

        If reduce_to_scalar is True, sums the active branch leaves and returns a scalar.
        If False, zeroes the inactive leaves and returns the full PyTree dictionary.
        """
        sorted_keys = sorted(self.dists["actions"].keys())

        if reduce_to_scalar:
            masked_scalars = []
            for i, branch_key in enumerate(sorted_keys):
                is_active = (option == i)
                branch_sum = jax.tree.reduce(jnp.add, branch_tree[branch_key])
                masked_scalars.append(jnp.where(is_active, branch_sum, 0.0))

            return jnp.sum(jnp.stack(masked_scalars), axis=0)

        else:
            masked_dict = {}
            for i, branch_key in enumerate(sorted_keys):
                is_active = (option == i)
                masked_dict[branch_key] = jax.tree.map(
                    lambda x: jnp.where(is_active, x, jnp.zeros_like(x)),
                    branch_tree[branch_key]
                )

            return masked_dict

    def sample(self, *, seed: PRNGKeyArray, sample_shape=()) -> Dict[str, jax.Array]:
        key_option, key_branch = jax.random.split(seed)
        option = self.dists["option"].sample(seed=key_option, sample_shape=sample_shape)

        zero_tree = jax.tree.map(
            lambda d: jnp.zeros(d.event_shape, dtype=d.dtype),
            self.dists["actions"],
            is_leaf=self.is_leaf
        )

        def _sample_single_branch(key, selector):
            active_sample = self.dists["actions"][selector].sample(seed=key)
            return {**zero_tree, selector: active_sample}

        sorted_keys = sorted(self.dists["actions"].keys())
        branch_fns = [partial(_sample_single_branch, selector=k) for k in sorted_keys]

        if sample_shape == ():
            actions = jax.lax.switch(
                option,
                branch_fns,
                key_branch
            )
        else:
            keys_branch = jax.random.split(key_branch, math.prod(sample_shape))
            flat_option = option.reshape(-1)
            flat_actions = jax.vmap(jax.lax.switch, in_axes=(0, None, 0))(
                flat_option,
                branch_fns,
                keys_branch
            )

            actions = jax.tree.map(
                lambda x: x.reshape(sample_shape + x.shape[1:]),
                flat_actions
            )

        return {
            "option": option,
            "actions": actions
        }

    def log_prob(self, x: Dict[str, jax.Array]) -> jax.Array:
        option = x["option"]
        option_log_prob = self.dists["option"].log_prob(option)

        branch_log_probs = jax.tree.map(
            lambda d, v: d.log_prob(v),
            self.dists["actions"],
            x["actions"],
            is_leaf=self.is_leaf
        )

        return option_log_prob + self._mask_branches(option, branch_log_probs, reduce_to_scalar=True)

    def entropy(self, x: Dict[str, jax.Array]) -> jax.Array:
        option = x["option"]
        option_entropy = self.dists["option"].entropy()

        branch_entropies = jax.tree.map(
            lambda d: d.entropy(),
            self.dists["actions"],
            is_leaf=self.is_leaf
        )

        return option_entropy + self._mask_branches(option, branch_entropies, reduce_to_scalar=True)

    def mode(self) -> Dict[str, jax.Array]:
        option_mode = self.dists["option"].mode()
        branch_modes = jax.tree.map(
            lambda d: d.mode(),
            self.dists["actions"],
            is_leaf=self.is_leaf
        )

        return {
            "option": option_mode,
            "actions": self._mask_branches(option_mode, branch_modes, reduce_to_scalar=False)
        }
