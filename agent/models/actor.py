import jax
import jax.numpy as jnp
import equinox as eqx
import distrax

from typing import Tuple, Union, Any, Optional
from jaxtyping import Array, Float, PRNGKeyArray
from .utils import get_activation_fn


class TanhNormal(distrax.Transformed):
    """
    Normal distribution transformed by an Tanh transformation: X -> tanh(X)
    """

    def __init__(self, mean, std):
        _distribution = distrax.Normal(mean, std)
        transform = distrax.Tanh()
        super().__init__(_distribution, transform)

    def mean(self, approximate=False) -> jnp.ndarray:
        """
        Approximate mean after Tanh transformation as tanh(base_mean) if `approximate` is set True
        """
        if approximate:
            return jnp.tanh(self.distribution.mean())
        else:
            raise NotImplementedError

    def sample(self, key: PRNGKeyArray, sample_shape=()):
        return super().sample(seed=seed, sample_shape=sample_shape)


class AffineBeta(distrax.Transformed):
    """
    Beta distribution transformed by an affine transformation: X -> loc + scale * X
    """

    def __init__(
            self,
            alpha: jnp.ndarray,
            beta: jnp.ndarray,
            loc: jnp.ndarray = 0.0,
            scale: jnp.ndarray = 1.0
    ):
        _distribution = distrax.Beta(alpha=alpha, beta=beta)
        transform = distrax.ScalarAffine(shift=loc, scale=scale)
        super().__init__(_distribution, transform)

    def mode(self) -> Optional[jnp.ndarray]:
        base_mode = super().mode()
        # In Beta, mode is undefined (NaN) for alpha<=1 or beta<=1
        if jnp.isnan(base_mode).any():
            return None
        return base_mode

    def sample(self, seed: PRNGKeyArray, sample_shape=()):
        return super().sample(seed=seed, sample_shape=sample_shape)


class SampleDist:
    def __init__(self, dist: distrax.Distribution, samples: int = 100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self) -> str:
        return "SampleDist"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dist, name)

    def mean(self, seed: PRNGKeyArray) -> Float[Array, "..."]:
        if hasattr(self._dist, "mean") and self._dist.mean() is not None:
            return self._dist.mean()
        samples = self._dist.sample(seed=seed, sample_shape=(samples,))
        return jnp.mean(samples, axis=0)

    def mode(self, seed: PRNGKeyArray) -> Float[Array, "..."]:
        if hasattr(self._dist, "mode") and self._dist.mode() is not None:
            return self._dist.mode()
        samples = self._dist.sample(seed=seed, sample_shape=(self._samples,))
        logprobs = self._dist.log_prob(samples)

        indices = jnp.argmax(logprobs, axis=0, keepdims=True)
        mode = jnp.take_along_axis(samples, indices[..., None], axis=0)

        return jnp.squeeze(mode, axis=0)

    def entropy(self, seed: PRNGKeyArray) -> Float[Array, "..."]:
        if hasattr(self._dist, "entropy") and self._dist.entropy() is not None:
            return self._dist.entropy()
        samples = self._dist.sample(seed=seed, sample_shape=(self._samples,))
        logprobs = self._dist.log_prob(samples)

        return -jnp.mean(logprobs, axis=0)

    def sample(self, seed: PRNGKeyArray) -> Float[Array, "..."]:
        return self._dist.sample(seed=seed)

    def rsample(self, seed: PRNGKeyArray) -> Float[Array, "..."]:
        return self._dist.sample(seed=seed)

    def log_prob(self, value: Float[Array, "..."]) -> Float[Array, "..."]:
        return self._dist.log_prob(value)


class ActorModel(eqx.Module):
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
        activation_function: str = "elu",
        min_std: float = 1e-4,
        init_std: float = 5.0,
        mean_scale: float = 5.0,
        *,
        key: jax.random.PRNGKey,
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
        input_tensor: Float[Array, "... input_dim"]
    ) -> distrax.Distribution:
        out = self.net(input_tensor)

        if self.head_type == "Beta":
            alpha_beta = jax.nn.softplus(out) + self.min_std
            alpha, beta = jnp.split(alpha_beta, 2, axis=-1)
            dist = AffineBeta(alpha=alpha, beta=beta)
        elif self.head_type == "Tanh Normal":
            mean, log_std = jnp.split(out, 2, axis=-1)
            mean = self.mean_scale * jnp.tanh(mean / self.mean_scale)
            std = jax.nn.softplus(log_std + self.raw_init_std) + self.min_std
            dist = TanhNormal(mean, std)
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")

        return distrax.Independent(dist, reinterpreted_batch_ndims=1)

    def get_action(
        self,
        input_tensor: Float[Array, "... input_dim"],
        det: bool = False,
        *,
        key: jax.random.PRNGKey,
    ) -> Float[Array, "... action_dim"]:
        base_dist = self(input_tensor)
        sample_dist = SampleDist(base_dist)

        if det:
            return sample_dist.mode(seed=key)
        else:
            if key is None:
                raise ValueError("key must be provided for stochastic sampling")
            return sample_dist.sample(seed=key)
