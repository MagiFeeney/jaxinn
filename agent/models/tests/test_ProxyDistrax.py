import jax
import jax.numpy as jnp
import functools
from typing import Any, Callable
from jaxtyping import PyTree
import equinox as eqx
import equinox.nn as nn
import distrax


class FixedDistrax(eqx.Module):
    cls: Callable = eqx.field(static=True)
    args: PyTree[Any]
    kwargs: PyTree[Any]

    def __init__(self, cls: Callable, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def _resolve(self, x):
        return jax.tree_util.tree_map(
            lambda leaf: leaf.dist if isinstance(leaf, FixedDistrax) else leaf,
            x,
            is_leaf=lambda l: isinstance(l, FixedDistrax)
        )

    @property
    def dist(self):
        resolved_args = self._resolve(self.args)
        resolved_kwargs = self._resolve(self.kwargs)
        return self.cls(*resolved_args, **resolved_kwargs)

    def __getattr__(self, name):
        return getattr(self.dist, name)


class FixedFactory(eqx.Module):
    cls: Callable = eqx.field(static=True)

    def __call__(self, *args, **kwargs):
        return FixedDistrax(self.cls, *args, **kwargs)


class ProxyDistrax:
    def __call__(self, module):
        """
        Wrap a custom distrax-compatible callable (function or class).
        """
        if not callable(module):
            raise TypeError("ProxyDistrax can only wrap callables")

        # return lambda *args, **kwargs: FixedDistrax(module, *args, **kwargs)
        return FixedFactory(module)

    def __getattr__(self, name):
        """
        Wrap a distrax method directly.
        """
        attr = getattr(distrax, name)

        if callable(attr):
            # return lambda *args, **kwargs: FixedDistrax(attr, *args, **kwargs)
            return FixedFactory(attr)
        return attr


dx = ProxyDistrax()


class Actor(eqx.Module):
    layers: nn.Sequential

    def __init__(self, obs_dim, action_dim, hidden_dim, *, key):
        keys = jax.random.split(key, num=4)
        self.layers = nn.Sequential([
            nn.Linear(obs_dim, hidden_dim, key=keys[0]),
            nn.Lambda(jax.nn.relu),
            nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
            nn.Lambda(jax.nn.relu),
            nn.Linear(hidden_dim, hidden_dim, key=keys[2]),
            nn.Lambda(jax.nn.relu),
            nn.Linear(hidden_dim, action_dim * 2, key=keys[3])
        ])

    def __call__(self, obs):
        mu, log_sigma = jnp.split(self.layers(obs), 2, axis=-1)
        log_sigma = jnp.clip(log_sigma, -5, 2)

        mean = mu
        std = jnp.exp(log_sigma)
        loc = jnp.ones_like(mean)
        scale = jnp.ones_like(mean)

        # base = FixedDistrax(distrax.Normal, mean, std)
        # transform = [FixedDistrax(distrax.ScalarAffine, shift=loc, scale=scale), distrax.Tanh()]
        # bijector = FixedDistrax(distrax.Chain, transform)
        # base = FixedDistrax(distrax.Transformed, base, bijector)
        # dist = FixedDistrax(distrax.Independent, base)
        # return dist

        dist = dx.Normal(mean, std)
        transform = [dx.ScalarAffine(shift=loc, scale=scale), dx.Tanh()]
        bijector = dx.Chain(transform)
        dist = dx.Transformed(dist, bijector)
        dist = dx.Independent(dist)
        return dist


# evaluate on batched data
states = jnp.zeros((32, 5))
model = Actor(5, 3, 32, key=jax.random.PRNGKey(0))

actor_eqx = eqx.filter_jit(eqx.filter_vmap(model))(states)
actor_jax = jax.jit(jax.vmap(model))(states)

print(" ... Equinox ... ")
sample_eqx = actor_eqx.sample(seed=jax.random.PRNGKey(0), sample_shape=(10, ))
print(f"sample shape {sample_eqx.shape}\n min   {sample_eqx.min()} \n max   {sample_eqx.max()}")
print(f"log prob shape {actor_eqx.log_prob(sample_eqx).shape}")

print("\n ... Jax ... ")
sample_jax = actor_jax.sample(seed=jax.random.PRNGKey(0), sample_shape=(10, ))
print(f"sample shape {sample_jax.shape}\n min   {sample_jax.min()} \n max   {sample_jax.max()}")
print(f"log prob shape {actor_jax.log_prob(sample_jax).shape}")
