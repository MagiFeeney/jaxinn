import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict

import distrax


def test_latent_state_pytree():
    state = LatentState(belief=jnp.ones((5, 10)), state=jnp.zeros((5, 4)))

    # Test flattening/unflattening
    leaves, treedef = jax.tree_util.tree_flatten(state)
    assert len(leaves) == 2  # belief and state

    new_state = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.all(new_state.belief == state.belief)

def test_latent_state_with_params_static_field():
    # Mock a distribution class
    class MockDist:
        def __init__(self, loc): self.loc = loc

    state = LatentState(belief=jnp.zeros((10,)), state=jnp.zeros((4,)))
    swp = LatentStateWithParams(
        latent_state=state,
        params={"loc": jnp.ones((3,))},
        dist_cls=MockDist
    )

    # Check that dist_cls is treated as static (not a leaf)
    leaves = jax.tree_util.tree_leaves(swp)
    # Leaves should be: belief, state, and the 'loc' array inside params
    assert len(leaves) == 3


def test_latent_state_manipulations():
    # Setup two states with batch size 5, feature size 10/4
    s1 = LatentState(belief=jnp.ones((5, 10)), state=jnp.ones((5, 4)))
    s2 = LatentState(belief=jnp.zeros((5, 10)), state=jnp.zeros((5, 4)))

    # Test feature concatenation: (10 + 4) = 14
    assert s1.feature.shape == (5, 14)

    # Test stacking: Result should be (2, 5, 10) and (2, 5, 4)
    stacked = LatentState.stack([s1, s2], axis=0)
    assert stacked.belief.shape == (2, 5, 10)
    assert stacked.batch_shape == (2, 5)

    # Test indexing (__getitem__)
    indexed = s1[0]
    assert indexed.belief.shape == (10,)

    # Test flattening: (5, 10) -> (5, 10)
    flat = s1.flatten()
    assert flat.belief.ndim == 2 # If input was 2D, check logic matches intent


def test_distribution_proxy():
    import distrax

    ls = LatentState(belief=jnp.zeros((2,)), state=jnp.zeros((2,)))
    swp = LatentStateWithParams(
        latent_state=ls,
        params={"loc": jnp.zeros((5,)), "scale": jnp.ones((5,))},
        dist_cls=distrax.Normal
    )

    # Test __getattr__ proxy to distrax.Normal
    assert swp.mean().shape == (5,)

    # Test sampling
    key = jax.random.PRNGKey(0)
    samples = swp.sample(seed=key, sample_shape=(10,))
    assert samples.shape == (10, 5)


def test_jit_and_vmap():
    state = LatentState(belief=jnp.ones((5, 10)), state=jnp.ones((5, 4)))

    @eqx.filter_jit
    def get_feat(s):
        return s.feature

    # Check JIT
    out = get_feat(state)
    assert out.shape == (5, 14)

    # Check Vmap (mapping over the batch dimension)
    v_feat = jax.vmap(lambda s: s.feature)(state)
    assert v_feat.shape == (5, 14)


@pytest.mark.parametrize("belief_shape, state_shape", [
    ((10,), (4,)),       # Single state
    ((5, 10), (5, 4)),   # Batched state
    ((2, 3, 10), (2, 3, 4)) # Multi-dim batch
])


def test_feature_concatenation(belief_shape, state_shape):
    belief = jnp.zeros(belief_shape)
    state = jnp.zeros(state_shape)
    ls = LatentState(belief=belief, state=state)

    expected_dim = belief_shape[-1] + state_shape[-1]
    assert ls.feature.shape[-1] == expected_dim
