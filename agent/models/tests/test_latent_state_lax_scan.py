import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Any, Callable

from typing import Optional, Callable, Union, Dict
from jaxtyping import Array, Float, PRNGKeyArray


class LatentState(eqx.Module):
    """
    Combine deterministic history encoding (belief) and the stochastic predictor (state) into a single state.
    """
    belief: jax.Array  # h_t
    state: jax.Array   # s_t

    @property
    def batch_shape(self) -> tuple:
        return self.belief.shape[:-1]

    @property
    def feature(self) -> jax.Array:
        return jnp.concatenate([self.belief, self.state], axis=-1)

    def __getitem__(self, index: Any) -> "LatentState":
        return jax.tree.map(lambda x: x[index], self)

    def flatten(self) -> "LatentState":
        return jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), self)

    def narrow(self, axis: int, start: int, length: int) -> "LatentState": # TODO: delete if not used
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


# --- Mock Distribution for testing ---
class MockDist:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
    def mean(self):
        return self.loc

# --- Testing the Scan Behavior ---

def test_scan_auto_stacking():
    # 1. Setup initial dimensions
    T = 5          # Time steps
    H_dim = 10     # Belief dim
    S_dim = 4      # State dim

    # 2. Create initial state
    init_latent = LatentState(
        belief=jnp.zeros((H_dim,)),
        state=jnp.zeros((S_dim,))
    )
    init_swp = LatentStateWithParams(
        latent_state=init_latent,
        params={"loc": jnp.zeros((3,)), "scale": jnp.ones((3,))},
        dist_cls=MockDist
    )

    params={"loc": jnp.zeros((3,)), "scale": jnp.ones((3,))}
    dist_cls = MockDist

    # Mock inputs for the scan (e.g., actions or observations)
    inputs = jnp.ones((T, H_dim))

    # 3. Define the transition function
    def core_step_init_with_params(carry, x):
        # Update logic: increment belief by input, keep state as is
        latent_state, params = carry

        new_latent = LatentState(
            belief=latent_state.belief + x,
            state=latent_state.state
        )
        # Update params: slightly shift the 'loc' parameter
        new_params = {
            "loc": params["loc"] + 0.1,
            "scale": params["scale"]
        }

        out = LatentStateWithParams(
            latent_state=new_latent,
            params=new_params,
            dist_cls=dist_cls
        )
        return (new_latent, new_params), out

    # 3. Define the transition function
    def core_step_init_no_params(carry, x):
        # Update logic: increment belief by input, keep state as is
        new_latent = LatentState(
            belief=carry.belief + x,
            state=carry.state
        )
        # Update params: slightly shift the 'loc' parameter
        new_params = {
            "loc": params["loc"] + 0.1,
            "scale": params["scale"]
        }

        out = LatentStateWithParams(
            latent_state=new_latent,
            params=new_params,
            dist_cls=dist_cls
        )
        return new_latent, out

    # 3. Define the transition function
    def core_step_io_aligned(carry, x):
        # Update logic: increment belief by input, keep state as is
        new_latent = LatentState(
            belief=carry.latent_state.belief + x,
            state=carry.latent_state.state
        )
        # Update params: slightly shift the 'loc' parameter
        new_params = {
            "loc": carry.params["loc"] + 0.1,
            "scale": carry.params["scale"]
        }

        out = LatentStateWithParams(
            latent_state=new_latent,
            params=new_params,
            dist_cls=carry.dist_cls
        )
        return out, out

    # 4. Run the scan
    final_carry, history = jax.lax.scan(core_step_io_aligned, init_swp, inputs)
    # final_carry, history = jax.lax.scan(core_step_init_no_params, init_latent, inputs)
    # final_carry, history = jax.lax.scan(core_step_init_with_params, (init_latent, params), inputs)

    # --- VERIFICATIONS ---

    # A. Check LatentState stacking
    assert history.latent_state.belief.shape == (T, H_dim)
    assert history.latent_state.state.shape == (T, S_dim)

    # B. Check Params dictionary stacking (Nested leaves)
    assert history.params["loc"].shape == (T, 3)
    assert history.params["scale"].shape == (T, 3)

    # C. Check that static field remained static
    assert history.dist_cls == MockDist

    # D. Check functionality (the __getattr__ proxy)
    # Even though 'history' is stacked, it behaves like a batched distribution
    means = history.mean()
    assert means.shape == (T, 3)

    print("Scan test passed: All leaves automatically stacked!")

if __name__ == "__main__":
    test_scan_auto_stacking()
