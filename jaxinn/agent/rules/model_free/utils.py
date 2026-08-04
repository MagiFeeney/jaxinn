import jax
import equinox as eqx

from jaxinn.structs import Transition


def reconstruct_rl_tuple(transition: Transition, boundary_obs: jax.Array | None = None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    terminated_or_boundary = transition.terminated[1:]
    if boundary_obs is None:    # Next-step autoreset introduces a dummy transition from terminated/truncated obs to the reset obs
        last_done = transition.terminated[:-1] | transition.truncated[:-1]
        terminated_or_boundary |= last_done
    return (
        transition.next_obs[:-1],
        transition.action[1:],
        transition.reward[1:],
        transition.next_obs[1:] if boundary_obs is None else boundary_obs[1:],
        terminated_or_boundary
    )


def soft_update(target_net: eqx.Module, source_net: eqx.Module, tau: float) -> eqx.Module:
    """EMA update of the target network with source network."""

    def update_leaf(t, s):
        if eqx.is_inexact_array(t):
            return t * (1.0 - tau) + s * tau
        return t

    return jax.tree.map(update_leaf, target_net, source_net)
