from typing import Tuple, Optional

import jax
import equinox as eqx

from jaxinn.structs import Transition


def reconstruct_rl_tuple(transition: Transition, terminal_obs: Optional[jax.Array] = None) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return (
        transition.next_obs[:-1],
        transition.action[1:],
        transition.reward[1:],
        transition.next_obs[1:],
        transition.terminated[1:],
        transition.truncated[1:],
    ), terminal_obs[1:] if terminal_obs is not None else transition.next_obs[1:]


def soft_update(target_net: eqx.Module, source_net: eqx.Module, tau: float) -> eqx.Module:
    """EMA update of the target network with source network."""

    def update_leaf(t, s):
        if eqx.is_inexact_array(t):
            return t * (1.0 - tau) + s * tau
        return t

    return jax.tree.map(update_leaf, target_net, source_net)
