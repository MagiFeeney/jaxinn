from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx
from .utils import differentiable

from .base import Loss, ActorLoss, CriticLoss, ActorCriticLoss


class PPOLossMixIn(Loss, ActorCriticLoss):
    @eqx.filter_value_and_grad(has_aux=True)
    @differentiable(['actor_critic'])
    def actor_critic_loss_fn(
            self,
            mini_batch: Tuple[jax.Array, ...],
            key: PRNGKeyArray,
    ) -> Tuple[jax.Array, Tuple[Dict[str, jax.Array]]]:
        obs, actions, advantages, returns, old_values, old_log_probs = mini_batch

        # Actor loss
        actor_dists = jax.vmap(self.actor_critic.get_actor_dist)(obs)
        log_probs = actor_dists.log_prob(actions)

        ratio = jnp.exp(log_probs - old_log_probs)
        surrogate = ratio * advantages
        clip_ratio = jnp.clip(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_clipped = clip_ratio * advantages

        actor_loss = -jnp.minimum(surrogate, surrogate_clipped).mean()

        # Critic loss
        values = jax.vmap(self.actor_critic.get_critic_dist)(obs).mean()

        if self.use_clipped_critic_loss:
            value_clipped = old_values + (values - old_values).clip(-self.clip_param, self.clip_param)
            critic_losses = (values - returns) ** 2
            critic_losses_clipped = (value_clipped - returns) ** 2
            critic_loss = jnp.maximum(critic_losses, critic_losses_clipped).mean()
        else:
            critic_loss = ((returns - values) ** 2).mean()

        metrics = {
            "ac/actor": actor_loss,
            "ac/critic": critic_loss,
        }
        total_loss = actor_loss + critic_loss
        return total_loss, metrics


class SACLossMixIn(Loss, ActorLoss, CriticLoss):
    pass
