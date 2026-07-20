from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx

from .base import Loss, ActorLoss, CriticLoss, ActorCriticLoss
from .utils import differentiable


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

        ratio = jnp.exp(log_probs - old_log_probs)[..., None]
        if self.normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
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

        entropy_loss = -self.entropy_coef * actor_dists.entropy().mean()

        metrics = {
            "ac/actor": actor_loss,
            "ac/critic": critic_loss,
            "ac/entropy": entropy_loss,
        }
        total_loss = actor_loss + critic_loss + entropy_loss
        return total_loss, metrics


class SACLossMixIn(Loss, ActorLoss, CriticLoss):
    @eqx.filter_value_and_grad(has_aux=True)
    @differentiable(['actor'])
    def actor_loss_fn(
            self,
            obs: jax.Array,
            key: PRNGKeyArray,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        actor_dists = jax.vmap(self.actor)(obs)
        actions, log_probs = actor_dists.sample_and_log_prob(seed=key)
        log_probs = log_probs[..., None]
        q_dists = jax.vmap(self.critic)(obs, actions)
        qs = q_dists.mean()
        min_q = jnp.min(qs, axis=1)

        actor_loss = ((self.alpha * log_probs) - min_q).mean()

        metrics = {
            "ac/actor": actor_loss,
        }
        return actor_loss, metrics

    @eqx.filter_value_and_grad(has_aux=True)
    @differentiable(['critic'])
    def critic_loss_fn(
            self,
            obs: jax.Array,
            actions: jax.Array,
            rewards: jax.Array,
            next_obs: jax.Array,
            terminated: jax.Array,
            key: PRNGKeyArray,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        actor_dists = jax.vmap(self.actor)(next_obs)
        next_actions, next_log_probs = actor_dists.sample_and_log_prob(seed=key)
        next_log_probs = next_log_probs[..., None]
        next_q_dists = jax.vmap(self.critic_target)(next_obs, next_actions)
        next_qs = next_q_dists.mean() # B x E x 1
        next_min_q = jnp.min(next_qs, axis=1) # B x 1
        bootstrapped_q = next_min_q - self.alpha * next_log_probs
        td_target = rewards + (1 - terminated) * self.discount_factor * bootstrapped_q
        td_target = jnp.expand_dims(td_target, axis=1)

        q_dists = jax.vmap(self.critic)(obs, actions)
        qs = q_dists.mean()
        critic_loss = ((qs - td_target)**2).mean(axis=0).sum()

        metrics = {
            "ac/critic": critic_loss,
        }
        return critic_loss, metrics
