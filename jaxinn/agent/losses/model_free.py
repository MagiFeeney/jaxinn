from .base import Loss, ActorLoss, CriticLoss


class PPOLossMixIn(Loss, ActorLoss, CriticLoss):
    @eqx.filter_value_and_grad(has_aux=True)
    @differentiable(['actor'])
    def actor_loss_fn(
            self,
            obs: jax.Array,
            actions: jax.Array,
            old_log_probs: jax.Array,
            advantages: jax.Array,
            key: PRNGKeyArray,
    ) -> Tuple[jax.Array, Tuple[Dict[str, jax.Array]]]:
        actor_params = jax.vmap(self.actor)(obs)
        actor_dists = self.actor.get_dist(actor_params)
        log_probs = actor_dists.log_prob(actions)

        ratio = jnp.exp(log_probs - old_log_probs)
        surrogate = ratio * advantages
        clip_ratio = jnp.clip(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_clipped = clip_ratio * advantages

        actor_loss = -jnp.minimum(surrogate, surrogate_clipped).mean()

        metrics = {
            "ac/actor": actor_loss,
        }
        return actor_loss, metrics

    @eqx.filter_value_and_grad(has_aux=True)
    @differentiable(['critic'])
    def critic_loss_fn(
            self,
            obs: jax.Array,
            returns: jax.Array,
            old_values: jax.Array,
            key: PRNGKeyArray,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        values = jax.vmap(self.critic)(obs).mean()

        if self.use_clipped_critic_loss:
            value_clipped = old_values + (values - old_values).clip(-self.clip_param, self.clip_param)
            critic_losses = (values - returns).pow(2)
            critic_losses_clipped = (value_clipped - returns).pow(2)
            critic_loss = jnp.maximum(critic_losses, critic_losses_clipped).mean()
        else:
            critic_loss = (returns - values).pow(2).mean()

        metrics = {
            "ac/critic": critic_loss,
        }
        return critic_loss, metrics


class SACLossMixIn(Loss, ActorLoss, CriticLoss):
    pass
