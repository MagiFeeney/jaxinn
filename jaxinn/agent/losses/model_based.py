import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx

from jaxinn.common.structs import Transition, LatentStateWithDist

from .base import Loss, ActorLoss, CriticLoss, WorldLoss
from .utils import differentiable


class DreamerLossMixIn(Loss, WorldLoss, ActorLoss, CriticLoss):
    def _compute_kl_loss(self, prior: LatentStateWithDist, posterior: LatentStateWithDist) -> jax.Array:
        if self.kl_balance > 0:
            kl_loss_post = posterior.kl_divergence(jax.lax.stop_gradient(prior).dist)
            kl_loss_prior = jax.lax.stop_gradient(posterior).kl_divergence(prior.dist)

            if self.kl_average:
                kl_loss_post = kl_loss_post.mean()
                kl_loss_prior = kl_loss_prior.mean()

            if self.free_nats > 0:
                kl_loss_post = jnp.clip(kl_loss_post, min=self.free_nats)
                kl_loss_prior = jnp.clip(kl_loss_prior, min=self.free_nats)

            kl_loss = self.kl_balance * kl_loss_prior + (1 - self.kl_balance) * kl_loss_post
        else:
            kl_loss = posterior.kl_divergence(prior.dist)

            if self.kl_average:
                kl_loss = kl_loss.mean()

            if self.free_nats > 0:
                kl_loss = jnp.clip(kl_loss, min=self.free_nats)

        return kl_loss if self.kl_average else kl_loss.mean()

    @eqx.filter_value_and_grad(has_aux=True)
    @differentiable(['world'])
    def world_loss_fn(
            self,
            data: Transition,
            key: PRNGKeyArray,
    ) -> tuple[jax.Array, tuple[dict[str, jax.Array], LatentStateWithDist]]:
        prior, posterior = self.reason(data, key)

        reward_dist = jax.vmap(jax.vmap(self.world.reward))(posterior.latent_state)
        reward_log_prob = reward_dist.log_prob(data.reward)
        reward_loss = -reward_log_prob.mean()

        observation_dist = jax.vmap(jax.vmap(self.world.decoder))(posterior.latent_state)
        observation_log_prob = observation_dist.log_prob(data.next_obs)
        observation_loss = -observation_log_prob.mean()

        kl_loss = self._compute_kl_loss(prior, posterior)

        total_loss = reward_loss + observation_loss + kl_loss

        # For logging
        reward_mse = jnp.mean((reward_dist.mean() - data.reward)**2)
        observation_mse = jnp.mean((observation_dist.mean() - data.next_obs)**2)

        metrics = {
            "model/reward": reward_loss,
            "model/observation": observation_loss,
            "model/kl": kl_loss,
            "model/total": total_loss,
            "model/reward_mse": reward_mse,
            "model/observation_mse": observation_mse
        }
        return total_loss, (metrics, posterior)

    @eqx.filter_value_and_grad(has_aux=True)
    @differentiable(['actor'])
    def actor_loss_fn(
            self,
            posterior: LatentStateWithDist,
            key: PRNGKeyArray,
    ) -> tuple[jax.Array, tuple[dict[str, jax.Array], jax.Array, jax.Array]]:
        imagined_latent_states, actions = self.plan(posterior.latent_state.flatten(), key)
        (advantages, return_predictions), aux = self.process(imagined_latent_states)

        actor_loss = -return_predictions.mean()
        metrics = {
            "ac/actor": actor_loss,
            **aux
        }
        return actor_loss, (metrics, (imagined_latent_states, return_predictions))

    @eqx.filter_value_and_grad(has_aux=True)
    @differentiable(['critic'])
    def critic_loss_fn(
            self,
            imagined_latent_states: jax.Array,
            return_prediction: jax.Array,
            key: PRNGKeyArray,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        value_dist = jax.vmap(jax.vmap(self.critic))(imagined_latent_states[:-1].detach())
        critic_loss = -value_dist.log_prob(jax.lax.stop_gradient(return_prediction))
        critic_loss = critic_loss.mean()

        metrics = {
            "ac/critic": critic_loss,
        }
        return critic_loss, metrics


class MixedActorGradientLoss(DreamerLossMixIn):
    @eqx.filter_value_and_grad(has_aux=True)
    @differentiable(['actor'])
    def actor_loss_fn(
            self,
            posterior: LatentStateWithDist,
            key: PRNGKeyArray,
    ) -> tuple[jax.Array, tuple[dict[str, jax.Array], jax.Array, jax.Array]]:
        imagined_latent_states, actions = self.plan(posterior.latent_state.flatten(), key)
        (advantages, return_predictions, action_log_probs, entropies, weights), aux = self.process(imagined_latent_states, actions)

        bptt_loss = -return_predictions
        likelihood_pg_loss = -action_log_probs * jax.lax.stop_gradient(advantages)
        entropy_loss = -self.entropy_coef * entropies

        actor_loss = self.pg_mix * bptt_loss + (1 - self.pg_mix) * likelihood_pg_loss + entropy_loss
        actor_loss = (weights * actor_loss).mean()

        metrics = {
            "ac/actor": actor_loss,
            "ac/bptt_loss": bptt_loss.mean(),
            "ac/likelihood_pg_loss": likelihood_pg_loss.mean(),
            "ac/entropy_loss": entropy_loss.mean(),
            **aux
        }
        return actor_loss, (metrics, (imagined_latent_states, return_predictions))
