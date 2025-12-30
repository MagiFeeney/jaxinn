import jax
import jax.random as jr
import equinox as eqx
import jax.numpy as jnp


class Transition(eqx.Module):
    action: jax.Array
    next_obs: jax.Array
    reward: jax.Array
    done: jax.Array

    @classmethod
    def init_empty(cls, obs_shape, action_dim):
        return cls(
            action=jnp.zeros((action_dim,)),
            next_obs=jnp.zeros(obs_shape),
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
        )


class Agent(eqx.Module):
    world_model: eqx.Module     # TODO: world model module
    policy: eqx.Module
    value: eqx.Module

    # TODO: init the world model with key
    def act():
        pass

    def predict(self, latent_state, action, key):
        """Transition Model.
        """
        prior = TransitionModel(latent_state, action, key)
        return prior

    def perceive(self, latent_state, action, observation, key):
        """Representation Model.
        """
        key_prior, key_posterior = jax.random.split(key, 2)
        prior = predict(latent_state, action, key_prior)
        posterior = RepresentationModel(prior.latent_state, observation, key_posterior)
        return posterior

    def learn(key):
        """
        key, key_memory, key_reasoning, key_planning = jax.random.split(key, 4)
        data ← memory(key_memory)
        posterior ← reasoning(data, key_reasoning): what is it? what might be missing? will it be?
        update(data, posterior):
          model
            reward_loss(posterior.latent_state, reward)
            observation_loss(posterior.latent_state, observation)
            kl_loss(posterior.params)
          lambda_return ← planning(posterior, key_planning)
            policy_loss
            value_loss
        """

        key, key_memory, key_reasoning, key_planning = jax.random.split(key, 4)
        data = memory.sample(self.batch_size, key_memory)
        posterior = reasoning(data, key_reasoning)

        # TODO: update function modulize
        @eqx.filter_value_and_grad
        def model_loss():           # TODO: abstract the loss function for three different considerations
            # reward
            reward_loss = -self.reward_model(posterior.latent_state).log_prob(data.reward)

            # observation
            observation_loss = -self.observation_model(posterior.latent_state).log_prob(data.observation)

            # KL divergence
            kl_loss = ...

            return reward_loss + observation_loss + kl_loss

        def make_step():
            loss, grads = model_loss()
            updates, opt_state = optim.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return loss, model, opt_state

        lambda_return = planning(posterior, key_planning)
        policy_loss = -lambda_return.mean()
        value_loss = -self.value_model(posterior.latent_state).log_prob(lambda_return).mean() # TODO: add PG

        loss_terms = {
            "model/reward": reward_loss,
            "model/observation": observation_loss,
            "model/kl": kl_loss,
            "policy": policy_loss,
            "value": value_loss,
        }
        return loss_terms, key

    def reasoning(data, key):
        """
        Reason about the relationship among data and to the goal with contexts, from model itself or memory given a fixed belief;
        Reasoning is on-demand learning, which creates new knowledge and will be offloaded to the offline learning stage, e.g. dreaming

        reward
        observation
        transition
        """
        key_init, key_scan = jax.random.split(key, 2)
        init_latent_state = init_state(key_init)

        def step_fn(carry, inputs):
            latent_state, key = carry
            action, obs = inputs

            key, subkey = jax.random.split(key)
            posterior = perceive(latent_state, action, obs, subkey)

            return (posterior.latent_state, key), posterior

        _, posteriors = jax.lax.scan(
            step_fn,
            (init_latent_state, key_scan),
            (data.action, data.observation)
        )
        return posteriors           # posterior with params for KL div loss

    def planning(posterior: LatentStateWithParams):
        """
        imagine()
        return()
        """
        return lambda_return
