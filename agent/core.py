import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

import equinox as eqx
import optax

from .models import World, Value, Actor
from .memory import Memory


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


class Learner(eqx.Module):
    model: eqx.Module
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    optimizer_state: optax.OptState

    @classmethod
    def create(cls, model_cls, config, *, key):
        model = model_cls(**config, key=key)
        optimizer = optax.adam(config.lr)
        params = eqx.filter(model, eqx.is_array)
        optimizer_state = optimizer.init(params)
        return cls(model, optimizer, optimizer_state)

    def update(self, grads):
        updates, new_optimizer_state = self.optimizer.update(
            grads, self.optimizer_state, self.model
        )
        new_model = optax.apply_updates(self.model, updates)
        return eqx.tree_at(
            lambda x: (x.model, x.optimizer_state),
            self,
            (new_model, new_optimizer_state)
        )


class Agent(eqx.Module):
    world: Learner[World]
    actor: Learner[Actor]
    value: Learner[Value]
    memory: Memory

    def __init__(
            self,
            config,
            env_params,
            *,
            key: PRNGKeyArray,
    ):
        key_world, key_actor, key_value = jax.random.split(key, 3)
        self.world = Learner.create(World, config.world, key=key_world)
        self.actor = Learner.create(Actor, config.actor, key=key_actor)
        self.value = Learner.create(Value, config.value, key=key_value)
        self.memory = Memory(**config.memory)

    def act():
        pass

    def predict(self, latent_state, action, key):
        """Transition .
        """
        prior = Transition(latent_state, action, key)
        return prior

    def perceive(self, latent_state, action, observation, key):
        """
        Perception is the process of recognizing existing knowledge or deriving new information from sensory inputs based on memory, representations, and predictive models.
        """
        key_prior, key_posterior = jax.random.split(key, 2)
        prior = predict(latent_state, action, key_prior)
        embedding = Encoder(observation)
        posterior = Representation(prior.latent_state, embedding, key_posterior)
        return posterior

    def reason(self, data, key):
        """
        Reason about the relationship among data and to the goal with contexts from predictive models or memory given a fixed belief;
        Reasoning is on-demand learning, which creates new knowledge and will be offloaded to the offline learning stage, e.g. dreaming
        """
        key_init, key_scan = jax.random.split(key, 2)
        init_latent_state = init_state(key_init)

        def step_fn(carry, inputs):
            latent_state, key = carry
            action, obs = inputs

            key, subkey = jax.random.split(key)
            posterior = self.perceive(latent_state, action, obs, subkey)

            return (posterior.latent_state, key), posterior

        _, posteriors = jax.lax.scan(
            step_fn,
            (init_latent_state, key_scan),
            (data.action, data.observation)
        )
        return posteriors           # posterior with params for KL div loss

    def plan(self, posterior: LatentStateWithParams):
        """
        imagine()
        return()
        """
        return lambda_return

    def learn(self, key):
        """
        key, key_memory, key_reasoning, key_planning = jax.random.split(key, 4)
        data ← memory(key_memory)
        posterior ← reason(data, key_reasoning): what is it? what might be missing? will it be?
        update(data, posterior):
          model
            reward_loss(posterior.latent_state, reward)
            observation_loss(posterior.latent_state, observation)
            kl_loss(posterior.params)
          lambda_return ← plan(posterior, key_planning)
            actor_loss
            value_loss
        """

        key, key_memory, key_reasoning, key_planning = jax.random.split(key, 4)
        data = self.memory.sample(self.batch_size, key_memory)
        posterior = self.reason(data, key_reasoning)

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

        lambda_return = self.plan(posterior, key_planning)
        actor_loss = -lambda_return.mean()
        value_loss = -self.value_model(posterior.latent_state).log_prob(lambda_return).mean() # TODO: add PG

        loss_terms = {
            "model/reward": reward_loss,
            "model/observation": observation_loss,
            "model/kl": kl_loss,
            "actor": actor_loss,
            "value": value_loss,
        }
        return loss_terms, key
