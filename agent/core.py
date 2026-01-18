import jax
import jax.numpy as jnp
from typing import Tuple, Generic, TypeVar
from jaxtyping import PRNGKeyArray

import equinox as eqx
import optax

from .models import World, Critic, Actor, LatentState, LatentStateWithParams
from .memory import Transition, Memory, Uniform, Prioritized


ModelType = TypeVar("ModelType", bound=eqx.Module)


class Learner(eqx.Module, Generic[ModelType]):
    model: ModelType
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    optimizer_state: optax.OptState

    @classmethod
    def create(cls, model_cls, config, *, key):
        model = model_cls(**config(), key=key)
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.optimizer.max_norm),
            optax.adam(config.optimizer.lr, eps=config.optimizer.eps)
        )
        params = eqx.filter(model, eqx.is_array)
        optimizer_state = optimizer.init(params)
        return cls(model, optimizer, optimizer_state)

    def update(self, grads):
        updates, new_optimizer_state = self.optimizer.update(
            grads, self.optimizer_state, self.model
        )
        new_model = eqx.apply_updates(self.model, updates)
        return eqx.tree_at(
            lambda x: (x.model, x.optimizer_state),
            self,
            (new_model, new_optimizer_state)
        )

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class Agent(eqx.Module):
    world: Learner[World]
    actor: Learner[Actor]
    critic: Learner[Critic]
    memory: Memory
    planning_horizon: int = eqx.field(static=True)
    discount_factor: float = eqx.field(static=True)
    uae_lambda: float = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    random_init: bool = eqx.field(static=True)
    belief_size: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)

    def __init__(
            self,
            config,
            *,
            key: PRNGKeyArray,
    ):
        key_world, key_actor, key_critic = jax.random.split(key, 3)
        self.world = Learner.create(World, config.world, key=key_world)
        self.actor = Learner.create(Actor, config.actor, key=key_actor)
        self.critic = Learner.create(Critic, config.critic, key=key_critic)
        if config.memory.type.lower() == "uniform":
            memory_cls = Uniform
        else:
            memory_cls = Prioritized
        self.memory = memory_cls(
            capacity=config.memory.capacity,
            obs_shape=config.world.perception.encoder.shape,
            action_size=config.world.transition.action_size
        )
        # For initialization of LatentState
        self.random_init = config.random_init
        self.belief_size = config.world.transition.belief_size
        self.state_size = config.world.transition.state_size

        # Extra particulars for agent learning
        self.__dict__.update(config.optimization())

    def act(self, last_latent_state, last_action, obs, *, key: PRNGKeyArray, eval=False):
        key_perceive, key_action = jax.random.split(key, 2)
        _, posterior = self.perceive(last_latent_state, last_action, obs, key_perceive)
        params = jax.vmap(self.actor)(posterior.latent_state)
        action = self.actor.sample(params, key_action, eval)
        return posterior.latent_state, action

    def predict(self, latent_state, action, key):
        """Predict based on the belief without seeing observation."""
        params, belief = jax.vmap(self.world.transition)(latent_state, action)
        state = self.world.transition.sample(params, key)
        prior = LatentStateWithParams(
            latent_state=LatentState(belief=belief, state=state),
            params=params,
            dist_cls=self.world.transition.dist_cls
        )
        return prior

    def init_state(self, key: PRNGKeyArray, batch_shape: Tuple[int, ...] = ()):
        return LatentState.initialize(self.belief_size, self.state_size, self.random_init, batch_shape, key=key)

    def perceive(self, latent_state, action, observation, key):
        """
        Perception is the process of recognizing existing knowledge or deriving new information from sensory inputs based on memory, representations, and predictive models.
        """
        key_prior, key_posterior = jax.random.split(key, 2)
        prior = self.predict(latent_state, action, key_prior)
        embedding = jax.vmap(self.world.perception.encoder)(observation)
        params, belief = jax.vmap(self.world.representation)(prior.latent_state.belief, embedding)
        state = self.world.representation.sample(params, key_posterior)
        posterior = LatentStateWithParams(
            latent_state=LatentState(belief=belief, state=state),
            params=params,
            dist_cls=self.world.representation.dist_cls
        )
        return prior, posterior

    def add_experience(self, transitions: Transition):
        new_memory = self.memory.add(transitions)

        return eqx.tree_at(
            lambda x: x.memory,
            self,
            new_memory
        )

    def reason(self, data, key):
        """
        Reason about the relationship among data and to the goal with contexts from predictive models or memory given a fixed belief;
        Reasoning is on-demand learning, which creates new knowledge and will be offloaded to the offline learning stage, e.g. dreaming
        """
        key_init, key_scan = jax.random.split(key, 2)
        init_latent_state = self.init_state(key_init, batch_shape=(data.action.shape[1],))

        def reason_step_fn(carry, inputs):
            latent_state, key = carry
            action, obs, done = inputs
            mask = 1 - done
            key, subkey = jax.random.split(key)

            # Mask the action and state if the observation results from reset
            # This happens when the sampled sequence contains multiple trajectories
            latent_state = jax.tree.map(lambda x: x * mask, latent_state)
            action = action * mask
            prior, posterior = self.perceive(latent_state, action, obs, subkey)

            return (posterior.latent_state, key), (prior, posterior)

        _, (priors, posteriors) = jax.lax.scan(
            reason_step_fn,
            (init_latent_state, key_scan),
            (data.action, data.next_obs, data.done)
        )
        return priors, posteriors

    def plan(self, posterior: LatentStateWithParams, key):
        key_scan = key

        # Imagination
        def imagine_step_fn(carry, _): # TODO: integrate the logic of masking inputs when terminated; require a terminal predictor d(s, a)
            latent_state, key = carry

            key, key_action, key_predict = jax.random.split(key, 3)
            params = jax.vmap(jax.vmap(self.actor))(latent_state)
            action = self.actor.sample(params, key_action)
            prior = self.predict(latent_state, action, key_predict)

            return (prior.latent_state, key), (prior.latent_state, action)

        _, (latent_states, actions) = jax.lax.scan(
            imagine_step_fn,
            (posterior.latent_state, key_scan),
            None,
            self.planning_horizon,
        )

        # Processing data
        rewards = self.world.reward(latent_states).mean() # Equivalent to r(s, a, s') instead of r(s, a)
        next_values = self.critic(latent_states).mean()
        first_value = self.critic(posterior.latent_states).mean()

        values = jnp.concatenate([first_value[None, ...], next_values[:-1]], axis=-1)
        last_value = next_values[-1]

        dones = jnp.ones_like(values)
        baselines = jnp.zeros_like(values)

        def uae_step_fn(carry, inputs):
            """
            Unified advantage estimator (UAE): a generalized version of GAE.

            When baseline function is zero, it reduces to λ-return.

            Reference: https://arxiv.org/pdf/2302.00533
            """
            uae, next_value = carry
            reward, value, baseline, done = inputs

            delta = (
                reward
                + self.discount_factor * next_value * (1 - done)
                - baseline
            )
            z = value - baseline
            discounted_uae = self.discount_factor * self.uae_lambda * (1 - done) * uae
            return_prediction = delta + discounted_uae + baseline
            uae = (delta - z) + discounted_uae

            return (uae, value), return_prediction

        _, return_predictions = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_value), last_value),
            (rewards, values, baselines, dones),
            reverse=True,
        )

        return return_predictions

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
            critic_loss
        """

        key, key_memory, key_reasoning, key_planning = jax.random.split(key, 4)
        data = self.memory.sample((self.batch_size, self.chunk_size), key_memory)
        prior, posterior = self.reason(data, key_reasoning)
        metrics = {}

        @eqx.filter_critic_and_grad
        def world_loss_fn(
                world: Learner,
                prior: LatentStateWithParams,
                posterior: LatentStateWithParams,
                data
        ):
            # reward
            reward_loss = -world.reward(posterior.latent_state).log_prob(data.reward).mean()

            # observation
            observation_loss = -world.observation(posterior.latent_state).log_prob(data.next_obs).mean()

            # KL divergence
            kl_loss = posterior.kl_divergence(prior.dist).mean()

            # Total loss
            total_loss = reward_loss + observation_loss + kl_loss

            return total_loss, {
                "model/reward": reward_loss,
                "model/observation": observation_loss,
                "model/kl": kl_loss,
                "model/total": total_loss,
            }

        (loss, aux), grads = world_loss_fn(self.world, prior, posterior, data)
        new_world = self.world.update(grads)
        metrics.update(**aux)

        @eqx.filter_critic_and_grad
        def ac_loss_fn(
                actor: Learner,
                critic: Learner,
                posterior: LatentStateWithParams,
        ):  # TODO: add PG
            imagination = self.plan(posterior, key_planning)
            return_prediction = self.processor(imagination)
            actor_loss = -return_prediction.mean()
            critic_loss = -self.critic(posterior.latent_state).log_prob(jax.lax.stop_gradient(return_prediction)).mean()
            total_loss = actor_loss + critic_loss # non-interleaved

            return total_loss, {
                "actor": actor_loss,
                "critic": critic_loss,
            }

        (loss, aux), (actor_grads, critic_grads) = ac_loss_fn(self.actor, self.critic, posterior)
        new_actor = self.actor.update(actor_grads)
        new_critic = self.critic.update(critic_grads)
        metrics.update(**aux)

        return eqx.tree_at(
            lambda x: (x.world, x.actor, x.critic),
            self,
            (new_world, new_actor, new_critic)
        ), metrics
