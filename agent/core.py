import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Generic, TypeVar
from jaxtyping import PRNGKeyArray

import equinox as eqx
import distrax
import optax

from envs import Transition
from .models import World, Critic, Actor, LatentState, LatentStateWithParams
from .memory import Memory, Uniform, Prioritized
from .utils import differentiable


class Experience(eqx.Module):
    transition: Transition
    terminal_observation: jax.Array


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
        params = eqx.filter(model, eqx.is_inexact_array)
        optimizer_state = optimizer.init(params)
        return cls(model, optimizer, optimizer_state)

    def update(self, grads) -> "Learner":
        if isinstance(grads, Learner):
            grads = grads.model
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

    free_nats: float = eqx.field(static=True)
    kl_average: bool = eqx.field(static=True)
    kl_balance: float = eqx.field(static=True)

    def __init__(
            self,
            config,
            *,
            key: PRNGKeyArray,
            memory_id: jax.Array,
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
            seed_idx=memory_id,
            capacity=config.memory.capacity,
            obs_shape=config.world.perception.encoder.shape,
            action_size=config.world.transition.action_size,
            num_seeds=config.memory.num_seeds,
        )
        # For initialization of LatentState
        self.random_init = config.random_init
        self.belief_size = config.world.transition.belief_size
        self.state_size = config.world.transition.state_size

        # Extra particulars for agent learning
        self.__dict__.update(config.optimization())

    def init_state(self, key: PRNGKeyArray, batch_shape: Tuple[int, ...] = (), eval=False) -> LatentState:
        return LatentState.initialize(self.belief_size, self.state_size, False if eval else self.random_init, batch_shape, key=key)

    @staticmethod
    def transform(obs) -> jax.Array:
        if obs.dtype == jnp.uint8 and obs.ndim > 3:
            return obs.astype(jnp.float32) / 255.0 - 0.5
        return obs

    def act(self, last_latent_state: LatentState, last_action: jax.Array, obs: jax.Array, *, key: PRNGKeyArray, eval: bool = False) -> Tuple[LatentState, jax.Array]:
        key_perceive, key_action = jax.random.split(key, 2)
        obs = jax.vmap(self.world.perception.encoder)(self.transform(obs))
        _, posterior = self.perceive(last_latent_state, last_action, obs, key_perceive)
        params = jax.vmap(self.actor)(posterior.latent_state)
        action = self.actor.sample(params, key_action, eval)
        return posterior.latent_state, action

    def predict(self, latent_state: LatentState, action: jax.Array, key: PRNGKeyArray) -> LatentStateWithParams:
        """Predict based on the belief without seeing observation."""
        params, belief = jax.vmap(self.world.transition)(latent_state, action)
        state = self.world.transition.sample(params, key)
        prior = LatentStateWithParams(
            latent_state=LatentState(belief=belief, state=state),
            params=params,
            dist_cls=self.world.transition.dist_cls
        )
        return prior

    def perceive(self, latent_state: LatentState, action: jax.Array, observation: jax.Array, key: PRNGKeyArray) -> Tuple[LatentStateWithParams, LatentStateWithParams]:
        """
        Perception is the process of recognizing existing knowledge or deriving new information from sensory inputs based on memory, representations, and predictive models.
        """
        key_prior, key_posterior = jax.random.split(key, 2)
        prior = self.predict(latent_state, action, key_prior)
        params, belief = jax.vmap(self.world.representation)(prior.latent_state.belief, observation)
        state = self.world.representation.sample(params, key_posterior)
        posterior = LatentStateWithParams(
            latent_state=LatentState(belief=belief, state=state),
            params=params,
            dist_cls=self.world.representation.dist_cls
        )
        return prior, posterior

    @staticmethod
    def replenish_and_flatten(experiences: Experience, source: int) -> Tuple[Transition, jax.Array]:
        def flatten_fn(x):
            # (T, B, ...) -> (B*T, ...)
            flattened = jnp.moveaxis(x, source=source, destination=source - 1).reshape(-1, *x.shape[source + 1:])
            # For storage
            if x.dtype == jnp.float32 and x.ndim > 3:
                is_normalized = x.max() <= 1.0
                return jax.lax.cond(
                    is_normalized,
                    lambda arr: (arr * 255.0).astype(jnp.uint8), # recover for storage
                    lambda arr: arr.astype(jnp.uint8),
                    flattened
                )
            return flattened

        # flatten and cast dtype in one go
        transitions_flatten = jax.tree.map(flatten_fn, experiences.transition)

        mask = transitions_flatten.done
        N = mask.shape[0]

        if experiences.terminal_observation is None:
            return transitions_flatten, None

        terminal_obs_flatten = flatten_fn(experiences.terminal_observation)

        # Indices for step transitions; we replenish ones at done = True with terminal_obs
        shifts = jnp.concatenate([jnp.array([False]), mask[:-1]])
        step_indices = jnp.arange(N) + jnp.cumsum(shifts)

        # Indices for reset transitions
        reset_indices = step_indices + 1 # To keep shape static; only indices at mask are meaningful

        # Construct reset transitions
        reset_transitions = jax.tree.map(lambda x: jnp.zeros_like(x), transitions_flatten)
        reset_transitions = eqx.tree_at(
            lambda x: x.next_obs,
            reset_transitions,
            transitions_flatten.next_obs
        )

        # Replenish terminal_obs
        mask_expanded = mask.reshape((N,) + (1,) * (terminal_obs_flatten.ndim - 1))
        new_next_obs = jnp.where(
            mask_expanded,
            terminal_obs_flatten,
            transitions_flatten.next_obs
        )
        step_transitions = eqx.tree_at(
            lambda x: x.next_obs,
            transitions_flatten,
            new_next_obs
        )

        padded_length = 2 * N
        # Create the merged empty array
        def merge_fn(step_leaf, reset_leaf):
            new_shape = (padded_length, *step_leaf.shape[1:]) # Fixed length for jit
            out = jnp.zeros(new_shape, dtype=step_leaf.dtype)
            out = out.at[step_indices].set(step_leaf)
            # Trick
            _reset_indices = jnp.where(mask, reset_indices, padded_length)
            out = out.at[_reset_indices].set(reset_leaf, mode='drop')
            return out

        valid_length = N + jnp.sum(mask) # Actual length
        return jax.tree.map(merge_fn, step_transitions, reset_transitions), valid_length

    def add_experience(self, experiences: Experience, source: int = 1) -> "Agent":
        transitions_flatten, valid_length = self.replenish_and_flatten(experiences, source) # handle terminal obs; critical for world modeling e.g. predict reward
        new_memory = self.memory.add(transitions_flatten, valid_length)
        return eqx.tree_at(
            lambda x: x.memory,
            self,
            new_memory
        )

    def reason(self, data: Transition, key: PRNGKeyArray) -> Tuple[LatentStateWithParams, LatentStateWithParams]:
        """
        Reason about the relationship among data and to the goal with contexts from predictive models or memory given a fixed belief;
        Reasoning is on-demand learning, which creates new knowledge and will be offloaded to the offline learning stage, e.g. dreaming
        """
        key_init, key_scan = jax.random.split(key, 2)
        init_latent_state = self.init_state(key_init, batch_shape=(data.action.shape[1],))
        init_mask = jnp.ones_like(data.done[0], dtype=jnp.int32) # fixed
        next_obs = jax.vmap(jax.vmap(self.world.perception.encoder))(data.next_obs) # Launch kernel once

        def reason_step_fn(carry, inputs):
            latent_state, last_mask, key = carry
            action, obs, done = inputs
            key, key_perceive = jax.random.split(key)

            # Mask the state if the last step is done; action is already zero
            latent_state = latent_state * last_mask
            prior, posterior = self.perceive(latent_state, action, obs, key_perceive)

            # Update mask
            mask = 1 - done # fixed
            return (posterior.latent_state, mask, key), (prior, posterior)

        _, (priors, posteriors) = jax.lax.scan(
            reason_step_fn,
            (init_latent_state, init_mask, key_scan),
            (data.action, next_obs, data.done)
        )
        return priors, posteriors

    def plan(self, latent_state: LatentState, key: PRNGKeyArray) -> Tuple[jax.Array, distrax.Distribution]:
        # Imagination
        def imagine_step_fn(carry, _): # TODO: integrate the logic of masking inputs when terminated; require a terminal predictor d(s, a)
            latent_state, key = carry

            key, key_action, key_predict = jax.random.split(key, 3)
            params = jax.vmap(self.actor)(latent_state.detach())
            action = self.actor.sample(params, key_action)
            prior = self.predict(latent_state, action, key_predict)

            return (prior.latent_state, key), latent_state

        (last_latent_state, _), latent_states_before_last = jax.lax.scan(
            imagine_step_fn,
            (latent_state.detach(), key),
            None,
            self.planning_horizon,
        )
        return LatentState.concatenate([latent_states_before_last, last_latent_state[None, ...]])

    def process(self, latent_states: LatentState) -> Tuple[jax.Array, ...]: # TODO: typing for the output as a general abstraction
        # Processing imagined data
        rewards = jax.vmap(jax.vmap(self.world.reward))(latent_states[1:]).mean() # Equivalent to r(s, a, s') instead of r(s, a)
        values = jax.vmap(
            jax.vmap(jax.lax.stop_gradient(self.critic))
        )(latent_states).mean()
        values_before_last = values[:-1]
        last_value = values[-1]

        dones = jnp.zeros_like(values_before_last)
        baselines = jnp.zeros_like(values_before_last)

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

        input_carry = (jnp.zeros_like(last_value), last_value)

        _, return_predictions = jax.lax.scan(
            uae_step_fn,
            input_carry,
            (rewards, values_before_last, baselines, dones),
            reverse=True,
        )
        return return_predictions, rewards

    def learn(self, key: PRNGKeyArray) -> Tuple["Agent", Dict[str, jax.Array]]:
        """Update world model, actor and critic."""
        key, key_memory, key_world, key_actor, key_critic = jax.random.split(key, 5)
        data = self.memory.sample((self.batch_size, self.chunk_size), key_memory) # T x B
        data = eqx.tree_at(
            lambda d: d.next_obs,
            data,
            replace_fn=self.transform
        )
        metrics = {}

        @eqx.filter_value_and_grad(has_aux=True)
        @differentiable(['world'])
        def world_loss_fn(
                agent: Agent,
                data: Transition,
                key: PRNGKeyArray,
        ):
            prior, posterior = agent.reason(data, key)

            reward_dist = jax.vmap(jax.vmap(agent.world.reward))(posterior.latent_state)
            reward_log_prob = reward_dist.log_prob(data.reward) # fixed
            reward_loss = -reward_log_prob.mean()

            observation_dist = jax.vmap(jax.vmap(agent.world.perception.decoder))(posterior.latent_state)
            observation_log_prob = observation_dist.log_prob(data.next_obs)
            observation_loss = -observation_log_prob.mean()

            if self.kl_balance > 0:
                kl_loss_post = posterior.kl_divergence(jax.lax.stop_gradient(prior).dist).sum(-1)
                kl_loss_prior = jax.lax.stop_gradient(posterior).kl_divergence(prior.dist).sum(-1)

                if self.kl_average:
                    kl_loss_post = kl_loss_post.mean()
                    kl_loss_prior = kl_loss_prior.mean()

                if self.free_nats > 0:
                    kl_loss_post = jnp.clip(kl_loss_post, min=self.free_nats)
                    kl_loss_prior = jnp.clip(kl_loss_prior, min=self.free_nats)

                kl_loss = self.kl_balance * kl_loss_prior + (1 - self.kl_balance) * kl_loss_post
            else:
                kl_loss = posterior.kl_divergence(prior.dist).sum(-1)

                if self.kl_average:
                    kl_loss = kl_loss.mean()

                if self.free_nats > 0:
                    kl_loss = jnp.clip(kl_loss, min=self.free_nats)

            kl_loss = kl_loss if self.kl_average else kl_loss.mean()

            total_loss = reward_loss + observation_loss + kl_loss

            # For logging
            reward_mse = jnp.mean((reward_dist.mean() - data.reward)**2) # fixed
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

        (loss, (aux, posterior)), grads = world_loss_fn(self, data, key_world)
        new_world = self.world.update(grads.world)
        agent = eqx.tree_at(lambda x: x.world, self, new_world)
        metrics.update(**aux)

        @eqx.filter_value_and_grad(has_aux=True)
        @differentiable(['actor'])
        def actor_loss_fn(
                agent: Agent,
                posterior: LatentStateWithParams,
                key: PRNGKeyArray,
        ):  # TODO: add PG
            imagined_latent_states = agent.plan(posterior.latent_state.flatten(), key)
            return_prediction, imagined_rewards = agent.process(imagined_latent_states)

            actor_loss = -return_prediction.mean()
            metrics = {
                "ac/actor": actor_loss,
                "aux/return_prediction": return_prediction.mean(),
                "aux/imagined_rewards": imagined_rewards.mean(),
            }
            return actor_loss, (metrics, imagined_latent_states, return_prediction)

        (loss, (aux, imagined_latent_states, return_prediction)), grads = actor_loss_fn(agent, posterior, key_actor)
        new_actor = self.actor.update(grads.actor)
        agent = eqx.tree_at(
            lambda x: x.actor,
            agent,
            new_actor
        )
        metrics.update(**aux)

        @eqx.filter_value_and_grad(has_aux=True)
        @differentiable(['critic'])
        def critic_loss_fn(
                agent: Agent,
                imagined_latent_states: jax.Array,
                return_prediction: jax.Array,
                key: PRNGKeyArray,
        ):  # TODO: add PG
            value_dist = jax.vmap(jax.vmap(agent.critic))(imagined_latent_states[:-1].detach())
            critic_loss = -value_dist.log_prob(jax.lax.stop_gradient(return_prediction))
            critic_loss = critic_loss.mean()

            metrics = {
                "ac/critic": critic_loss,
            }
            return critic_loss, metrics

        (loss, aux), grads = critic_loss_fn(agent, imagined_latent_states, return_prediction, key_critic)
        new_critic = self.critic.update(grads.critic)
        agent = eqx.tree_at(
            lambda x: x.critic,
            agent,
            new_critic
        )
        metrics.update(**aux)
        return agent, metrics
