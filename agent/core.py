import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict, Generic, TypeVar
from jaxtyping import PRNGKeyArray

import equinox as eqx
import distrax
import optax

from envs import Transition
from .models import World, Critic, Actor, LatentState, LatentStateWithParams
from .memory import Memory, Uniform, Prioritized
from .utils import differentiable


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

    def init_state(self, key: PRNGKeyArray, batch_shape: Tuple[int, ...] = ()) -> LatentState:
        return LatentState.initialize(self.belief_size, self.state_size, self.random_init, batch_shape, key=key)

    def process(self, obs) -> jax.Array:
        return obs.astype(jnp.float32) / 255.0 - 0.5

    def act(self, last_latent_state: LatentState, last_action: jax.Array, obs: jax.Array, *, key: PRNGKeyArray, eval: bool = False) -> Tuple[LatentState, jax.Array]:
        key_perceive, key_action = jax.random.split(key, 2)
        obs = jax.vmap(self.world.perception.encoder)(self.process(obs))
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
    def replenish_and_flatten(transitions: Transition, terminal_obs: jax.Array | None = None) -> Tuple[Transition, jax.Array]:
        def flatten_fn(x):
            # (T, B, ...) -> (B*T, ...)
            flattened = jnp.swapaxes(x, 0, 1).reshape(-1, *x.shape[2:])
            # For storage
            if x.dtype == jnp.float32 and x.ndim > 3:
                return flattened.astype(jnp.uint8)
            return flattened

        # flatten and cast dtype in one go
        transitions_flatten = jax.tree.map(flatten_fn, transitions)

        if terminal_obs is None:
            return transitions_flatten

        terminal_obs_flatten = flatten_fn(terminal_obs)

        mask = transitions_flatten.done
        N = mask.shape[0]

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
        new_next_obs = jnp.where(
            mask[:, None, None, None],
            terminal_obs_flatten,
            transitions_flatten.next_obs
        ) # TODO: handle vector obs
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

    def add_experience(self, transitions: Transition, terminal_obs: jax.Array | None = None) -> "Agent":
        transitions_flatten, valid_length = self.replenish_and_flatten(transitions, terminal_obs) # handle terminal obs; critical for world modeling e.g. predict reward
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
        init_mask = jnp.ones_like(data.done[0][..., None], dtype=jnp.int32)
        next_obs = jax.vmap(jax.vmap(self.world.perception.encoder))(self.process(data.next_obs)) # Launch kernel once

        def reason_step_fn(carry, inputs):
            latent_state, last_mask, key = carry
            action, obs, done = inputs
            key, key_perceive = jax.random.split(key)

            # Mask the state if the last step is done; action is already zero
            latent_state = latent_state * last_mask # TODO: test the performance for the above
            prior, posterior = self.perceive(latent_state, action, obs, key_perceive)

            # Update mask
            mask = 1 - done[..., None]
            return (posterior.latent_state, mask, key), (prior, posterior)

        _, (priors, posteriors) = jax.lax.scan(
            reason_step_fn,
            (init_latent_state, init_mask, key_scan),
            (data.action, next_obs, data.done) # TODO: env interaction part may also need to check
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

            return (prior.latent_state, key), (prior.latent_state, action)

        _, (latent_states, actions) = jax.lax.scan(
            imagine_step_fn,
            (latent_state.detach(), key),
            None,
            self.planning_horizon,
        )

        # Processing data
        rewards = jax.vmap(jax.vmap(self.world.reward))(latent_states).mean() # Equivalent to r(s, a, s') instead of r(s, a)
        last_value = jax.vmap(self.critic)(latent_states[-1]).mean()
        latent_states = LatentState.concatenate([latent_state[None, ...], latent_states[:-1]], axis=0) # Shift one step left and concatenate the first step
        value_dists = jax.vmap(jax.vmap(self.critic))(latent_states)
        values = value_dists.mean()

        dones = jnp.zeros_like(values)
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
            uae_step_fn,
            (jnp.zeros_like(last_value), last_value),
            (rewards, values, baselines, dones),
            reverse=True,
        )

        return return_predictions, value_dists

    def learn(self, key: PRNGKeyArray) -> Tuple["Agent", Dict[str, jax.Array]]:
        """Update world model, actor and critic."""
        key, key_memory, key_world, key_ac = jax.random.split(key, 4)
        data = self.memory.sample((self.batch_size, self.chunk_size), key_memory) # T x B
        metrics = {}

        @eqx.filter_value_and_grad(has_aux=True)
        @differentiable(['world'])
        def world_loss_fn(
                agent: Agent,
                data: Transition,
                key: PRNGKeyArray,
        ):
            prior, posterior = agent.reason(data, key)

            reward_loss = -jax.vmap(jax.vmap(agent.world.reward))(posterior.latent_state).log_prob(data.reward).mean()
            observation_loss = -jax.vmap(jax.vmap(agent.world.perception.decoder))(posterior.latent_state).log_prob(data.next_obs).mean()
            kl_loss = posterior.kl_divergence(prior.dist).mean()
            total_loss = reward_loss + observation_loss + kl_loss

            metrics = {
                "model/reward": reward_loss,
                "model/observation": observation_loss,
                "model/kl": kl_loss,
                "model/total": total_loss,
            }
            return total_loss, (metrics, posterior)

        (loss, (aux, posterior)), grads = world_loss_fn(self, data, key_world)
        new_world = self.world.update(grads.world)
        metrics.update(**aux)

        # Incorporate the newly updated world
        agent = eqx.tree_at(lambda x: x.world, self, new_world)

        @eqx.filter_value_and_grad(has_aux=True)
        @differentiable(['actor', 'critic'])
        def ac_loss_fn(
                agent: Agent,
                posterior: LatentStateWithParams,
                key: PRNGKeyArray,
        ):  # TODO: add PG
            return_prediction, value_dist = agent.plan(posterior.latent_state.flatten(), key) # TODO: keep data as is and add processor for processing
            actor_loss = -return_prediction.mean()
            critic_loss = -value_dist.log_prob(jax.lax.stop_gradient(return_prediction)).mean()
            total_loss = actor_loss + critic_loss # non-interleaved

            metrics = {
                "actor": actor_loss,
                "critic": critic_loss,
            }
            return total_loss, metrics

        (loss, aux), grads = ac_loss_fn(agent, posterior, key_ac) # TODO: make grads split inside the wrapper
        new_actor = self.actor.update(grads.actor)
        new_critic = self.critic.update(grads.critic)
        metrics.update(**aux)

        # Incorporate the newly updated actor and critic
        agent = eqx.tree_at(
            lambda x: (x.actor, x.critic),
            agent,
            (new_actor, new_critic)
        )
        return agent, metrics
