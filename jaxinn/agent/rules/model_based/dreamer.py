from typing import Dict, Tuple, ClassVar, Type
from jaxtyping import PRNGKeyArray

import jax
import jax.numpy as jnp
import equinox as eqx

from jaxinn.structs import Transition, LatentState, LatentStateWithDist
from jaxinn.configs.agent.dreamer import (
    DreamerAgentConfig,
    DreamerV2AgentConfig,
)
from jaxinn.agent.rules.base import Agent
from jaxinn.agent.rules.learner import Learner
from jaxinn.agent.rules.utils import transform, compute_adv_and_ret
from jaxinn.agent.losses import DreamerLossMixIn, MixedActorGradientLoss
from jaxinn.agent.memory import Memory, Uniform, Prioritized
from jaxinn.agent.models import World, Actor, Critic


class DreamerAgent(DreamerLossMixIn, Agent):
    config_cls: ClassVar[Type] = DreamerAgentConfig

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

    @classmethod
    def create(
            cls,
            config,
            *,
            key: PRNGKeyArray,
            memory_id: jax.Array,
    ):
        key_world, key_actor, key_critic = jax.random.split(key, 3)

        world = Learner.create(World, config.world, key=key_world)
        actor = Learner.create(Actor, config.actor, key=key_actor)
        critic = Learner.create(Critic, config.critic, key=key_critic)

        if config.memory.type.lower() == "uniform": # TODO: fix this
            memory_cls = Uniform
        else:
            memory_cls = Prioritized
        memory = memory_cls(
            seed_idx=memory_id,
            capacity=config.memory.capacity,
            obs_shape=config.memory.obs_shape,
            obs_dtype=config.memory.obs_dtype,
            action_shape=config.memory.action_shape,
            action_dtype=config.memory.action_dtype,
            num_seeds=config.memory.num_seeds,
        )

        # For initialization of LatentState
        random_init = config.random_init
        belief_size = config.world.transition.belief_size
        state_size = config.world.transition.state_size

        return cls(
            world=world,
            actor=actor,
            critic=critic,
            memory=memory,
            random_init=random_init,
            belief_size=belief_size,
            state_size=state_size,
            **config.optimization() # Extra particulars for agent learning
        )

    def init_latent_state(self, key: PRNGKeyArray, batch_shape: Tuple[int, ...] = (), eval=False) -> LatentState:
        return LatentState.initialize(self.belief_size, self.state_size, False if eval else self.random_init, batch_shape, key=key)

    def act(self, last_latent_state: LatentState, last_action: jax.Array, obs: jax.Array, *, key: PRNGKeyArray, eval: bool = False) -> Tuple[LatentState, jax.Array]:
        key_perceive, key_action = jax.random.split(key, 2)
        obs = jax.vmap(self.world.perception.encoder)(transform(obs))
        _, posterior = self.perceive(last_latent_state, last_action, obs, key_perceive)
        actor_dist = jax.vmap(self.actor)(posterior.latent_state)
        action = self.actor.sample(actor_dist, key_action, eval)
        return posterior.latent_state, action

    def predict(self, latent_state: LatentState, action: jax.Array, key: PRNGKeyArray) -> LatentStateWithDist:
        """Predict based on the belief without seeing observation."""
        dist, belief = jax.vmap(self.world.transition)(latent_state, action)
        state = self.world.transition.sample(dist, key)
        prior = LatentStateWithDist(
            latent_state=LatentState(belief=belief, state=state),
            fixed_dist=dist,
        )
        return prior

    def perceive(self, latent_state: LatentState, action: jax.Array, observation: jax.Array, key: PRNGKeyArray) -> Tuple[LatentStateWithDist, LatentStateWithDist]:
        """
        Perception is the process of recognizing existing knowledge or deriving new information from sensory inputs based on memory, representations, and predictive models.
        """
        key_prior, key_posterior = jax.random.split(key, 2)
        prior = self.predict(latent_state, action, key_prior)
        dist, belief = jax.vmap(self.world.representation)(prior.latent_state.belief, observation)
        state = self.world.representation.sample(dist, key_posterior)
        posterior = LatentStateWithDist(
            latent_state=LatentState(belief=belief, state=state),
            fixed_dist=dist,
        )
        return prior, posterior

    def reason(self, data: Transition, key: PRNGKeyArray) -> Tuple[LatentStateWithDist, LatentStateWithDist]:
        """
        Reason about the relationship among data and to the goal with contexts from predictive models or memory given a fixed belief;
        Reasoning is on-demand learning, which creates new knowledge and will be offloaded to the offline learning stage, e.g. dreaming
        """
        key_init, key_scan = jax.random.split(key, 2)
        init_latent_state = self.init_latent_state(key_init, batch_shape=(data.action.shape[1],))
        init_mask = jnp.ones_like(data.terminated[0], dtype=jnp.int32)
        next_obs = jax.vmap(jax.vmap(self.world.perception.encoder))(data.next_obs) # Launch kernel once

        def reason_step_fn(carry, inputs):
            latent_state, last_mask, key = carry
            action, obs, done = inputs
            key, key_perceive = jax.random.split(key)

            # Mask the state if the last step is done; action is already zero
            latent_state = latent_state * last_mask
            prior, posterior = self.perceive(latent_state, action, obs, key_perceive)

            # Update mask
            mask = 1 - done
            return (posterior.latent_state, mask, key), (prior, posterior)

        _, (priors, posteriors) = jax.lax.scan(
            reason_step_fn,
            (init_latent_state, init_mask, key_scan),
            (data.action, next_obs, data.terminated | data.truncated)
        )
        return priors, posteriors

    def plan(self, latent_state: LatentState, key: PRNGKeyArray) -> Tuple[jax.Array, jax.Array]:
        # Imagination
        def imagine_step_fn(carry, _): # TODO: integrate the logic of masking inputs when terminated; require a terminal predictor d(s, a)
            latent_state, key = carry

            key, key_action, key_predict = jax.random.split(key, 3)
            actor_dist = jax.vmap(self.actor)(latent_state.detach())
            action = self.actor.sample(actor_dist, key_action)
            prior = self.predict(latent_state, action, key_predict)

            return (prior.latent_state, key), (latent_state, action)

        (last_latent_state, _), (latent_states_before_last, actions) = jax.lax.scan(
            imagine_step_fn,
            (latent_state.detach(), key),
            None,
            self.planning_horizon,
        )
        return LatentState.concatenate([latent_states_before_last, last_latent_state[None, ...]]), actions

    def process(self, latent_states: LatentState) -> Tuple[Tuple[jax.Array, ...], Dict[str, jax.Array]]:
        # Processing imagined data
        rewards = jax.vmap(jax.vmap(self.world.reward))(latent_states[1:]).mean() # Equivalent to r(s, a, s') instead of r(s, a)
        values = jax.vmap(
            jax.vmap(jax.lax.stop_gradient(self.critic))
        )(latent_states).mean()
        values_before_last = values[:-1]

        dones = jnp.zeros_like(values_before_last) # TODO: replace with termination predictor
        baselines = values_before_last

        advantages, return_predictions = compute_adv_and_ret(
            rewards,
            values_before_last,
            baselines,
            dones,
            next_values=values[1:],
            discount_factor=self.discount_factor,
            uae_lambda=self.uae_lambda
        )
        out = (advantages, return_predictions)
        metrics = {
            "aux/return_prediction": return_predictions.mean(),
            "aux/imagined_rewards": rewards.mean(),
        }
        return out, metrics

    def make_batch_fn(self) -> callable:
        def step_fn(key: PRNGKeyArray):
            data = self.memory.sample((self.batch_size, self.chunk_size), key) # T x B
            data = eqx.tree_at(
                lambda d: d.next_obs,
                data,
                replace_fn=transform
            )
            return data
        return step_fn

    def learn(self, data: Transition, key: PRNGKeyArray) -> Tuple["Agent", Dict[str, jax.Array]]:
        """Update world model, actor and critic."""
        key, key_world, key_actor, key_critic = jax.random.split(key, 4)
        metrics = {}

        # Update world model
        (loss, (aux, posterior)), grads = self.world_loss_fn(data, key_world)
        new_world = self.world.update(grads.world)
        agent = eqx.tree_at(lambda x: x.world, self, new_world)
        metrics.update(**aux)

        # Update actor
        (loss, (aux, (imagined_latent_states, return_predictions))), grads = agent.actor_loss_fn(posterior, key_actor)
        new_actor = agent.actor.update(grads.actor)
        agent = eqx.tree_at(lambda x: x.actor, agent, new_actor)
        metrics.update(**aux)

        # Update critic
        (loss, aux), grads = agent.critic_loss_fn(imagined_latent_states, return_predictions, key_critic)
        new_critic = agent.critic.update(grads.critic)
        agent = eqx.tree_at(lambda x: x.critic, agent, new_critic)
        metrics.update(**aux)

        return agent, metrics


class DreamerV2Agent(MixedActorGradientLoss, DreamerAgent):
    config_cls: ClassVar[Type] = DreamerV2AgentConfig

    pg_mix: float = eqx.field(static=True)

    def process(self, latent_states: LatentState, actions: jax.Array) -> Tuple[Tuple[jax.Array, ...], Dict[str, jax.Array]]:
        processed, metrics = super().process(latent_states)

        actor_dists = jax.vmap(jax.vmap(self.actor))(latent_states[:-1])
        action_log_probs = actor_dists.log_prob(jax.lax.stop_gradient(actions))

        out = (*processed, action_log_probs)
        return out, metrics
