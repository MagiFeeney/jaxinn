from typing import Any, ClassVar
from collections.abc import Callable
from jaxtyping import PRNGKeyArray

import jax
import jax.numpy as jnp
import equinox as eqx

from jaxinn.common.structs import Transition, LatentState, LatentStateWithDist
from jaxinn.common.transforms import Chain, Stateless, Scale, EMANormalizer
from jaxinn.configs.agent.dreamer import (
    DreamerAgentConfig,
    DreamerV2AgentConfig,
)
from jaxinn.agent.rules.base import Agent
from jaxinn.agent.rules.learner import Learner
from jaxinn.agent.rules.utils import compute_adv_and_ret, soft_update
from jaxinn.agent.losses import DreamerLossMixIn, MixedActorGradientLoss
from jaxinn.agent.memory import Memory
from jaxinn.agent.models import World, Actor, Critic
from jaxinn.agent.models.distributions import SampleDist


class DreamerAgent(DreamerLossMixIn, Agent):
    config_cls: ClassVar[type] = DreamerAgentConfig

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

    obs_transform: Callable | None = eqx.field(static=True)
    imagined_reward_transform: Callable | None = eqx.field(static=True)

    @classmethod
    def create(cls, config: Any, *, key: PRNGKeyArray, memory_id: jax.Array):
        kwargs = cls._build_kwargs(config, key=key, memory_id=memory_id)
        return cls(**kwargs)

    @classmethod
    def _build_kwargs(cls, config: DreamerAgentConfig, *, key: PRNGKeyArray, memory_id: jax.Array) -> dict[str, Any]:
        key_world, key_actor, key_critic = jax.random.split(key, 3)

        world = Learner.create(World, config.world, key=key_world)
        actor = Learner.create(Actor, config.actor, key=key_actor)
        critic = Learner.create(Critic, config.critic, key=key_critic)

        memory = Memory.create(config.memory, seed_idx=memory_id)

        if config.memory.obs_dtype == jnp.uint8 and len(config.memory.obs_shape) == 3:
            obs_transform = Stateless(
                forward=lambda x: x.astype(jnp.float32) / 255.0 - 0.5,
                inverse=lambda x: (x + 0.5) * 255.0,
            )
        else:
            obs_transform = None

        imagined_reward_transform = None

        # For initialization of LatentState
        random_init = config.random_init
        belief_size = config.world.model.transition.belief_size
        state_size = config.world.model.transition.state_size

        return dict(
            world=world,
            actor=actor,
            critic=critic,
            memory=memory,
            random_init=random_init,
            belief_size=belief_size,
            state_size=state_size,
            obs_transform=obs_transform,
            imagined_reward_transform=imagined_reward_transform,
            **config.optimization() # Extra particulars for agent learning
        )

    def init_latent_state(self, key: PRNGKeyArray, batch_shape: tuple[int, ...] = (), eval=False) -> LatentState:
        return LatentState.initialize(self.belief_size, self.state_size, False if eval else self.random_init, batch_shape, key=key)

    def act(self, last_latent_state: LatentState, last_action: jax.Array, obs: jax.Array, *, key: PRNGKeyArray, eval: bool = False) -> tuple[LatentState, jax.Array]:
        key_perceive, key_action = jax.random.split(key, 2)
        obs = self.obs_transform(obs) if self.obs_transform is not None else obs
        obs = jax.vmap(self.world.encoder)(obs)
        _, posterior = self.perceive(last_latent_state, last_action, obs, key_perceive)
        actor_dist = jax.vmap(self.actor)(posterior.latent_state)

        if not eval:
            action = actor_dist.sample(seed=key)
        else:
            try:
                action = actor_dist.mode()
            except (AttributeError, TypeError, NotImplementedError):
                action = SampleDist(actor_dist).mode(seed=key) # Fall back to sample-based estimates
        return posterior.latent_state, action

    def predict(self, latent_state: LatentState, action: jax.Array, key: PRNGKeyArray) -> LatentStateWithDist:
        """Predict based on the belief without seeing observation."""
        dist, belief = jax.vmap(self.world.transition)(latent_state, action)
        state = dist.sample(seed=key)
        prior = LatentStateWithDist(
            latent_state=LatentState(belief=belief, state=state),
            fixed_dist=dist,
        )
        return prior

    def perceive(self, latent_state: LatentState, action: jax.Array, observation: jax.Array, key: PRNGKeyArray) -> tuple[LatentStateWithDist, LatentStateWithDist]:
        """
        Perception is the process of recognizing existing knowledge or deriving new information from sensory inputs based on memory, representations, and predictive models.
        """
        key_prior, key_posterior = jax.random.split(key, 2)
        prior = self.predict(latent_state, action, key_prior)
        dist, belief = jax.vmap(self.world.representation)(prior.latent_state.belief, observation)
        state = dist.sample(seed=key_posterior)
        posterior = LatentStateWithDist(
            latent_state=LatentState(belief=belief, state=state),
            fixed_dist=dist,
        )
        return prior, posterior

    def reason(self, data: Transition, key: PRNGKeyArray) -> tuple[LatentStateWithDist, LatentStateWithDist]:
        """
        Reason about the relationship among data and to the goal with contexts from predictive models or memory given a fixed belief;
        Reasoning is on-demand learning, which creates new knowledge and will be offloaded to the offline learning stage, e.g. dreaming
        """
        key_init, key_scan = jax.random.split(key, 2)
        init_latent_state = self.init_latent_state(key_init, batch_shape=(data.reward.shape[1],))
        init_mask = jnp.ones_like(data.terminated[0], dtype=jnp.int32)
        next_obs = jax.vmap(jax.vmap(self.world.encoder))(data.next_obs) # Launch kernel once

        def reason_step_fn(carry, inputs):
            latent_state, last_mask, key = carry
            action, obs, done = inputs
            key, key_perceive = jax.random.split(key)

            # Mask the state if the last step is done
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

    def plan(self, latent_state: LatentState, key: PRNGKeyArray) -> tuple[jax.Array, jax.Array]:
        # Imagination
        def imagine_step_fn(carry, _):
            latent_state, key = carry

            key, key_action, key_predict = jax.random.split(key, 3)
            actor_dist = jax.vmap(self.actor)(latent_state.detach())
            action = actor_dist.sample(seed=key_action)
            prior = self.predict(latent_state, action, key_predict)

            return (prior.latent_state, key), (latent_state, action)

        (last_latent_state, _), (latent_states_before_last, actions) = jax.lax.scan(
            imagine_step_fn,
            (latent_state.detach(), key),
            None,
            self.planning_horizon,
        )
        return LatentState.concatenate([latent_states_before_last, last_latent_state[None, ...]]), actions

    def _get_critic(self) -> Critic | Learner[Critic]:
        return self.critic

    def process(self, latent_states: LatentState) -> tuple[tuple[jax.Array, ...], dict[str, jax.Array], tuple[jax.Array, ...]]:
        # Processing imagined data
        rewards = jax.vmap(jax.vmap(self.world.reward))(latent_states[1:]).mean() # Equivalent to r(s, a, s') instead of r(s, a)
        transformed_rewards = self.imagined_reward_transform(rewards) if self.imagined_reward_transform is not None else rewards

        all_values = jax.vmap(
            jax.vmap(jax.lax.stop_gradient(self._get_critic()))
        )(latent_states).mean()
        values = all_values[:-1]
        next_values = all_values[1:]
        baselines = values

        continues = jax.vmap(jax.vmap(self.world.continuation))(latent_states.detach()).mean() if self.world.continuation is not None else jnp.ones_like(all_values)
        merged_discount = self.discount_factor * continues[1:]

        advantages, return_predictions = compute_adv_and_ret(
            transformed_rewards,
            values,
            next_values,
            baselines,
            terminated=None,
            discount_factor=merged_discount,
            uae_lambda=self.uae_lambda
        )

        out = (advantages, return_predictions)
        metrics = {
            "aux/return_prediction": return_predictions.mean(),
            "aux/imagined_rewards": rewards.mean(),
        }
        if self.imagined_reward_transform is not None:
            metrics["aux/imagined_rewards_transformed"] = transformed_rewards.mean()

        state = (continues, merged_discount)
        return out, metrics, state

    def make_batch_fn(self) -> callable:
        def step_fn(key: PRNGKeyArray):
            data = self.memory.sample((self.batch_size, self.chunk_size), key) # T x B
            if self.obs_transform is not None:
                data = eqx.tree_at(
                    lambda d: d.next_obs,
                    data,
                    replace_fn=self.obs_transform
                )
            return data
        return step_fn

    def learn(self, data: Transition, key: PRNGKeyArray) -> tuple["Agent", dict[str, jax.Array]]:
        """Update world model, actor and critic."""
        key, key_world, key_actor, key_critic = jax.random.split(key, 4)
        metrics = {}

        # Update world model
        (loss, (aux, posterior)), grads = self.world_loss_fn(data, key_world)
        new_world = self.world.update(grads.world)
        agent = eqx.tree_at(lambda a: a.world, self, new_world)
        metrics.update(**aux)

        # Update actor
        (loss, (aux, out)), grads = agent.actor_loss_fn(posterior, key_actor)
        new_actor = agent.actor.update(grads.actor)
        agent = eqx.tree_at(lambda a: a.actor, agent, new_actor)
        metrics.update(**aux)

        # Update critic
        (loss, aux), grads = agent.critic_loss_fn(*out, key_critic)
        new_critic = agent.critic.update(grads.critic)
        agent = eqx.tree_at(lambda a: a.critic, agent, new_critic)
        metrics.update(**aux)

        return agent, metrics


class DreamerV2Agent(MixedActorGradientLoss, DreamerAgent):
    config_cls: ClassVar[type] = DreamerV2AgentConfig

    critic_target: Critic

    pg_mix: float = eqx.field(static=True)
    entropy_coef: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    target_update_interval: int = eqx.field(static=True)
    momentum: float = eqx.field(static=True)

    num_updates: jax.Array

    @classmethod
    def _build_kwargs(cls, config: DreamerV2AgentConfig, *, key: PRNGKeyArray, memory_id: jax.Array) -> dict[str, Any]:
        key, key_target = jax.random.split(key, 2)

        kwargs = super()._build_kwargs(config, key=key, memory_id=memory_id)

        critic_target = Critic.create(config.critic.model, key=key_target)

        imagined_reward_transform = Chain(
            transforms=(
                EMANormalizer(
                    shape=(1,),
                    statistics={"magnitude": lambda x: jnp.mean(jnp.abs(x), axis=0)},
                    aggregation=None,
                    init_ema={"magnitude": 1.0},
                    momentum=kwargs["momentum"],
                    center=False
                ),
                Scale(scale=1.0)
            )
        )

        kwargs.update(
            critic_target=critic_target,
            imagined_reward_transform=imagined_reward_transform,
            num_updates=jnp.array(0, dtype=jnp.int32)
        )

        return kwargs

    def _get_critic(self) -> Critic | Learner[Critic]:
        return self.critic_target

    def process(self, latent_states: LatentState, actions: jax.Array, key: PRNGKeyArray) -> tuple[tuple[jax.Array, ...], dict[str, jax.Array], tuple[jax.Array, ...]]:
        # Processing imagined data
        out, metrics, state = super().process(latent_states)
        continues, merged_discount = state

        actor_dists = jax.vmap(jax.vmap(self.actor))(latent_states[:-1].detach())
        action_log_probs = actor_dists.log_prob(jax.lax.stop_gradient(actions))

        # Safe guard on entropy the same way we do for the mode
        try:
            entropies = actor_dists.entropy()
        except (AttributeError, TypeError, NotImplementedError):
            entropies = SampleDist(actor_dists).entropy(seed=key) # Fall back to sample-based estimates

        shifted_discount = jnp.concatenate([continues[:1], merged_discount[:-1]], axis=0)
        weights = jnp.cumprod(shifted_discount, axis=0)

        out = (*out, action_log_probs, entropies, weights)
        return out, metrics, ()

    def learn(self, data: Transition, key: PRNGKeyArray) -> tuple["Agent", dict[str, jax.Array]]:
        agent, metrics = super().learn(data, key)

        do_update = (self.num_updates % self.target_update_interval == 0)
        new_critic_target = jax.lax.cond(
            do_update,
            lambda: soft_update(self.critic_target, agent.critic.model, self.tau),
            lambda: self.critic_target
        )
        agent = eqx.tree_at(
            lambda a: (a.critic_target, a.num_updates),
            agent,
            (new_critic_target, self.num_updates + 1)
        )

        return agent, metrics
