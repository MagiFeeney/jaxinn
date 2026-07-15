from typing import Any, Dict, Optional, Tuple, ClassVar, Type

import math
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree
import equinox as eqx
import distrax

from jaxinn.structs import Experience
from jaxinn.configs.agent.ppo import (
    PPOAgentConfig,
    ActorCriticSharedConfig,
    ActorCriticDecoupledConfig,
    PerceptionActorConfig,
    PerceptionCriticConfig
)
from jaxinn.agent.registry import Registrable
from jaxinn.agent.rules.base import Agent
from jaxinn.agent.rules.learner import Learner
from jaxinn.agent.rules.utils import compute_adv_and_ret, staircase_lr_schedule
from jaxinn.agent.models import Actor, Critic
from jaxinn.agent.models.perception import Encoder
from jaxinn.agent.models.utils import apply_init
from jaxinn.agent.losses import PPOLossMixIn


class PerceptionActor(eqx.Module):
    encoder: Encoder
    actor: Actor

    @classmethod
    def create(cls, config: PerceptionActorConfig, *, key: PRNGKeyArray):
        key_encoder, key_actor = jax.random.split(key)

        encoder = Encoder.create(config.encoder, key=key_encoder)
        actor = Actor.create(
            config.actor,
            key=key_actor
        )

        return cls(encoder=encoder, actor=actor)

    def __call__(self, obs: jax.Array):
        feature = self.encoder(obs)
        return self.actor(feature)

    def sample(
            self,
            dist: distrax.Distribution,
            key: PRNGKeyArray,
            det: bool = False,
    ) -> jax.Array:
        return self.actor.sample(dist, key, det=det)


class PerceptionCritic(eqx.Module):
    encoder: Encoder
    critic: Critic

    @classmethod
    def create(cls, config: PerceptionCriticConfig, *, key: PRNGKeyArray):
        key_encoder, key_critic = jax.random.split(key)

        encoder = Encoder.create(config.encoder, key=key_encoder)
        critic = Critic.create(
            config.critic,
            key=key_critic
        )

        return cls(encoder=encoder, critic=critic)

    def __call__(self, obs: jax.Array, action: Optional[jax.Array] = None) -> distrax.Distribution:
        feature = self.encoder(obs)
        return self.critic(feature, action)


# Base class for registration
class ActorCritic(Registrable, eqx.Module):
    pass


# Independent encoders
class ActorCriticDecoupled(ActorCritic):
    config_cls: ClassVar[Type] = ActorCriticDecoupledConfig

    actor: PerceptionActor
    critic: PerceptionCritic

    @classmethod
    def create(cls, config: ActorCriticDecoupledConfig, *, key: PRNGKeyArray):
        key_actor, key_critic = jax.random.split(key, 2)

        key_actor_model, key_actor_init = jax.random.split(key_actor, 2)
        actor = PerceptionActor.create(config.perception_actor, key=key_actor_model)
        actor = apply_init(
            actor,
            weight_init=jax.nn.initializers.orthogonal(scale=math.sqrt(2)),
            output_weight_init=jax.nn.initializers.orthogonal(scale=0.01),
            key=key_actor_init
        )

        key_critic_model, key_critic_init = jax.random.split(key_critic, 2)
        critic = PerceptionCritic.create(config.perception_critic, key=key_critic_model)
        critic = apply_init(
            critic,
            weight_init=jax.nn.initializers.orthogonal(scale=math.sqrt(2)),
            output_weight_init=jax.nn.initializers.orthogonal(scale=1.0),
            key=key_critic_init
        )

        return cls(actor=actor, critic=critic)

    def get_actor_dist(self, obs: jax.Array) -> distrax.Distribution:
        return self.actor(obs)

    def get_critic_dist(self, obs: jax.Array) -> distrax.Distribution:
        return self.critic(obs)


# Shared encoder
class ActorCriticShared(ActorCritic):
    config_cls: ClassVar[Type] = ActorCriticSharedConfig

    encoder: Encoder
    actor: Actor
    critic: Critic

    @classmethod
    def create(cls, config: ActorCriticSharedConfig, *, key: PRNGKeyArray):
        key_encoder, key_actor, key_critic = jax.random.split(key, 3)

        key_encoder_model, key_encoder_init = jax.random.split(key_encoder, 2)
        encoder = Encoder.create(config.encoder, key=key_encoder_model)
        encoder = apply_init(encoder, weight_init=jax.nn.initializers.orthogonal(scale=math.sqrt(2)), key=key_encoder_init)

        key_actor_model, key_actor_init = jax.random.split(key_actor, 2)
        actor = Actor.create(config.actor, key=key_actor_model)
        actor = apply_init(
            actor,
            weight_init=jax.nn.initializers.orthogonal(scale=math.sqrt(2)),
            output_weight_init=jax.nn.initializers.orthogonal(scale=0.01),
            key=key_actor_init
        )

        key_critic_model, key_critic_init = jax.random.split(key_critic, 2)
        critic = Critic.create(config.critic, key=key_critic_model)
        critic = apply_init(
            critic,
            weight_init=jax.nn.initializers.orthogonal(scale=math.sqrt(2)),
            output_weight_init=jax.nn.initializers.orthogonal(scale=1.0),
            key=key_critic_init
        )

        return cls(encoder=encoder, actor=actor, critic=critic)

    def get_actor_dist(self, obs: jax.Array) -> distrax.Distribution:
        feature = self.encoder(obs)
        return self.actor(feature)

    def get_critic_dist(self, obs: jax.Array) -> distrax.Distribution:
        feature = self.encoder(obs)
        return self.critic(feature)


class PPOAgent(PPOLossMixIn, Agent):
    config_cls: ClassVar[Type] = PPOAgentConfig

    actor_critic: Learner[ActorCritic]
    memory: Experience

    clip_param: float = eqx.field(static=True)
    use_clipped_critic_loss: bool = eqx.field(static=True)
    num_mini_batch: int = eqx.field(static=True)
    discount_factor: float = eqx.field(static=True)
    uae_lambda: float = eqx.field(static=True)
    normalize_adv: bool = eqx.field(static=True)

    @classmethod
    def create(
            cls,
            config,
            *,
            key: PRNGKeyArray,
            memory_id: jax.Array,
    ):
        actor_critic = Learner.create(ActorCritic, config.actor_critic, lr_scheduler=staircase_lr_schedule, key=key)
        memory = Experience.initialize(
            capacity=config.memory.capacity,
            obs_shape=config.memory.obs_shape,
            obs_dtype=config.memory.obs_dtype,
            action_shape=config.memory.action_shape,
            action_dtype=config.memory.action_dtype,
            needs_terminal_obs=config.memory.needs_terminal_obs
        )

        return cls(
            actor_critic=actor_critic,
            memory=memory,
            **config.optimization() # Extra particulars for agent learning
        )

    def init_latent_state(self, key: PRNGKeyArray, batch_shape: Tuple[int, ...] = (), eval=False) -> Any:
        return None

    def act(self, last_latent_state: Optional[jax.Array], last_action: jax.Array, obs: jax.Array, *, key: PRNGKeyArray, eval: bool = False) -> Tuple[None, jax.Array]:
        actor_dist = jax.vmap(self.actor_critic.get_actor_dist)(obs)
        action = self.actor_critic.actor.sample(actor_dist, key, eval)
        return None, action

    def make_batch_fn(self) -> callable:
        transition, terminal_obs = self.memory.transition, self.memory.terminal_observation

        # Get advantages and returns
        if terminal_obs is None: # Next-step autoreset
            all_values = jax.vmap(jax.vmap(self.actor_critic.get_critic_dist))(transition.next_obs).mean()
            values = all_values[:-1]
            next_values = all_values[1:]
            last_dones = transition.terminated | transition.truncated
            next_values = next_values * (1 - last_dones[:-1])
        else:                    # current-step autoreset
            values = jax.vmap(jax.vmap(self.actor_critic.get_critic_dist))(transition.next_obs[:-1]).mean()
            next_values = jax.vmap(jax.vmap(self.actor_critic.get_critic_dist))(terminal_obs[1:]).mean() # Recalculate values on actual terminal observation to handle truncation

        baselines = values
        advantages, returns = compute_adv_and_ret(
            transition.reward[1:],
            values,
            baselines,
            transition.terminated[1:],
            next_values,
            discount_factor=self.discount_factor,
            uae_lambda=self.uae_lambda
        )

        # Get action log probs
        actor_dists = jax.vmap(jax.vmap(self.actor_critic.get_actor_dist))(transition.next_obs[:-1])
        actions = jax.tree.map(lambda x: x[1:], transition.action)
        log_probs = actor_dists.log_prob(actions)

        # Apply shuffle and split for training data
        train_data = (transition.next_obs[:-1], actions, advantages, returns, values[:-1], log_probs)
        flatten_train_data = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), train_data)

        def step_fn(key: PRNGKeyArray):
            mini_batches = self.shuffle_and_split(flatten_train_data, key)
            return mini_batches
        return step_fn

    def learn(self, mini_batches: Tuple[jax.Array], key: PRNGKeyArray) -> Tuple["Agent", Dict[str, jax.Array]]:
        def mini_batch_step_fn(carry, mini_batch):
            agent, key = carry
            key, key_ac = jax.random.split(key, 2)
            metrics = {}

            # Update the actor-critic
            (loss, aux), grads = agent.actor_critic_loss_fn(mini_batch, key_ac)
            new_actor_critic = agent.actor_critic.update(grads.actor_critic)
            agent = eqx.tree_at(lambda x: x.actor_critic, agent, new_actor_critic)
            metrics.update(**aux)

            return (agent, key), metrics

        (agent, _), metrics = jax.lax.scan(
            mini_batch_step_fn,
            (self, key),
            mini_batches
        )
        avg_metrics = jax.tree.map(jnp.mean, metrics)
        return agent, avg_metrics

    def shuffle_and_split(self, batch: PyTree, key: PRNGKeyArray):
        size = jax.tree.leaves(batch)[0].shape[0]

        sample_index = jax.random.permutation(key, size)
        shuffled_batch = jax.tree.map(lambda x: x[sample_index], batch)

        mini_batch_size = size // self.num_mini_batch
        valid_size = mini_batch_size * self.num_mini_batch

        split_data = jax.tree.map(
            lambda x: x[:valid_size].reshape(self.num_mini_batch, mini_batch_size, *x.shape[1:]),
            shuffled_batch
        )
        return split_data
