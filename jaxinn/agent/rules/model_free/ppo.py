from typing import Any, Dict, Optional, Tuple, ClassVar, Type

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree
import equinox as eqx
import distrax

from jaxinn.configs import PPOAgentConfig, ActorCriticSharedConfig, ActorCriticDecoupledConfig, PerceptionActorConfig, PerceptionCriticConfig
from jaxinn.agent.registry import Registrable
from jaxinn.agent.rules.base import Agent
from jaxinn.agent.rules.learner import Learner
from jaxinn.agent.rules.utils import compute_adv_and_ret
from jaxinn.agent.models import Actor, Critic
from jaxinn.agent.models.world.perception import Encoder
from jaxinn.agent.losses import PPOLossMixIn
from jaxinn.agent.memory import Memory, Batched


class PerceptionActor(eqx.Module):
    encoder: Encoder
    actor: Actor

    @classmethod
    def create(cls, config: PerceptionActorConfig, *, key: PRNGKeyArray):
        key_encoder, key_actor = jax.random.split(key)

        encoder = Encoder.create(config.encoder, key=key_encoder)
        actor = Actor(
            **config.actor(),
            key=key_actor
        )

        return cls(encoder=encoder, actor=actor)

    def __call__(self, obs: jax.Array):
        feature = self.encoder(obs)
        return self.actor(feature)

    def get_dist(self, params: Dict[str, Any]) -> distrax.Distribution:
        return self.actor.get_dist(params)

    def sample(
            self,
            params: Dict[str, Any],
            key: PRNGKeyArray,
            det: bool = False,
    ) -> jax.Array:
        return self.actor.sample(params, key, det=det)


class PerceptionCritic(eqx.Module):
    encoder: Encoder
    critic: Critic

    @classmethod
    def create(cls, config: PerceptionCriticConfig, *, key: PRNGKeyArray):
        key_encoder, key_critic = jax.random.split(key)

        encoder = Encoder.create(config.encoder, key=key_encoder)
        critic = Critic(
            **config.critic(),
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

        actor = PerceptionActor.create(config.perception_actor, key=key_actor)
        critic = PerceptionCritic.create(config.perception_critic, key=key_critic)

        return cls(actor=actor, critic=critic)

    def get_actor_dist(self, obs: jax.Array) -> distrax.Distribution:
        actor_params = self.actor(obs)
        actor_dist = self.actor.get_dist(actor_params)
        return actor_dist

    def get_critic_dist(self, obs: jax.Array) -> distrax.Distribution:
        return self.critic(obs)

    def sample_action(self, obs: jax.Array, key: PRNGKeyArray, det: bool = False) -> jax.Array:
        actor_params = self.actor(obs)
        action = self.actor.sample(actor_params, key, det)
        return action


# Shared encoder
class ActorCriticShared(ActorCritic):
    config_cls: ClassVar[Type] = ActorCriticSharedConfig

    encoder: Encoder
    actor: Actor
    critic: Critic

    @classmethod
    def create(cls, config: ActorCriticSharedConfig, *, key: PRNGKeyArray): # TODO: import
        key_encoder, key_actor, key_critic = jax.random.split(key, 3
                                                              )
        encoder = Encoder.create(config.encoder, key=key_encoder)
        actor = Actor(**config.actor(), key=key_actor)
        critic = Critic(**config.critic(), key=key_critic)

        return cls(encoder=encoder, actor=actor, critic=critic)

    def get_actor_dist(self, obs: jax.Array) -> distrax.Distribution:
        feature = self.encoder(obs)
        actor_params = self.actor(feature)
        actor_dist = self.actor.get_dist(actor_params)
        return actor_dist

    def get_critic_dist(self, obs: jax.Array) -> distrax.Distribution:
        feature = self.encoder(obs)
        return self.critic(feature)

    def sample_action(self, obs: jax.Array, key: PRNGKeyArray, det: bool = False) -> jax.Array:
        feature = self.encoder(obs)
        actor_params = self.actor(feature)
        action = self.actor.sample(actor_params, key, det)
        return action


class PPOAgent(PPOLossMixIn, Agent):
    config_cls: ClassVar[Type] = PPOAgentConfig

    actor_critic: Learner[ActorCritic]
    memory: Memory

    clip_param: float = eqx.field(static=True)
    use_clipped_critic_loss: bool = eqx.field(static=True)
    num_mini_batch: int = eqx.field(static=True)
    discount_factor: float = eqx.field(static=True)
    uae_lambda: float = eqx.field(static=True)

    @classmethod
    def create(
            cls,
            config,
            *,
            key: PRNGKeyArray,
            memory_id: jax.Array,
    ):
        actor_critic = Learner.from_config(ActorCritic, config.actor_critic, key=key)
        memory = Batched(
            seed_idx=memory_id,
            capacity=config.memory.capacity,
            obs_shape=config.actor_critic.obs_shape,
            action_size=config.actor_critic.action_size,
            num_seeds=config.memory.num_seeds,
        )

        return cls(
            actor_critic=actor_critic,
            memory=memory,
            **config.optimization() # Extra particulars for agent learning
        )

    def init_state(self, key: PRNGKeyArray, batch_shape: Tuple[int, ...] = (), eval=False) -> Any:
        return None

    def act(self, last_latent_state: Optional[jax.Array], last_action: jax.Array, obs: jax.Array, *, key: PRNGKeyArray, eval: bool = False) -> Tuple[None, jax.Array]:
        action = jax.vmap(self.actor_critic.sample_action, in_axes=(0, None, None))(obs, key, eval)
        return None, action

    def learn(self, key: PRNGKeyArray) -> Tuple["Agent", Dict[str, jax.Array]]:
        key_shuffle, key_scan = jax.random.split(key, 2)
        data = self.memory.get_all()

        # Get advantages and returns
        values = jax.vmap(self.actor_critic.get_critic_dist)(data.next_obs).mean()
        baselines = values[:-1]
        advantages, returns = compute_adv_and_ret(
            data.reward[1:],
            values[:-1],
            baselines,
            data.done[1:],
            bootstrap=values[-1],
            discount_factor=self.discount_factor,
            uae_lambda=self.uae_lambda
        )

        # Get action log probs
        actor_dists = jax.vmap(self.actor_critic.get_actor_dist)(data.next_obs[:-1])
        log_probs = actor_dists.log_prob(data.action[1:])

        # Apply shuffle and split for training data
        train_data = (data.next_obs[:-1], data.action[1:], advantages, returns, values[:-1], log_probs)
        mini_batches = self.shuffle_and_split(train_data, key_shuffle)

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
            (self, key_scan),
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
