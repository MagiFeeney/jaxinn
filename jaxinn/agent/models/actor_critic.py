from typing import ClassVar

import jax
from jaxtyping import PRNGKeyArray

from jaxinn.configs.model import (
    ActorCriticSharedConfig,
    ActorCriticDecoupledConfig,
)
from jaxinn.configs.agent.ppo import (
    PPOActorCriticSharedConfig,
    PPOActorCriticDecoupledConfig,
)
from jaxinn.agent.registry import Registrable

from .base import Model
from .actor import Actor, PerceptionActor
from .critic import Critic, PerceptionCritic
from .perception import Encoder
from .distributions import DistributionLike


# Base class for registration
class ActorCritic(Registrable, Model):
    pass


# Independent encoders
class ActorCriticDecoupled(ActorCritic):
    config_cls: ClassVar[tuple[type, ...]] = (
        ActorCriticDecoupledConfig,
        PPOActorCriticDecoupledConfig
    )

    actor: PerceptionActor
    critic: PerceptionCritic

    @classmethod
    def create(cls, config: ActorCriticDecoupledConfig, *, key: PRNGKeyArray):
        key_actor, key_critic, key_init = jax.random.split(key, 3)

        actor = PerceptionActor.create(config.perception_actor, key=key_actor)
        critic = PerceptionCritic.create(config.perception_critic, key=key_critic)

        return cls(actor=actor, critic=critic).apply_init(config.initializer, key=key_init)

    def get_actor_dist(self, obs: jax.Array) -> DistributionLike:
        return self.actor(obs)

    def get_critic_dist(self, obs: jax.Array) -> DistributionLike:
        return self.critic(obs)


# Shared encoder
class ActorCriticShared(ActorCritic):
    config_cls: ClassVar[tuple[type, ...]] = (
        ActorCriticSharedConfig,
        PPOActorCriticSharedConfig
    )

    encoder: Encoder
    actor: Actor
    critic: Critic

    @classmethod
    def create(cls, config: ActorCriticSharedConfig, *, key: PRNGKeyArray):
        key_encoder, key_actor, key_critic, key_init = jax.random.split(key, 4)

        encoder = Encoder.create(config.encoder, key=key_encoder)
        actor = Actor.create(config.actor, key=key_actor)
        critic = Critic.create(config.critic, key=key_critic)

        return cls(encoder=encoder, actor=actor, critic=critic).apply_init(config.initializer, key=key_init)

    def get_actor_dist(self, obs: jax.Array) -> DistributionLike:
        feature = self.encoder(obs)
        return self.actor(feature)

    def get_critic_dist(self, obs: jax.Array) -> DistributionLike:
        feature = self.encoder(obs)
        return self.critic(feature)
