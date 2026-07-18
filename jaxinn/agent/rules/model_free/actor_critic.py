from typing import Any, Optional, ClassVar, Type, Generic, TypeVar

import math
import jax
from jaxtyping import PRNGKeyArray
import equinox as eqx
import distrax

from jaxinn.configs.agent.ppo import (
    ActorCriticSharedConfig,
    ActorCriticDecoupledConfig,
    PerceptionActorConfig,
    PerceptionCriticConfig
)
from jaxinn.agent.registry import Registrable
from jaxinn.agent.models import Actor, Critic
from jaxinn.agent.models.perception import Encoder
from jaxinn.agent.models.utils import apply_init


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


T = TypeVar("T", bound=eqx.Module)


class Ensemble(eqx.Module, Generic[T]):
    nets: T

    def __init__(self, model_cls: type[T], num_ensembles: int, config: Any, *, key: PRNGKeyArray):
        keys = jax.random.split(key, num_ensembles)

        def make_model(k):
            return model_cls(config, key=k)

        self.nets = eqx.filter_vmap(make_model)(keys)

    def __call__(self, *args, **kwargs) -> Any:
        @eqx.filter_vmap(in_axes=(0, None, None))
        def forward(net, a, kw):
            return net(*a, **kw)

        return forward(self.nets, args, kwargs)


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
        actor = apply_init(     # TODO: pass into weight init options and add it to configs
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
