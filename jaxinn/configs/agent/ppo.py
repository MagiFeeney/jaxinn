from typing import Union
from dataclasses import dataclass, field

from jaxinn.configs.base import Base, Model, Resolvable, Actor, Critic, EncoderUnion, LinearEncoder, OptimizerShared

from .base import Agent


@dataclass
class PerceptionActor(Resolvable, Model):
    encoder: EncoderUnion = field(default_factory=LinearEncoder)
    actor: Actor = field(default_factory=Actor)


@dataclass
class PerceptionCritic(Resolvable, Model):
    encoder: EncoderUnion = field(default_factory=LinearEncoder)
    critic: Critic = field(default_factory=Critic)


@dataclass
class ActorCriticOptimizer(OptimizerShared):
    """Optimizer for actor-critic."""
    lr: float = 3e-4
    max_norm: float = 0.5


@dataclass
class ActorCriticDecoupled(Resolvable, Model):
    perception_actor: PerceptionActor = field(default_factory=PerceptionActor)
    perception_critic: PerceptionCritic = field(default_factory=PerceptionCritic)

    optimizer: ActorCriticOptimizer = field(default_factory=ActorCriticOptimizer)

    @property
    def obs_shape(self):
        return self.perception_actor.encoder.shape

    @property
    def action_size(self):
        return self.perception_actor.actor.action_size


@dataclass
class ActorCriticShared(Resolvable, Model):
    encoder: EncoderUnion = field(default_factory=LinearEncoder)
    actor: Actor = field(default_factory=Actor)
    critic: Critic = field(default_factory=Critic)

    optimizer: ActorCriticOptimizer = field(default_factory=ActorCriticOptimizer)

    @property
    def obs_shape(self):
        return self.encoder.shape

    @property
    def action_size(self):
        return self.actor.action_size


ActorCriticUnion = Union[ActorCriticDecoupled, ActorCriticShared]


# PPO
@dataclass
class PPOOptimization(Base):
    clip_param: float = 0.2
    use_clipped_critic_loss: bool = True
    num_mini_batch: int = 8
    discount_factor: float = 0.99
    uae_lambda: float = 0.95


@dataclass
class PPOAgent(Agent):
    actor_critic: ActorCriticUnion = field(default_factory=ActorCriticDecoupled)

    optimization: PPOOptimization = field(default_factory=PPOOptimization)
