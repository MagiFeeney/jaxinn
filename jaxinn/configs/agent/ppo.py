from typing import Union
from dataclasses import dataclass, field

from jaxinn.configs.base import Base, Resolvable
from jaxinn.configs.model import Model, ActorConfig, CriticConfig, EncoderUnion, LinearEncoderConfig, OptimizerShared

from .base import AgentConfig


def _resolve_input_size(ctx: dict, *modules) -> None:
    if "embedding_size" not in ctx:
        return

    obs_shape = ctx.get("obs_shape", ())
    embedding_size = ctx.get("embedding_size", None)

    if embedding_size is not None:
        state_size = embedding_size
    elif len(obs_shape) == 1:
        state_size = obs_shape[0]
    else:
        raise ValueError(
            f"Cannot determine state_size from obs_shape {obs_shape} "
            f"and embedding_size {embedding_size}."
        )

    for module in modules:
        if hasattr(module, "state_size"):
            module.state_size = state_size
        if hasattr(module, "belief_size"):
            module.belief_size = 0


@dataclass
class PerceptionActorConfig(Resolvable, Model):
    encoder: EncoderUnion = field(default_factory=LinearEncoderConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)

    def _resolve(self, ctx: dict) -> None:
        _resolve_input_size(ctx, self.actor)


@dataclass
class PerceptionCriticConfig(Resolvable, Model):
    encoder: EncoderUnion = field(default_factory=LinearEncoderConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)

    def _resolve(self, ctx: dict) -> None:
        _resolve_input_size(ctx, self.critic)


@dataclass
class ActorCriticOptimizer(OptimizerShared):
    """Optimizer for actor-critic."""
    lr: float = 3e-4
    max_norm: float = 0.5


@dataclass
class ActorCriticDecoupledConfig(Resolvable, Model):
    perception_actor: PerceptionActorConfig = field(default_factory=PerceptionActorConfig)
    perception_critic: PerceptionCriticConfig = field(default_factory=PerceptionCriticConfig)

    optimizer: ActorCriticOptimizer = field(default_factory=ActorCriticOptimizer)


@dataclass
class ActorCriticSharedConfig(Resolvable, Model):
    encoder: EncoderUnion = field(default_factory=LinearEncoderConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)

    optimizer: ActorCriticOptimizer = field(default_factory=ActorCriticOptimizer)

    def _resolve(self, ctx: dict) -> None:
        _resolve_input_size(ctx, self.actor, self.critic)


ActorCriticUnion = Union[ActorCriticDecoupledConfig, ActorCriticSharedConfig]


# PPO
@dataclass
class PPOOptimization(Base):
    clip_param: float = 0.2
    use_clipped_critic_loss: bool = True
    num_mini_batch: int = 8
    discount_factor: float = 0.99
    uae_lambda: float = 0.95


@dataclass
class PPOAgentConfig(AgentConfig):
    actor_critic: ActorCriticUnion = field(default_factory=ActorCriticDecoupledConfig)

    optimization: PPOOptimization = field(default_factory=PPOOptimization)
