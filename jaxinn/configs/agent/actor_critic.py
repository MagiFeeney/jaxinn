from typing import Union, Optional
from dataclasses import dataclass, field

from jaxinn.configs.base import Resolvable
from jaxinn.configs.model import Model, ActorConfig, CriticConfig, EncoderUnion, LinearEncoderConfig, OptimizerShared

from ..model import LearningRateScheduler


def _resolve_input_size(ctx: dict, *modules) -> None:
    if "embedding_size" not in ctx:
        return

    obs_shape = ctx["observation_space"].shape
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
class ActorCriticOptimizer(Resolvable, OptimizerShared):
    """Optimizer for actor-critic."""
    lr: float = 3e-4
    max_norm: Optional[float] = 0.5
    use_lr_scheduler: bool = field(default=True, metadata={"transient": True})
    lr_scheduler: Optional[LearningRateScheduler] = None


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
