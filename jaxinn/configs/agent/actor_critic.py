from typing import Union
from dataclasses import dataclass, field

from jaxinn.configs.base import Resolvable
from jaxinn.configs.model import Model, ActorConfig, CriticConfig, EncoderUnion, LinearEncoderConfig


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
class ActorCriticDecoupledConfig(Resolvable, Model):
    perception_actor: PerceptionActorConfig = field(default_factory=PerceptionActorConfig)
    perception_critic: PerceptionCriticConfig = field(default_factory=PerceptionCriticConfig)


@dataclass
class ActorCriticSharedConfig(Resolvable, Model):
    encoder: EncoderUnion = field(default_factory=LinearEncoderConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)

    def _resolve(self, ctx: dict) -> None:
        _resolve_input_size(ctx, self.actor, self.critic)


ActorCriticUnion = Union[ActorCriticDecoupledConfig, ActorCriticSharedConfig]
