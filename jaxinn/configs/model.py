from copy import deepcopy
from enum import StrEnum
from dataclasses import dataclass, field
from typing import ClassVar, Generic, TypeVar
from collections.abc import Sequence

from jaxtyping import PyTree

import jaxinn.envs.spaces as jaxinn_spaces

from .base import Base, Resolvable, StaticShared, ConfigNamespace
from .head import (
    HeadUnion,
    ContinuousHeadUnion,
    DictHeadConfig,
    TupleHeadConfig,
    HierarchicalHeadConfig,
    DeterministicHeadConfig,
    IsotropicNormalHeadConfig,
    TanhNormalHeadConfig,
    NormalHeadConfig,
    BernoulliHeadConfig,
    CategoricalHeadConfig,
    OneHotCategoricalHeadConfig,
    MultiCategoricalHeadConfig
)
from .scheduler import LearningRateSchedulerUnion
from .initializer import Initializer


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
class Model(Base):
    initializer: Initializer = field(default_factory=Initializer)

    def __call__(self):
        models = {k: v for k, v in vars(self).items() if isinstance(v, Model)}
        if len(models) == 0:
            return super().__call__()
        return ConfigNamespace(**models) # Dot notation compatibility


@dataclass
class ModelShared(Model, StaticShared):
    """Shared parameters across different models."""
    belief_size: int = 200
    state_size: int | tuple[int, ...] = 30


class Domain(StrEnum):
    STATE = "state"
    PIXEL = "pixel"


@dataclass
class PerceptionShared(Resolvable, Model):
    """Shared parameters across perception modules."""
    DOMAIN: ClassVar[Domain]

    obs_shape: PyTree[tuple[int, ...]] | None = field(default=None, init=False)
    activation_function: str = "relu"

    def _resolve(self, ctx: dict) -> None:
        obs_shape = ctx["observation_space"].shape

        env_domain = Domain.PIXEL if len(obs_shape) > 1 else Domain.STATE

        if self.DOMAIN != env_domain:
            raise ValueError(
                f"Architecture Mismatch: Environment provides {env_domain.value} "
                f"observations {obs_shape}, but {type(self).__name__} "
                f"expects {self.DOMAIN.value} inputs."
            )

        self.obs_shape = obs_shape


@dataclass
class EncoderConfig(PerceptionShared):
    embedding_size: int | None = None

    def _resolve(self, ctx: dict) -> None:
        super()._resolve(ctx)

        # Propagate downstream for representation to consume
        embedding_size = getattr(self, "embedding_size", None)
        if "embedding_size" not in ctx:
            ctx["embedding_size"] = embedding_size


@dataclass
class DecoderConfig(PerceptionShared, ModelShared):
    pass


@dataclass
class Optimizer(Base):
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    lr: float = 3e-4
    weight_decay: float = 0.0
    max_norm: float | None = None
    lr_scheduler: LearningRateSchedulerUnion | None = None


T = TypeVar('T')


@dataclass
class LearnerConfig(Resolvable, Base, Generic[T]):
    model: T
    optimizer: Optimizer


# World Model
## For pixel-based tasks
@dataclass
class CNNEncoderConfig(EncoderConfig):
    DOMAIN: ClassVar[Domain] = Domain.PIXEL
    embedding_size: int = 1024
    num_layers: int | None = 4
    kernel_size: int | Sequence[int] = 4
    depth: int | Sequence[int] = 32
    stride: int | Sequence[int] = 2
    dtype: str = "bfloat16"


@dataclass
class CNNDecoderConfig(DecoderConfig):
    DOMAIN: ClassVar[Domain] = Domain.PIXEL
    num_layers: int | None = 4
    kernel_size: int | Sequence[int] = 4
    depth: int | Sequence[int] = 32
    stride: int | Sequence[int] = 2
    dtype: str = "bfloat16"


## For state-based tasks
@dataclass
class LinearEncoderConfig(EncoderConfig):
    DOMAIN: ClassVar[Domain] = Domain.STATE
    hidden_size: list[int] | None = None
    embedding_size: int | None = None


@dataclass
class LinearDecoderConfig(DecoderConfig):
    DOMAIN: ClassVar[Domain] = Domain.STATE
    hidden_size: list[int] = field(default_factory=lambda: [300, 300])


EncoderUnion = CNNEncoderConfig | LinearEncoderConfig # TODO: automate this
DecoderUnion = CNNDecoderConfig | LinearDecoderConfig


@dataclass
class RepresentationConfig(Resolvable, ModelShared):
    embedding_size: int | None = field(default=None, init=False)
    hidden_size: list[int] = field(default_factory=lambda: [200])
    activation_function: str = "elu"

    head: HeadUnion = field(default_factory=NormalHeadConfig)

    def _resolve(self, ctx: dict) -> None:
        if "embedding_size" not in ctx:
            return

        obs_shape = ctx["observation_space"].shape
        embedding_size = ctx["embedding_size"]

        if embedding_size is not None:
            self.embedding_size = embedding_size
        elif len(obs_shape) == 1:
            self.embedding_size = obs_shape[0]
        else:
            raise ValueError(
                f"Cannot infer embedding_size for 2D+ obs_shape {obs_shape} "
                "without an explicit size provided by the encoder."
            )


@dataclass
class TransitionConfig(Resolvable, ModelShared):
    hidden_size: list[int] = field(default_factory=lambda: [200])
    action_shape: PyTree[tuple[int, ...]] | None = field(default=None, init=False)
    activation_function: str = "elu"

    head: HeadUnion = field(default_factory=NormalHeadConfig)

    def _resolve(self, ctx: dict) -> None:
        self.action_shape = ctx["action_space"].shape


@dataclass
class RewardConfig(ModelShared):
    hidden_size: list[int] = field(default_factory=lambda: [300, 300, 300])
    action_shape: PyTree[tuple[int, ...]] | None = field(default=None, init=False)
    use_action: bool = field(default=False, metadata={"transient": True}) # will be discarded after resolve
    activation_function: str = "elu"

    head: HeadUnion = field(default_factory=IsotropicNormalHeadConfig)

    def _resolve(self, ctx: dict) -> None:
        if self.use_action:
            self.action_shape = ctx["action_space"].shape
        else:
            self.action_shape = None


@dataclass
class ContinuationConfig(ModelShared):
    hidden_size: list[int] = field(default_factory=lambda: [300, 300, 300])
    action_shape: PyTree[tuple[int, ...]] | None = field(default=None, init=False)
    use_action: bool = field(default=False, metadata={"transient": True}) # will be discarded after resolve
    activation_function: str = "elu"

    head: BernoulliHeadConfig | DeterministicHeadConfig = field(default_factory=BernoulliHeadConfig)

    def _resolve(self, ctx: dict) -> None:
        if self.use_action:
            self.action_shape = ctx["action_space"].shape
        else:
            self.action_shape = None


@dataclass
class WorldConfig(Resolvable, Model):
    encoder: EncoderUnion = field(default_factory=CNNEncoderConfig)
    decoder: DecoderUnion | None = field(default_factory=CNNDecoderConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    continuation: ContinuationConfig | None = None


@dataclass
class ActorConfig(Resolvable, ModelShared):
    hidden_size: list[int] = field(default_factory=lambda: [300, 300, 300])
    activation_function: str = "elu"
    action_size: PyTree[int] | None = field(default=None, init=False) # Pass from the env params

    head: PyTree[HeadUnion] | None = field(default=None, init=False) # TODO: add head_overrides

    # For building head; discard after use
    continuous_head: ContinuousHeadUnion = field(default_factory=TanhNormalHeadConfig)

    def _resolve(self, ctx: dict) -> None:
        action_space = ctx["action_space"]
        self.action_size = action_space.size

        def _resolve_head(space):
            if isinstance(space, jaxinn_spaces.Dict):
                return {k: _resolve_head(s) for k, s in space.spaces.items()}
            elif isinstance(space, jaxinn_spaces.Tuple):
                return tuple(_resolve_head(s) for s in space.spaces)
            elif isinstance(space, jaxinn_spaces.Box):
                return deepcopy(self.continuous_head)
            elif isinstance(space, jaxinn_spaces.OneHotDiscrete):
                return OneHotCategoricalHeadConfig()
            elif isinstance(space, jaxinn_spaces.MultiDiscrete):
                return MultiCategoricalHeadConfig(nvec=space.nvec)
            elif isinstance(space, jaxinn_spaces.Discrete):
                return CategoricalHeadConfig()
            else:
                raise TypeError(
                    f"Unsupported space type for creating head: '{type(space).__name__}'."
                )

        resolved_head = _resolve_head(action_space)
        if isinstance(resolved_head, dict):
            if isinstance(action_space, jaxinn_spaces.Hierarchical):
                self.head = HierarchicalHeadConfig(resolved_head)
            else:
                self.head = DictHeadConfig(resolved_head)
        elif isinstance(resolved_head, tuple):
            self.head = TupleHeadConfig(resolved_head)
        else:
            self.head = resolved_head


@dataclass
class PerceptionActorConfig(Resolvable, Model):
    encoder: EncoderUnion = field(default_factory=LinearEncoderConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)

    def _resolve(self, ctx: dict) -> None:
        _resolve_input_size(ctx, self.actor)


@dataclass
class CriticConfig(Resolvable, ModelShared):
    hidden_size: list[int] = field(default_factory=lambda: [300, 300, 300])
    action_shape: PyTree[tuple[int, ...]] | None = field(default=None, init=False)
    use_action: bool = field(default=False, metadata={"transient": True}) # will be discarded after resolve
    activation_function: str = "elu"

    head: HeadUnion = field(default_factory=DeterministicHeadConfig)

    def _resolve(self, ctx: dict) -> None:
        if self.use_action:
            self.action_shape = ctx["action_space"].shape
        else:
            self.action_shape = None


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


ActorCriticUnion = ActorCriticDecoupledConfig | ActorCriticSharedConfig
