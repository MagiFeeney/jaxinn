from copy import deepcopy
from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, Union, ClassVar

from jaxtyping import PyTree

import jaxinn.envs.spaces as jaxinn_spaces

from .base import (
    Base,
    Resolvable,
    StaticShared,
    ConfigNamespace,
    DictConfig,
    TupleConfig,
    HierarchicalConfig,
)
from .head import (
    HeadUnion,
    ContinuousHeadUnion,
    IsotropicNormalHeadConfig,
    TanhNormalHeadConfig,
    NormalHeadConfig,
    CategoricalHeadConfig,
    OneHotCategoricalHeadConfig,
    MultiCategoricalHeadConfig
)


@dataclass
class Model(Base):
    def __call__(self):
        models = {k: v for k, v in vars(self).items() if isinstance(v, Model)}
        if len(models) == 0:
            return super().__call__()
        return ConfigNamespace(**models) # Dot notation compatibility


@dataclass
class ModelShared(Model, StaticShared):
    """Shared parameters across different models."""
    belief_size: int = 200
    state_size: Union[int, Tuple[int, ...]] = 30


class Domain(str, Enum):
    STATE = "state"
    PIXEL = "pixel"


@dataclass
class PerceptionShared(Resolvable, Model):
    """Shared parameters across perception modules."""
    DOMAIN: ClassVar[Domain]

    obs_shape: Optional[PyTree[Tuple[int, ...]]] = field(default=None, init=False)
    activation_function: str = "elu"

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
    embedding_size: Optional[int] = None

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
class LearningRateScheduler(Base):
    kwargs: Dict[str, Union[int, float, bool]] = field(default_factory=dict)

    def __call__(self) -> dict:
        return self.kwargs


@dataclass
class OptimizerShared(Base):
    """Shared parameters across optimizers."""
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-7


# World Model
## For pixel-based tasks
@dataclass
class CNNEncoderConfig(EncoderConfig):
    DOMAIN: ClassVar[Domain] = Domain.PIXEL
    embedding_size: int = 1024
    dtype: str = "bfloat16"


@dataclass
class CNNDecoderConfig(DecoderConfig):
    DOMAIN: ClassVar[Domain] = Domain.PIXEL
    dtype: str = "bfloat16"


## For state-based tasks
@dataclass
class LinearEncoderConfig(EncoderConfig):
    DOMAIN: ClassVar[Domain] = Domain.STATE
    hidden_size: Optional[list[int]] = None
    embedding_size: Optional[int] = None


@dataclass
class LinearDecoderConfig(DecoderConfig):
    DOMAIN: ClassVar[Domain] = Domain.STATE
    hidden_size: list[int] = field(default_factory=lambda: [300, 300])


EncoderUnion = Union[CNNEncoderConfig, LinearEncoderConfig] # TODO: automate this
DecoderUnion = Union[CNNDecoderConfig, LinearDecoderConfig]


@dataclass
class RepresentationConfig(Resolvable, ModelShared):
    embedding_size: Optional[int] = field(default=None, init=False)
    hidden_size: list[int] = field(default_factory=lambda: [200])
    activation_function: str = "elu"

    head: HeadUnion = field(default_factory=NormalHeadConfig)

    def _resolve(self, ctx: dict) -> None:
        if "embedding_size" not in ctx:
            return
        self.embedding_size = ctx["embedding_size"]


@dataclass
class TransitionConfig(Resolvable, ModelShared):
    hidden_size: int = 200
    action_shape: Optional[PyTree[Tuple[int, ...]]] = field(default=None, init=False)
    activation_function: str = "elu"

    head: HeadUnion = field(default_factory=NormalHeadConfig)

    def _resolve(self, ctx: dict) -> None:
        self.action_shape = ctx["action_space"].shape


@dataclass
class RewardConfig(ModelShared):
    hidden_size: list[int] = field(default_factory=lambda: [300, 300, 300])
    action_shape: Optional[PyTree[Tuple[int, ...]]] = field(default=None, init=False)
    use_action: bool = field(default=False, metadata={"transient": True}) # will be discarded after resolve
    activation_function: str = "elu"

    head: HeadUnion = field(default_factory=IsotropicNormalHeadConfig)

    def _resolve(self, ctx: dict) -> None:
        if self.use_action:
            self.action_shape = ctx["action_space"].shape
        else:
            self.action_shape = None


@dataclass
class WorldOptimizer(OptimizerShared):
    """Optimizer for world model."""
    lr: float = 6e-4
    max_norm: float = 100


@dataclass
class WorldConfig(Resolvable, Model):
    encoder: Optional[PyTree[EncoderUnion]] = field(default=None, init=False)
    decoder: Optional[PyTree[DecoderUnion]] = field(default=None, init=False)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    optimizer: WorldOptimizer = field(default_factory=WorldOptimizer)

    pixel_encoder: PixelEncoderUnion = field(default_factory=CNNEncoderConfig)
    pixel_decoder: PixelEncoderUnion = field(default_factory=CNNDecoderConfig)
    decoder_free: bool = field(default=False, metadata={"transient": True})

    def _resolve(self, ctx: dict) -> None:
        observation_space = ctx["observation_space"]

        def _resolve_encoder_decoder(space):
            if isinstance(space, jaxinn_spaces.Dict):
                return {k: _resolve_encoder_decoder(s) for k, s in space.spaces.items()}
            elif isinstance(space, jaxinn_spaces.Tuple):
                return tuple(_resolve_encoder_decoder(s) for s in space.spaces)
            else:
                if len(space.shape) > 1:
                    return (deepcopy(self.pixel_encoder), deepcopy(self.pixel_decoder))
                else:
                    return (LinearEncoderConfig(), LinearDecoderConfig())

        resolved_encoder, resolved_decoder = _resolve_encoder_decoder(observation_space)

        if isinstance(resolved_encoder, dict):
            if isinstance(observation_space, jaxinn_spaces.Hierarchical):
                self.encoder = HierarchicalConfig(resolved_encoder)
                self.decoder = HierarchicalConfig(resolved_decoder)
            else:
                self.encoder = DictConfig(resolved_encoder)
                self.decoder = DictConfig(resolved_decoder)
        elif isinstance(resolved_encoder, tuple):
            self.encoder = TupleConfig(resolved_encoder)
            self.decoder = TupleConfig(resolved_decoder)
        else:
            self.encoder = resolved_encoder
            self.decoder = resolved_decoder

        if self.decoder_free:
            resolved_decoder = None


# Actor
@dataclass
class ActorOptimizer(OptimizerShared):
    """Optimizer for actor."""
    lr: float = 8e-5
    max_norm: int = 100


@dataclass
class ActorConfig(Resolvable, ModelShared):
    hidden_size: list[int] = field(default_factory=lambda: [300, 300, 300])
    activation_function: str = "elu"
    action_size: Optional[PyTree[int]] = field(default=None, init=False) # Pass from the env params

    head: Optional[PyTree[HeadUnion]] = field(default=None, init=False) # TODO: add head_overrides
    optimizer: ActorOptimizer = field(default_factory=ActorOptimizer)

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
                self.head = HierarchicalConfig(resolved_head)
            else:
                self.head = DictConfig(resolved_head)
        elif isinstance(resolved_head, tuple):
            self.head = TupleConfig(resolved_head)
        else:
            self.head = resolved_head


# Critic
@dataclass
class CriticOptimizer(OptimizerShared):
    """Optimizer for critic."""
    lr: float = 8e-5
    max_norm: int = 100


@dataclass
class CriticConfig(Resolvable, ModelShared):
    hidden_size: list[int] = field(default_factory=lambda: [300, 300, 300])
    action_shape: Optional[PyTree[Tuple[int, ...]]] = field(default=None, init=False)
    use_action: bool = field(default=False, metadata={"transient": True}) # will be discarded after resolve
    activation_function: str = "elu"

    head: HeadUnion = field(default_factory=IsotropicNormalHeadConfig)
    optimizer: CriticOptimizer = field(default_factory=CriticOptimizer)

    def _resolve(self, ctx: dict) -> None:
        if self.use_action:
            self.action_shape = ctx["action_space"].shape
        else:
            self.action_shape = None
