from enum import Enum
from dataclasses import dataclass, field, InitVar
from typing import Tuple, Optional, Union, ClassVar

from jaxtyping import PyTree, DTypeLike

from .base import Base, Resolvable, StaticShared, ConfigNamespace
from .head import HeadUnion, IsotropicNormalHeadConfig, TanhNormalHeadConfig, NormalHeadConfig, CategoricalHeadConfig, OneHotCategoricalHeadConfig


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
        obs_shape = ctx.get("obs_shape", None)
        self.obs_shape = obs_shape

        env_domain = Domain.PIXEL if len(obs_shape) > 1 else Domain.STATE

        if self.DOMAIN != env_domain:
            raise ValueError(
                f"Architecture Mismatch: Environment provides {env_domain.value} "
                f"observations {obs_shape}, but {type(self).__name__} "
                f"expects {self.DOMAIN.value} inputs."
            )

        # Propagate downstream for representation to consume
        embedding_size = getattr(self, "embedding_size", None)
        if "embedding_size" not in ctx:
            ctx["embedding_size"] = embedding_size


@dataclass
class OptimizerShared(Base):
    """Shared parameters across optimizers."""
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-7


# World Model
## For pixel-based tasks
@dataclass
class CNNEncoderConfig(PerceptionShared):
    DOMAIN: ClassVar[Domain] = Domain.PIXEL
    embedding_size: int = 1024
    dtype: str = "bfloat16"


@dataclass
class CNNDecoderConfig(PerceptionShared, ModelShared):
    DOMAIN: ClassVar[Domain] = Domain.PIXEL
    dtype: str = "bfloat16"


## For state-based tasks
@dataclass
class LinearEncoderConfig(PerceptionShared):
    DOMAIN: ClassVar[Domain] = Domain.STATE
    hidden_size: Optional[list[int]] = None
    embedding_size: Optional[int] = None


@dataclass
class LinearDecoderConfig(PerceptionShared, ModelShared):
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

        obs_shape = ctx["obs_shape"]
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
    hidden_size: int = 200
    action_shape: Optional[PyTree[Tuple[int, ...]]] = field(default=None, init=False)
    activation_function: str = "elu"

    head: HeadUnion = field(default_factory=NormalHeadConfig)

    def _resolve(self, ctx: dict) -> None:
        self.action_shape = ctx["action_shape"]


@dataclass
class RewardConfig(ModelShared):
    hidden_size: list[int] = field(default_factory=lambda: [300, 300, 300])
    action_shape: Optional[PyTree[Tuple[int, ...]]] = field(default=None, init=False)
    use_action: InitVar[bool] = False # will be discarded after resolve
    activation_function: str = "elu"

    head: HeadUnion = field(default_factory=IsotropicNormalHeadConfig)

    def _resolve(self, ctx: dict) -> None:
        if self.use_action:
            self.action_shape = ctx["action_shape"]


@dataclass
class WorldOptimizer(OptimizerShared):
    """Optimizer for world model."""
    lr: float = 6e-4
    max_norm: float = 100


@dataclass
class WorldConfig(Resolvable, Model):
    encoder: EncoderUnion = field(default_factory=CNNEncoderConfig)
    decoder: Optional[DecoderUnion] = field(default_factory=CNNDecoderConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    optimizer: WorldOptimizer = field(default_factory=WorldOptimizer)


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

    head: HeadUnion = field(default_factory=TanhNormalHeadConfig)
    optimizer: ActorOptimizer = field(default_factory=ActorOptimizer)

    def _resolve(self, ctx: dict) -> None:
        if ctx["is_action_space_discrete"] ^ isinstance(self.head, (CategoricalHeadConfig, OneHotCategoricalHeadConfig)):
            raise ValueError(f"Inconsistent actor head: action space is discrete={ctx['is_action_space_discrete']}, but received head type {type(self.head).__name__}.")

        if isinstance(self.head, OneHotCategoricalHeadConfig) != ctx["use_one_hot_action"]:
            raise ValueError(
                f"Mismatch: Head is {type(self.head).__name__}, but env use_one_hot_action is {ctx['use_one_hot_action']}."
            )

        self.action_size = ctx["action_size"]


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
    use_action: InitVar[bool] = False # will be discarded after resolve
    activation_function: str = "elu"

    head: HeadUnion = field(default_factory=IsotropicNormalHeadConfig)
    optimizer: CriticOptimizer = field(default_factory=CriticOptimizer)

    def _resolve(self, ctx: dict) -> None:
        if self.use_action:
            self.action_shape = ctx["action_shape"]
