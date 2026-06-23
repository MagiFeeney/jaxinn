from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal, Union, ClassVar

from .base import Base, Resolvable, StaticShared, ConfigNamespace


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

    shape: Tuple[int, ...] = field(init=False)
    activation_function: str = "elu"

    def _resolve(self, ctx: dict) -> None:
        obs_shape = ctx["obs_shape"]
        self.shape = obs_shape

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
class CNNEncoder(PerceptionShared):
    DOMAIN: ClassVar[Domain] = Domain.PIXEL
    embedding_size: int = 1024
    dtype: str = "bfloat16"


@dataclass
class CNNDecoder(PerceptionShared, ModelShared):
    DOMAIN: ClassVar[Domain] = Domain.PIXEL
    dtype: str = "bfloat16"


## For state-based tasks
@dataclass
class LinearEncoder(PerceptionShared):
    DOMAIN: ClassVar[Domain] = Domain.STATE
    hidden_size: Optional[list[int]] = None
    embedding_size: Optional[int] = None


@dataclass
class LinearDecoder(PerceptionShared, ModelShared):
    DOMAIN: ClassVar[Domain] = Domain.STATE
    hidden_size: list[int] = field(default_factory=lambda: [300, 300])


EncoderUnion = Union[CNNEncoder, LinearEncoder] # TODO: automate this
DecoderUnion = Union[CNNDecoder, LinearDecoder]


@dataclass
class Perception(Resolvable, Model):
    encoder: EncoderUnion = field(default_factory=CNNEncoder)
    decoder: Optional[DecoderUnion] = field(default_factory=CNNDecoder)


@dataclass
class Representation(Resolvable, ModelShared):
    embedding_size: int = field(init=False)
    hidden_size: list[int] = field(default_factory=lambda: [200])
    activation_function: str = "elu"
    head_type: Literal['Normal', 'Categorical'] = "Normal"

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
class Transition(Resolvable, ModelShared):
    hidden_size: int = 200
    action_size: int = field(init=False)
    activation_function: str = "elu"
    head_type: Literal['Normal', 'Categorical'] = "Normal"

    def _resolve(self, ctx: dict) -> None:
        if ctx["is_action_space_discrete"] and not ctx["use_one_hot_action"]:
            self.action_size = 1
        else:
            self.action_size = ctx["action_size"]


@dataclass
class Reward(ModelShared):
    hidden_size: list[int] = field(default_factory=lambda: [300, 300, 300])
    action_size: Optional[int] = field(default=None, init=False)
    use_action: bool = False
    activation_function: str = "elu"
    head_type: str = "Isotropic Normal"
    min_std: float = 0.0

    def _resolve(self, ctx: dict) -> None:
        if self.use_action:
            self.action_size = ctx["action_size"]


@dataclass
class WorldOptimizer(OptimizerShared):
    """Optimizer for world model."""
    lr: float = 6e-4
    max_norm: float = 100


@dataclass
class World(Resolvable, Model):
    perception: Perception = field(default_factory=Perception)
    representation: Representation = field(default_factory=Representation)
    transition: Transition = field(default_factory=Transition)
    reward: Reward = field(default_factory=Reward)
    optimizer: WorldOptimizer = field(default_factory=WorldOptimizer)


# Actor
@dataclass
class ActorOptimizer(OptimizerShared):
    """Optimizer for actor."""
    lr: float = 8e-5
    max_norm: int = 100


@dataclass
class Actor(Resolvable, ModelShared):
    hidden_size: list[int] = field(default_factory=lambda: [300, 300, 300])
    activation_function: str = "elu"
    action_size: int = field(init=False) # Pass from the env params
    min_std: float = 0.0
    head_type: Literal['Tanh Normal', 'Beta', 'Categorical', 'OneHotCategorical'] = "Tanh Normal"

    optimizer: ActorOptimizer = field(default_factory=ActorOptimizer)

    def _resolve(self, ctx: dict) -> None:
        if ctx["is_action_space_discrete"] ^ (self.head_type in ("Categorical", "OneHotCategorical")):
            raise ValueError(f"Inconsistent actor head: action space is discrete={ctx['is_action_space_discrete']}, but received head type {self.head_type!r}.")
        self.action_size = ctx["action_size"]


# Critic
@dataclass
class CriticOptimizer(OptimizerShared):
    """Optimizer for critic."""
    lr: float = 8e-5
    max_norm: int = 100


@dataclass
class Critic(Resolvable, ModelShared):
    hidden_size: list[int] = field(default_factory=lambda: [300, 300, 300])
    action_size: Optional[int] = field(default=None, init=False)
    use_action: bool = False
    activation_function: str = "elu"
    min_std: float = 0.0
    head_type: Literal['Isotropic Normal', 'Normal'] = "Isotropic Normal"

    optimizer: CriticOptimizer = field(default_factory=CriticOptimizer)

    def _resolve(self, ctx: dict) -> None:
        if self.use_action:
            self.action_size = ctx["action_size"]
