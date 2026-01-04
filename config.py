from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Optional
from types import SimpleNamespace


class ConfigNamespace(SimpleNamespace):
    def keys(self):
        return vars(self).keys()

    def items(self):
        return vars(self).items()

    def __contains__(self, key):
        return key in vars(self)

    def __iter__(self):
        return iter(vars(self))

    def __getitem__(self, key):
        return vars(self)[key]

    def __repr__(self):
        items = [f"{k}={v}" for k, v in self.items()]
        return f"Namespace({', '.join(items)})"


@dataclass
class Base:
    """
    Nested configuration management.

    If there is sub-config, return the arguments at the current level excluding those from sub-configs.
    """
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)     # Preserve hierarchy

    def __call__(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not is_dataclass(v)} # Sub-node is considered as next phase

    def update(self, updates: dict[str, Any] = None, **kwargs) -> None:
        """Update or add config attributes. Allows adding new attributes."""
        if updates is None:
            updates = kwargs
        else:
            updates.update(kwargs)

        for key, value in updates.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if is_dataclass(attr) and isinstance(value, dict):
                    attr.update(value)
                else:
                    setattr(self, key, value)
            else:
                # Add new attribute that doesn't exist
                setattr(self, key, value)


@dataclass
class Model(Base):
    def __call__(self):
	models = {k: v for k, v in vars(self).items() if isinstance(v, Model)}
        if len(models) == 0:
            return super().__call__()
        return ConfigNamespace(**models) # Dot notation compatibility


@dataclass
class ModelShared(Model):
    """Shared parameters across different models."""
    belief_size: int
    state_size: int


@dataclass
class OptimizerShared(Base):
    """Shared parameters across optimizers."""
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8           # TODO: check default value of eps of adam optimizer


# World Model
@dataclass
class Encoder(ModelShared):
    pass


@dataclass
class Decoder(ModelShared):
    pass


@dataclass
class Perception(Model):
    encoder: Encoder = field(default_factory=Encoder)
    decoder: Decoder = field(default_factory=Decoder)


@dataclass
class Representation(ModelShared):
    pass


@dataclass
class Transition(ModelShared):
    pass


@dataclass
class Reward(ModelShared):
    pass


@dataclass
class WorldOptimizer(OptimizerShared):
    """Optimizer exclusive for world model."""
    lr: float = 6e-4            # TODO: add Optimizer(Base) and extra here


@dataclass
class World(Model):
    perception: Perception = field(default_factory=Perception)
    representation: Representation = field(default_factory=Representation)
    transition: Transition = field(default_factory=Transition)
    reward: Reward = field(default_factory=Reward)
    optimizer: WorldOptimizer = field(default_factory=WorldOptimizer)


# Actor
@dataclass
class ActorOptimizer(OptimizerShared):
    """Optimizer exclusive for actor."""
    lr: float = 8e-5


@dataclass
class Actor(ModelShared):
    hidden_size: int
    activation_function: str = "elu"
    action_size: Optional[int] = None # Pass from the env params
    min_std: float = 0.0
    head_type: str = "Isotropic Normal"

    optimizer: ActorOptimizer = field(default_factory=ActorOptimizer)


# Critic
@dataclass
class CriticOptimizer(OptimizerShared):
    """Optimizer exclusive for actor."""
    lr: float = 8e-5


@dataclass
class Critic(ModelShared):
    optimizer: CriticOptimizer = field(default_factory=CriticOptimizer)


# Memory
@dataclass
class Memory(ModelShared):
    pass


# Environment
@dataclass
class Env(Base):
    pass


# Console
@dataclass
class Config(Base):
    world: World = field(default_factory=World)
    actor: Actor = field(default_factory=Actor)
    critic: Critic = field(default_factory=Critic)
    memory: Memory = field(default_factory=Memory)
    env: Env = field(default_factory=Env)
