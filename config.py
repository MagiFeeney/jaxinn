from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Optional


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
class OptimizerShared(Base):
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class WorldOptimizer(OptimizerShared):
    lr: float = 1e-3


@dataclass
class ModelShared(Base):
    belief_size: int
    state_size: int


@dataclass
class World(ModelShared):
    hidden_size: int
    optimizer: WorldOptimizer = field(default_factory=WorldOptimizer)
    activation_function: str = "elu"
    action_size: Optional[int] = None # Pass from the env params
    min_std: float = 0.0
    head_type: str = "Isotropic Normal"


@dataclass
class Actor(ModelShared):
    pass


@dataclass
class Critic(ModelShared):
    pass


@dataclass
class Memory(ModelShared):
    pass


@dataclass
class Env(Base):
    pass


@dataclass
class Config(Base):
    world: World = field(default_factory=World)
    actor: Actor = field(default_factory=Actor)
    critic: Critic = field(default_factory=Critic)
    memory: Memory = field(default_factory=Memory)
    env: Env = field(default_factory=Env)
