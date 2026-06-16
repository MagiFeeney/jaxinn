from dataclasses import dataclass, field

from jaxinn.configs.base import Base
from jaxinn.configs.model import Actor, Critic, World

from .base import Agent


# Dreamer
@dataclass
class DreamerOptimization(Base):
    planning_horizon: int = 15
    discount_factor: float = 0.99
    uae_lambda: float = 0.95
    batch_size: int = 50
    chunk_size: int = 50
    free_nats: float = 3.0
    kl_average: bool = False
    kl_balance: float = 0.0


@dataclass
class DreamerAgent(Agent):
    actor: Actor = field(default_factory=Actor)
    critic: Critic = field(default_factory=Critic)
    world: World = field(default_factory=World)

    optimization: DreamerOptimization = field(default_factory=DreamerOptimization)
    random_init: bool = False   # Whether to initialize the state by following a simple distribution


@dataclass
class DreamerV2Optimization(DreamerOptimization):
    pg_mix: float = 1.0


@dataclass
class DreamerV2Agent(DreamerAgent):
    optimization: DreamerV2Optimization = field(default_factory=DreamerV2Optimization)
