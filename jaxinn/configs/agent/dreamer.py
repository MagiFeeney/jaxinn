from dataclasses import dataclass, field

from jaxinn.configs.base import Base
from jaxinn.configs.model import (
    ActorConfig,
    CriticConfig,
    WorldConfig,
    Optimizer,
    LearnerConfig,
)
from jaxinn.configs.head import IsotropicNormalHeadConfig

from .base import AgentConfig
from .memory import MemoryUnion, EpisodicMemoryConfig


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
class DreamerAgentConfig(AgentConfig):
    actor: LearnerConfig[ActorConfig] = field(
        default_factory=lambda: LearnerConfig(
            model=ActorConfig(),
            optimizer=Optimizer(lr=8e-5, max_norm=100, eps=1e-7)
        )
    )
    critic: LearnerConfig[CriticConfig] = field(
        default_factory=lambda: LearnerConfig(
            model=CriticConfig(head=IsotropicNormalHeadConfig()),
            optimizer=Optimizer(lr=8e-5, max_norm=100, eps=1e-7)
        )
    )
    world: LearnerConfig[WorldConfig] = field(
        default_factory=lambda: LearnerConfig(
            model=WorldConfig(),
            optimizer=Optimizer(lr=6e-4, max_norm=100, eps=1e-7)
        )
    )

    optimization: DreamerOptimization = field(default_factory=DreamerOptimization)
    random_init: bool = False   # Whether to initialize the state by following a simple distribution


@dataclass
class DreamerV2Optimization(DreamerOptimization):
    pg_mix: float = 1.0


@dataclass
class DreamerV2AgentConfig(DreamerAgentConfig):
    memory: MemoryUnion = field(default_factory=EpisodicMemoryConfig)

    optimization: DreamerV2Optimization = field(default_factory=DreamerV2Optimization)
