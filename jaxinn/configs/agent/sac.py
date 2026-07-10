from dataclasses import dataclass, field

from jaxinn.configs.base import Base
from .base import AgentConfig


# SAC
@dataclass
class SACOptimization(Base):
    planning_horizon: int = 15
    discount_factor: float = 0.99
    uae_lambda: float = 0.95
    batch_size: int = 50
    chunk_size: int = 50
    free_nats: float = 3.0
    kl_average: bool = False
    kl_balance: float = 0.0


@dataclass
class SACAgentConfig(AgentConfig):
    optimization: SACOptimization = field(default_factory=SACOptimization)
