from dataclasses import dataclass, field

from jaxinn.configs.base import Base
from .base import AgentConfig
from .actor_critic import PerceptionActorConfig, PerceptionCriticConfig
from ..model import ActorOptimizer, CriticOptimizer


@dataclass
class SACPerceptionActorConfig(PerceptionActorConfig):
    optimizer: ActorOptimizer = field(default_factory=ActorOptimizer(max_norm=None))


@dataclass
class SACPerceptionCriticConfig(PerceptionCriticConfig):
    optimizer: CriticOptimizer = field(default_factory=CriticOptimizer(max_norm=None))


# SAC
@dataclass
class SACOptimization(Base):
    discount_factor: float = 0.99
    alpha: float = 0.2
    tau: float = 0.005
    target_update_interval: int = 2


@dataclass
class SACAgentConfig(AgentConfig):
    optimization: SACOptimization = field(default_factory=SACOptimization)

    actor: SACPerceptionActorConfig = field(default_factory=SACPerceptionActorConfig)
    critic: SACPerceptionCriticConfig = field(default_factory=SACPerceptionCriticConfig)
