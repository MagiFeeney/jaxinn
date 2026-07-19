from dataclasses import dataclass, field

from jaxinn.configs.base import Base
from .base import AgentConfig
from .actor_critic import PerceptionActorConfig, PerceptionCriticConfig
from ..model import ActorOptimizer, CriticOptimizer


@dataclass
class SACPerceptionActorConfig(PerceptionActorConfig):
    optimizer: ActorOptimizer = field(default_factory=lambda: ActorOptimizer(lr=3e-4, max_norm=None, eps=1e-8))


@dataclass
class SACPerceptionCriticConfig(PerceptionCriticConfig):
    optimizer: CriticOptimizer = field(default_factory=lambda: CriticOptimizer(lr=3e-4, max_norm=None, eps=1e-8))


# SAC
@dataclass
class SACOptimization(Base):
    discount_factor: float = 0.99
    alpha: float = 0.2
    batch_size: int = 256
    tau: float = 0.005
    target_update_interval: int = 1


@dataclass
class SACAgentConfig(AgentConfig):
    optimization: SACOptimization = field(default_factory=SACOptimization)

    actor: SACPerceptionActorConfig = field(default_factory=SACPerceptionActorConfig)
    critic: SACPerceptionCriticConfig = field(default_factory=SACPerceptionCriticConfig)
