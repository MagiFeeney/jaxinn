from dataclasses import dataclass, field

from jaxinn.configs.base import Base
from .base import AgentConfig
from .actor_critic import PerceptionActorConfig, PerceptionCriticConfig
from ..model import Optimizer, LearnerConfig
from ..initializer import Initializer, LecunNormalConfig, ConstantConfig


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

    actor: LearnerConfig[PerceptionActorConfig] = field(
        default_factory=lambda: LearnerConfig(
            model=PerceptionActorConfig(
                initializer=Initializer(
                    weight_init=LecunNormalConfig(),
                    bias_init=ConstantConfig(value=0.0),
                    fused=True
                )
            ),
            optimizer=Optimizer()
        )
    )
    critic: LearnerConfig[PerceptionCriticConfig] = field(
        default_factory=lambda: LearnerConfig(
            model=PerceptionCriticConfig(
                initializer=Initializer(
                    weight_init=LecunNormalConfig(),
                    bias_init=ConstantConfig(value=0.0),
                    fused=True
                )
            ),
            optimizer=Optimizer()
        )
    )
