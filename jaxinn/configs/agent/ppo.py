import math
from dataclasses import dataclass, field

from jaxinn.configs.base import Base, Resolvable

from .base import AgentConfig
from .actor_critic import ActorCriticUnion, ActorCriticSharedConfig
from ..model import Optimizer, LearnerConfig
from ..scheduler import LearningRateSchedulerUnion, StaircaseScheduleConfig


@dataclass
class PPOActorCriticOptimizer(Optimizer):
    lr: float = 3e-4
    max_norm: float = 0.5
    lr_scheduler: LearningRateSchedulerUnion | None = field(
        default_factory=lambda: StaircaseScheduleConfig(
            init_value=math.nan,
            num_iterations=-1,
            updates_per_iteration=-1
        )
    )

    def _resolve(self, ctx: dict) -> None:
        if self.lr_scheduler is None:
            return

        if hasattr(self.lr_scheduler, "init_value"):
            self.lr_scheduler.init_value = self.lr
        elif hasattr(self.lr_scheduler, "value"):
            self.lr_scheduler.value = self.lr

        if isinstance(self.lr_scheduler, StaircaseScheduleConfig):
            if self.lr_scheduler.num_iterations == -1:
                self.lr_scheduler.num_iterations = ctx["num_environment_steps"] // ctx["episode_length"]
            if self.lr_scheduler.updates_per_iteration == -1:
                self.lr_scheduler.updates_per_iteration = ctx["train_iterations"] * ctx["num_mini_batch"]


# PPO
@dataclass
class PPOOptimization(Resolvable, Base):
    clip_param: float = 0.2
    use_clipped_critic_loss: bool = True
    entropy_coef: float = 0.0
    num_mini_batch: int = 8
    discount_factor: float = 0.99
    uae_lambda: float = 0.95
    normalize_adv: bool = True

    def _resolve(self, ctx: dict) -> None:
        if "num_mini_batch" not in ctx:
            ctx["num_mini_batch"] = self.num_mini_batch


@dataclass
class PPOAgentConfig(AgentConfig):
    optimization: PPOOptimization = field(default_factory=PPOOptimization)

    actor_critic: LearnerConfig[ActorCriticUnion] = field(
        default_factory=lambda: LearnerConfig(
            model=ActorCriticSharedConfig(),
            optimizer=PPOActorCriticOptimizer()
        )
    )
