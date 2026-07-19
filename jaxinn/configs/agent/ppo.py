from dataclasses import dataclass, field

from jaxinn.configs.base import Base, Resolvable

from .base import AgentConfig
from .actor_critic import ActorCriticUnion, ActorCriticSharedConfig
from ..model import LearningRateScheduler, Optimizer, LearnerConfig


@dataclass
class PPOActorCriticOptimizer(Optimizer):
    lr: float = 3e-4
    max_norm: float = 0.5
    use_lr_scheduler: bool = field(default=True, metadata={"transient": True})

    def _resolve(self, ctx: dict) -> None:
        if self.use_lr_scheduler:
            num_iterations = ctx["num_environment_steps"] // ctx["episode_length"]
            updates_per_iteration = ctx["train_iterations"] * ctx["num_mini_batch"]
            self.lr_scheduler = LearningRateScheduler({
                "num_iterations": num_iterations,
                "updates_per_iteration": updates_per_iteration
            })


# PPO
@dataclass
class PPOOptimization(Resolvable, Base):
    clip_param: float = 0.2
    use_clipped_critic_loss: bool = True
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
