import math
from dataclasses import dataclass, field

from jaxinn.configs.base import Base, Resolvable

from .base import AgentConfig
from ..model import Optimizer, LearnerConfig, PerceptionActorConfig, PerceptionCriticConfig, ActorCriticSharedConfig, ActorCriticDecoupledConfig
from ..scheduler import LearningRateSchedulerUnion, StaircaseScheduleConfig
from ..initializer import Initializer, OrthogonalConfig, ConstantConfig


@dataclass
class PPOActorCriticSharedConfig(ActorCriticSharedConfig):
    initializer: Initializer = field(
        default_factory=lambda: Initializer(
            weight_init=OrthogonalConfig(scale=math.sqrt(2)),
            bias_init=ConstantConfig(value=0.0),
        )
    )

    def _resolve(self, ctx: dict) -> None:
        super()._resolve(ctx)

        base_weight_init = self.initializer.weight_init
        base_bias_init = self.initializer.bias_init

        if base_weight_init is not None and isinstance(base_weight_init, OrthogonalConfig):
            if self.encoder.initializer.is_empty:
                self.encoder.initializer.weight_init = base_weight_init
                self.encoder.initializer.bias_init = base_bias_init

            if self.actor.initializer.is_empty:
                self.actor.initializer.weight_init = base_weight_init
                self.actor.initializer.output_weight_init = OrthogonalConfig(scale=0.01)
                self.actor.initializer.bias_init = base_bias_init

            if self.critic.initializer.is_empty:
                self.critic.initializer.weight_init = base_weight_init
                self.critic.initializer.output_weight_init = OrthogonalConfig(scale=1.0)
                self.critic.initializer.bias_init = base_bias_init


@dataclass
class PPOActorCriticDecoupledConfig(ActorCriticDecoupledConfig):
    perception_actor: PerceptionActorConfig = field(
        default_factory=lambda: PerceptionActorConfig(
            initializer=Initializer(
                weight_init=OrthogonalConfig(scale=math.sqrt(2)),
                output_weight_init=OrthogonalConfig(scale=0.01),
                bias_init=ConstantConfig(value=0.0),
                fused=True,
            )
        )
    )
    perception_critic: PerceptionCriticConfig = field(
        default_factory=lambda: PerceptionCriticConfig(
            initializer=Initializer(
                weight_init=OrthogonalConfig(scale=math.sqrt(2)),
                output_weight_init=OrthogonalConfig(scale=1.0),
                bias_init=ConstantConfig(value=0.0),
                fused=True,
            )
        )
    )


PPOActorCriticUnion = PPOActorCriticDecoupledConfig | PPOActorCriticSharedConfig


@dataclass
class PPOActorCriticOptimizer(Resolvable, Optimizer):
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

    actor_critic: LearnerConfig[PPOActorCriticUnion] = field(
        default_factory=lambda: LearnerConfig(
            model=PPOActorCriticSharedConfig(),
            optimizer=PPOActorCriticOptimizer()
        )
    )
