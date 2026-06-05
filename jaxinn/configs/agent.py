from typing import Union
from dataclasses import dataclass, field

from .base import Resolvable, Base, Actor, Critic, World, Memory, _sync_statics


# Base class
@dataclass
class Agent(Resolvable, Base):
    actor: Actor = field(default_factory=Actor)
    critic: Critic = field(default_factory=Critic)
    memory: Memory = field(default_factory=Memory)

    def _resolve(self, ctx: dict) -> None:
        _sync_statics(self)


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
    world: World = field(default_factory=World)
    optimization: DreamerOptimization = field(default_factory=DreamerOptimization)
    random_init: bool = False   # Whether to initialize the state by following a simple distribution


@dataclass
class DreamerV2Optimization(DreamerOptimization):
    pg_mix: float = 1.0


@dataclass
class DreamerV2Agent(DreamerAgent):
    optimization: DreamerV2Optimization = field(default_factory=DreamerV2Optimization)


# PPO
@dataclass
class PPOOptimization(Base):
    clip_param: float = 0.2
    use_clipped_critic_loss: bool = True
    num_mini_batch: int = 8
    discount_factor: float = 0.99
    uae_lambda: float = 0.95


@dataclass
class PPOAgent(Agent):
    optimization: PPOOptimization = field(default_factory=PPOOptimization)


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
class SACAgent(Agent):
    optimization: SACOptimization = field(default_factory=SACOptimization)


AgentUnion = Union[DreamerAgent, DreamerV2Agent, PPOAgent, SACAgent]
