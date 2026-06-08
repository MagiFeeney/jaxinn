from typing import Union

from .base import Agent
from .dreamer import DreamerAgent, DreamerV2Agent
from .ppo import PPOAgent, PerceptionActor, PerceptionCritic, ActorCriticDecoupled, ActorCriticShared
from .sac import SACAgent

AgentUnion = Union[DreamerAgent, DreamerV2Agent, PPOAgent, SACAgent]


__all__ = [
    "Agent",
    "DreamerAgent",
    "DreamerV2Agent",
    "PPOAgent",
    "PerceptionActor",
    "PerceptionCritic",
    "ActorCriticDecoupled",
    "ActorCriticShared",
    "SACAgent",
]
