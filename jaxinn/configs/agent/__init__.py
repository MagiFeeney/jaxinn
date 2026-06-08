from typing import Union

from .dreamer import DreamerAgent, DreamerV2Agent
from .ppo import PPOAgent, PerceptionActor, PerceptionCritic, ActorCriticDecoupled, ActorCriticShared
from .sac import SACAgent

AgentUnion = Union[DreamerAgent, DreamerV2Agent, PPOAgent, SACAgent]


__all__ = [
    "PerceptionActor",
    "PerceptionCritic",
    "ActorCriticDecoupled",
    "ActorCriticShared",

    "DreamerAgent",
    "DreamerV2Agent",
    "PPOAgent",
    "SACAgent",
]
