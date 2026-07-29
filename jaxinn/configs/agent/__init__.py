from .base import AgentConfig
from .dreamer import DreamerAgentConfig, DreamerV2AgentConfig
from .ppo import PPOAgentConfig
from .sac import SACAgentConfig

AgentUnion = DreamerAgentConfig | DreamerV2AgentConfig | PPOAgentConfig | SACAgentConfig

__all__ = [
    "AgentConfig",
    "DreamerAgentConfig",
    "DreamerV2AgentConfig",
    "PPOAgentConfig",
    "SACAgentConfig",
]
