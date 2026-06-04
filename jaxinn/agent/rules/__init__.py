from typing import Type, Any, Dict
from configs import AgentConfig

registered_agents: Dict[Type[AgentConfig], Type[Any]] = {}


def register_agent(config_class: Type[AgentConfig]):
    """
    A decorator that maps a configuration class to an agent class
    in the registered_agents dictionary.
    """
    def decorator(agent_class: Type[Any]):
        registered_agents[config_class] = agent_class
        return agent_class
    return decorator


def get_agent_cls(config: AgentConfig) -> Type[Any]:
    config_type = type(config)
    try:
        return registered_agents[config_type]
    except KeyError:
        raise KeyError(f"Configuration type '{config_type.__name__}' is not registered in the agent registry.")


from .model_based import DreamerAgent, DreamerV2Agent
from .model_free import PPOAgent, SACAgent
from .base import Experience, Agent
from .learner import Learner
from .utils import compute_adv_and_ret


__all__ = [
    "register_agent",
    "get_agent_cls",
    "Agent",
    "DreamerAgent",
    "DreamerV2Agent",
    "PPOAgent",
    "SACAgent",
    "Experience",
    "Learner",
    "compute_adv_and_ret"
]
