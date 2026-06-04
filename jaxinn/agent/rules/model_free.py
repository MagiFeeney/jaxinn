import jax
from jaxtyping import PRNGKeyArray
import equinox as eqx

from . import register_agent
from configs import (
    PPOAgentConfig,
    SACAgentConfig
)
from .base import Agent, Experience
from .learner import Learner
from ..models import Actor, Critic
from ..losses import PPOLossMixIn, SACLossMixIn
from ..memory import Memory, Uniform, Prioritized, Batched


@register_agent(PPOAgentConfig)
class PPOAgent(PPOLossMixIn, Agent):
    actor: Learner[Actor]
    critic: Learner[Critic]
    memory: Memory

    pass


@register_agent(SACAgentConfig)
class SACAgent(SACLossMixIn, Agent):
    actor: Learner[Actor]
    critic: Learner[Critic]
    memory: Memory

    pass
