from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree
import equinox as eqx

from jaxinn.agent.rules import register_agent
from configs import SACAgentConfig
from jaxinn.agent.rules.base import Agent
from jaxinn.agent.rules.learner import Learner
from jaxinn.agent.rules.utils import compute_adv_and_ret
from jaxinn.agent.models import Actor, Critic
from jaxinn.agent.losses import SACLossMixIn
from jaxinn.agent.memory import Memory, Uniform, Prioritized


@register_agent(SACAgentConfig)
class SACAgent(SACLossMixIn, Agent):
    actor: Learner[Actor]
    critic: Learner[Critic]
    memory: Memory

    def __init__(
            self,
            config,
            *,
            key: PRNGKeyArray,
            memory_id: jax.Array,
    ):
        key_actor, key_critic = jax.random.split(key, 2)
        self.actor = Learner.create(Actor, config.actor, key=key_actor)
        self.critic = Learner.create(Critic, config.critic, key=key_critic)
        if config.memory.type.lower() == "uniform":
            memory_cls = Uniform
        else:
            memory_cls = Prioritized
        self.memory = memory_cls(
            seed_idx=memory_id,
            capacity=config.memory.capacity,
            obs_shape=config.world.perception.encoder.shape,
            action_size=config.world.transition.action_size,
            num_seeds=config.memory.num_seeds,
        )
        self.belief_size = 0

        # Extra particulars for agent learning
        self.__dict__.update(config.optimization())

    def init_state(self, key: PRNGKeyArray, batch_shape: Tuple[int, ...] = (), eval=False) -> Any:
        return None

    def act(self, last_latent_state: Optional[jax.Array], last_action: jax.Array, obs: jax.Array, *, key: PRNGKeyArray, eval: bool = False) -> Tuple[None, jax.Array]:
        params = jax.vmap(self.actor)(obs)
        action = self.actor.sample(params, key, eval)
        return None, action
