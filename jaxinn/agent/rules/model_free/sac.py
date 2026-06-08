from typing import Any, Dict, Optional, Tuple, ClassVar, Type

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree
import equinox as eqx

from jaxinn.configs import SACAgentConfig
from jaxinn.agent.rules.base import Agent
from jaxinn.agent.rules.learner import Learner
from jaxinn.agent.rules.utils import compute_adv_and_ret
from jaxinn.agent.models import Actor, Critic
from jaxinn.agent.losses import SACLossMixIn
from jaxinn.agent.memory import Memory, Uniform, Prioritized


class SACAgent(SACLossMixIn, Agent):
    config_cls: ClassVar[Type] = SACAgentConfig

    actor: Learner[Actor]
    critic: Learner[Critic]
    memory: Memory

    @classmethod
    def create(
            cls,
            config,
            *,
            key: PRNGKeyArray,
            memory_id: jax.Array,
    ):
        key_actor, key_critic = jax.random.split(key, 2)
        actor = Learner.create(Actor, config.actor, key=key_actor)
        critic = Learner.create(Critic, config.critic, key=key_critic)
        if config.memory.type.lower() == "uniform":
            memory_cls = Uniform
        else:
            memory_cls = Prioritized
        memory = memory_cls(
            seed_idx=memory_id,
            capacity=config.memory.capacity,
            obs_shape=config.world.perception.encoder.shape,
            action_size=config.world.transition.action_size,
            num_seeds=config.memory.num_seeds,
        )

        return cls(
            actor=actor,
            critic=critic,
            memory=memory,
            **config.optimization() # Extra particulars for agent learning
        )

    def init_state(self, key: PRNGKeyArray, batch_shape: Tuple[int, ...] = (), eval=False) -> Any:
        return None

    def act(self, last_latent_state: Optional[jax.Array], last_action: jax.Array, obs: jax.Array, *, key: PRNGKeyArray, eval: bool = False) -> Tuple[None, jax.Array]:
        params = jax.vmap(self.actor)(obs)
        action = self.actor.sample(params, key, eval)
        return None, action
