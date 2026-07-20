from typing import Any, Dict, Optional, Tuple, ClassVar, Type

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx

from jaxinn.structs import Transition
from jaxinn.configs.agent.sac import SACAgentConfig
from jaxinn.agent.rules.base import Agent
from jaxinn.agent.rules.learner import Learner
from jaxinn.agent.rules.utils import transform
from jaxinn.agent.losses import SACLossMixIn
from jaxinn.agent.memory import Memory, Uniform, Prioritized

from .actor_critic import PerceptionActor, PerceptionCritic, Ensemble, make_ensemble_cls
from .utils import soft_update


class SACAgent(SACLossMixIn, Agent):
    config_cls: ClassVar[Type] = SACAgentConfig

    actor: Learner[PerceptionActor]
    critic: Learner[Ensemble[PerceptionCritic]]
    critic_target: Ensemble[PerceptionCritic]
    memory: Memory

    discount_factor: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    target_update_interval: int = eqx.field(static=True)

    num_updates: jax.Array

    @classmethod
    def create(
            cls,
            config,
            *,
            key: PRNGKeyArray,
            memory_id: jax.Array,
    ):
        key_actor, key_critic = jax.random.split(key, 2)

        actor = Learner.create(PerceptionActor, config.actor, key=key_actor)
        DoubleCritic = make_ensemble_cls(PerceptionCritic, num_ensembles=2)
        critic = Learner.create(DoubleCritic, config.critic, key=key_critic)

        critic_target = jax.tree.map(lambda x: jnp.copy(x) if eqx.is_inexact_array(x) else x, critic.model)

        if config.memory.type.lower() == "uniform":
            memory_cls = Uniform
        else:
            memory_cls = Prioritized
        memory = memory_cls(
            seed_idx=memory_id,
            capacity=config.memory.capacity,
            obs_shape=config.memory.obs_shape,
            obs_dtype=config.memory.obs_dtype,
            action_shape=config.memory.action_shape,
            action_dtype=config.memory.action_dtype,
            num_seeds=config.memory.num_seeds,
        )

        return cls(
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            memory=memory,
            num_updates=jnp.array(0, dtype=jnp.int32),
            **config.optimization() # Extra particulars for agent learning
        )

    def init_latent_state(self, key: PRNGKeyArray, batch_shape: Tuple[int, ...] = (), eval=False) -> Any:
        return None

    def act(self, last_latent_state: Optional[jax.Array], last_action: jax.Array, obs: jax.Array, *, key: PRNGKeyArray, eval: bool = False) -> Tuple[None, jax.Array]:
        actor_dist = jax.vmap(self.actor)(obs)
        action = self.actor.sample(actor_dist, key, eval)
        return None, action

    def make_batch_fn(self) -> callable:
        def step_fn(key: PRNGKeyArray):
            data = self.memory.sample((self.batch_size, 2), key) # 2 x B
            data = eqx.tree_at(
                lambda d: d.next_obs,
                data,
                replace_fn=transform
            )

            # Map a 2-step temporal sequence into a single (s, a, r, s') causal transition
            return (
                data.next_obs[0],
                data.action[1],
                data.reward[1],
                data.next_obs[1],
                data.terminated[1],
                data.truncated[1]
            )
        return step_fn

    def learn(self, data: Transition, key: PRNGKeyArray) -> Tuple["Agent", Dict[str, jax.Array]]:
        key, key_actor, key_critic = jax.random.split(key, 3)
        metrics = {}

        obs, actions, rewards, next_obs, terminated, truncated = data

        # Update critic
        (loss, aux), grads = self.critic_loss_fn(obs, actions, rewards, next_obs, terminated, key_critic)
        new_critic = self.critic.update(grads.critic)
        agent = eqx.tree_at(lambda a: a.critic, self, new_critic)
        metrics.update(**aux)

        # Update actor
        (loss, aux), grads = agent.actor_loss_fn(obs, key_actor)
        new_actor = agent.actor.update(grads.actor)
        agent = eqx.tree_at(lambda a: a.actor, agent, new_actor)
        metrics.update(**aux)

        do_update = (self.num_updates % self.target_update_interval == 0)
        new_critic_target = jax.lax.cond(
            do_update,
            lambda: soft_update(self.critic_target, new_critic.model, self.tau),
            lambda: self.critic_target
        )
        agent = eqx.tree_at(
            lambda a: (a.critic_target, a.num_updates),
            agent,
            (new_critic_target, self.num_updates + 1)
        )

        return agent, metrics
