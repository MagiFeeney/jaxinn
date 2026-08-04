from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree
import equinox as eqx

from jaxinn.common.structs import Experience
from jaxinn.configs.agent.ppo import PPOAgentConfig
from jaxinn.agent.rules.base import Agent
from jaxinn.agent.rules.learner import Learner
from jaxinn.agent.rules.utils import compute_adv_and_ret, staircase_lr_schedule
from jaxinn.agent.losses import PPOLossMixIn

from .actor_critic import ActorCritic
from ..utils import reconstruct_rl_tuple


class PPOAgent(PPOLossMixIn, Agent):
    config_cls: ClassVar[type] = PPOAgentConfig

    actor_critic: Learner[ActorCritic]
    memory: Experience

    clip_param: float = eqx.field(static=True)
    use_clipped_critic_loss: bool = eqx.field(static=True)
    entropy_coef: float = eqx.field(static=True)
    num_mini_batch: int = eqx.field(static=True)
    discount_factor: float = eqx.field(static=True)
    uae_lambda: float = eqx.field(static=True)
    normalize_adv: bool = eqx.field(static=True)

    @classmethod
    def create(
            cls,
            config,
            *,
            key: PRNGKeyArray,
            memory_id: jax.Array,
    ):
        actor_critic = Learner.create(ActorCritic, config.actor_critic, lr_scheduler=staircase_lr_schedule, key=key)
        memory = Experience.initialize(
            capacity=config.memory.capacity,
            obs_shape=config.memory.obs_shape,
            obs_dtype=config.memory.obs_dtype,
            action_shape=config.memory.action_shape,
            action_dtype=config.memory.action_dtype,
            needs_boundary_obs=config.memory.needs_boundary_obs
        )

        return cls(
            actor_critic=actor_critic,
            memory=memory,
            **config.optimization() # Extra particulars for agent learning
        )

    def init_latent_state(self, key: PRNGKeyArray, batch_shape: tuple[int, ...] = (), eval=False) -> Any:
        return None

    def act(self, last_latent_state: jax.Array | None, last_action: jax.Array, obs: jax.Array, *, key: PRNGKeyArray, eval: bool = False) -> tuple[None, jax.Array]:
        actor_dist = jax.vmap(self.actor_critic.get_actor_dist)(obs)
        action = self.actor_critic.actor.sample(actor_dist, key, eval)
        return None, action

    def make_batch_fn(self) -> callable:
        transition, boundary_obs = self.memory.transition, self.memory.boundary_obs
        obs, actions, rewards, next_obs, terminated, truncated = reconstruct_rl_tuple(transition, boundary_obs)

        # Get advantages and returns
        values = jax.vmap(jax.vmap(self.actor_critic.get_critic_dist))(obs).mean()
        next_values = jax.vmap(jax.vmap(self.actor_critic.get_critic_dist))(next_obs).mean() # Recalculate values on actual terminal observation to handle truncation

        baselines = values
        advantages, returns = compute_adv_and_ret(
            rewards,
            values,
            baselines,
            terminated,
            next_values,
            discount_factor=self.discount_factor,
            uae_lambda=self.uae_lambda
        )

        # Get action log probs
        actor_dists = jax.vmap(jax.vmap(self.actor_critic.get_actor_dist))(obs)
        log_probs = actor_dists.log_prob(actions)

        # Apply shuffle and split for training data
        train_data = (obs, actions, advantages, returns, values, log_probs)
        flatten_train_data = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), train_data)

        def step_fn(key: PRNGKeyArray):
            mini_batches = self.shuffle_and_split(flatten_train_data, key)
            return mini_batches
        return step_fn

    def learn(self, mini_batches: tuple[jax.Array], key: PRNGKeyArray) -> tuple["Agent", dict[str, jax.Array]]:
        def mini_batch_step_fn(carry, mini_batch):
            agent, key = carry
            key, key_ac = jax.random.split(key, 2)
            metrics = {}

            # Update the actor-critic
            (loss, aux), grads = agent.actor_critic_loss_fn(mini_batch, key_ac)
            new_actor_critic = agent.actor_critic.update(grads.actor_critic)
            agent = eqx.tree_at(lambda a: a.actor_critic, agent, new_actor_critic)
            metrics.update(**aux)

            return (agent, key), metrics

        (agent, _), metrics = jax.lax.scan(
            mini_batch_step_fn,
            (self, key),
            mini_batches
        )
        avg_metrics = jax.tree.map(jnp.mean, metrics)
        return agent, avg_metrics

    def shuffle_and_split(self, batch: PyTree, key: PRNGKeyArray):
        size = jax.tree.leaves(batch)[0].shape[0]

        sample_index = jax.random.permutation(key, size)
        shuffled_batch = jax.tree.map(lambda x: x[sample_index], batch)

        mini_batch_size = size // self.num_mini_batch
        valid_size = mini_batch_size * self.num_mini_batch

        split_data = jax.tree.map(
            lambda x: x[:valid_size].reshape(self.num_mini_batch, mini_batch_size, *x.shape[1:]),
            shuffled_batch
        )
        return split_data
