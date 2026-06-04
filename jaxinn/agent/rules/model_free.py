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

    clip_param: float = eqx.field(static=True)
    num_mini_batch: int = eqx.field(static=True)

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
        self.memory = Batched(
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

    def learn(self, key: PRNGKeyArray) -> Tuple["Agent", Dict[str, jax.Array]]:
        key_memory, key_scan = jax.random.split(key, 2)
        mini_batches = self.memory.shuffle_and_batch(key_memory, self.num_mini_batch)

        def mini_batch_step_fn(carry, mini_batch):
            agent, key = carry
            key, key_actor, key_critic = jax.random.split(key, 3)
            metrics = {}

            # Update actor
            (loss, aux), grads = agent.actor_loss_fn(mini_batch, key_actor)
            new_actor = agent.actor.update(grads.actor)
            agent = eqx.tree_at(lambda x: x.actor, agent, new_actor)
            metrics.update(**aux)

            # Update critic
            (loss, aux), grads = agent.critic_loss_fn(mini_batch, key_critic)
            new_critic = agent.critic.update(grads.critic)
            agent = eqx.tree_at(lambda x: x.critic, agent, new_critic)
            metrics.update(**aux)
            return (agent, key), metrics

        (agent, _), metrics = jax.lax.scan(
            mini_batch_step_fn,
            (self, key_scan),
            mini_batches
        )
        avg_metrics = jax.tree.map(jnp.mean, metrics)
        return agent, avg_metrics


@register_agent(SACAgentConfig)
class SACAgent(SACLossMixIn, Agent):
    actor: Learner[Actor]
    critic: Learner[Critic]
    memory: Memory

    pass
