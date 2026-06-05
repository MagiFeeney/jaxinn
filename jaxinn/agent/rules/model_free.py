import jax
from jaxtyping import PRNGKeyArrayi, PyTree
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
    use_clipped_critic_loss: bool = eqx.field(static=True)
    num_mini_batch: int = eqx.field(static=True)
    discount_factor: float = eqx.field(static=True)
    uae_lambda: float = eqx.field(static=True)

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

        # Extra particulars for agent learning
        self.__dict__.update(config.optimization())

    def init_state(self, key: PRNGKeyArray, batch_shape: Tuple[int, ...] = (), eval=False) -> Any:
        return None

    def act(self, last_latent_state: Optional[jax.Array], last_action: jax.Array, obs: jax.Array, *, key: PRNGKeyArray, eval: bool = False) -> Tuple[None, jax.Array]:
        params = jax.vmap(self.actor)(obs)
        action = self.actor.sample(params, key, eval)
        return None, action

    def learn(self, key: PRNGKeyArray) -> Tuple["Agent", Dict[str, jax.Array]]:
        key_shuffle, key_scan = jax.random.split(key, 2)
        data = self.memory.get_all()

        # Get advantages and returns
        values = jax.vmap(self.critic)(data.next_obs).mean()
        baselines = values[:-1]
        advantages, returns = compute_adv_and_ret(
            data.reward[1:],
            values[:-1],
            baselines,
            data.done[1:],
            bootstrap=values[-1],
            discount_factor=self.discount_factor,
            uae_lambda=self.uae_lambda
        )

        # Get action log probs
        actor_params = jax.vmap(self.actor)(data.next_obs[:-1])
        actor_dists = self.actor.get_dist(actor_params)
        log_probs = actor_dists.log_prob(data.action[1:])

        # Apply shuffle and split for training data
        train_data = (data.next_obs[:-1], data.action[1:], advantages, returns, values[:-1], log_probs)
        mini_batches = self.shuffle_and_split(train_data, key_shuffle)

        def mini_batch_step_fn(carry, mini_batch):
            agent, key = carry
            key, key_actor, key_critic = jax.random.split(key, 3)
            obs, actions, advantages, returns, values, log_probs = mini_batch
            metrics = {}

            # Update actor
            (loss, aux), grads = agent.actor_loss_fn(obs, actions, log_probs, advantages, key_actor)
            new_actor = agent.actor.update(grads.actor)
            agent = eqx.tree_at(lambda x: x.actor, agent, new_actor)
            metrics.update(**aux)

            # Update critic
            (loss, aux), grads = agent.critic_loss_fn(obs, returns, values, key_critic)
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
