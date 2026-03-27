import tyro
from typing import Optional, Tuple, Literal, Any, Dict
from jaxtyping import PRNGKeyArray

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import equinox as eqx

from config import Config
from custom import EnvSelector, get_config
from envs import make_env, Environment, Transition
from agent import Agent
from agent.models import LatentState, LatentStateWithParams


class Trainer(eqx.Module):
    env: Environment = eqx.field(static=True)

    num_environment_steps: int = eqx.field(static=True)
    prefill_steps: int = eqx.field(static=True)
    eval_interval: int = eqx.field(static=True)
    train_interval: int = eqx.field(static=True)
    train_iterations: int = eqx.field(static=True)
    episode_length: int = eqx.field(static=True)
    num_eval_episodes: int = eqx.field(static=True)
    action_noise: float = eqx.field(static=True)

    @classmethod
    def create(cls, config: Config):
        env = make_env(**config.env(), wrapper=config.env.wrapper())
        return cls(env=env, **config.exploration())

    def __call__(self, agent: Agent, key: PRNGKeyArray) -> Tuple[Agent, Tuple[Dict[str, Any], jax.Array]]:
        key_prefill, key_interleaved = jax.random.split(key, 2)

        # Prefill
        agent = self.prefill(agent, key_prefill)

        # Train and evaluate
        def interleaved_step_fn(carry, iteration): # Evaluation truck with unit being Training truck
            agent, key = carry
            key, key_train, key_evaluate = jax.random.split(key, 3)

            # Training
            (agent, _), metrics = jax.lax.scan(
                lambda carry, _: self.train(*carry),
                (agent, key_train),
                None,
                self.eval_interval // self.train_interval
            )

            # Evaluation
            episodic_returns = jax.vmap(self.evaluate, in_axes=(None, 0))(agent, jax.random.split(key_evaluate, self.num_eval_episodes)) # Parallel evaluation
            evaluation = jnp.mean(episodic_returns)

            jax.debug.print(    # Callback
                """Step {k}: Train

            --- Model Loss ---
            reward:       {model/reward}
            observation:  {model/observation}
            kl:           {model/kl}
            total:        {model/total}

            --- Actor and Critic Loss ---
            actor:        {actor}
            critic:       {critic}

            --- Evaluation ({n} episodes) ---
            return:       {r}
            mean:         {e}
                """,
                k=(iteration + 1) * self.eval_interval + self.prefill_steps,
                **metrics,
                n=self.num_eval_episodes,
                r=episodic_returns,
                e=evaluation,
            )

            return (agent, key), (metrics, evaluation)

        (final_agent, _), (metrics, evaluation) = jax.lax.scan(
            interleaved_step_fn,
            (agent, key_interleaved),
            jnp.arange(self.num_environment_steps // self.eval_interval),
        )

        return final_agent, (metrics, evaluation)

    def prefill(self, agent: Agent, key: PRNGKeyArray) -> Agent:
        num_prefill_episodes = self.prefill_steps // self.episode_length
        keys = jax.random.split(key, num_prefill_episodes)
        transitions, terminal_obs = jax.vmap(lambda k: self.interact(agent, k, prefill=True))(keys) # N x T x E
        agent = agent.add_experience(transitions, terminal_obs, source=2)
        return agent

    def train(self, agent: Agent, key: PRNGKeyArray) -> Tuple[Tuple[Agent, PRNGKeyArray], jax.Array]:
        key, key_interact, key_learn = jax.random.split(key, 3)
        transitions, terminal_obs = self.interact(agent, key_interact)

        # Store them
        agent = agent.add_experience(transitions, terminal_obs, source=1)

        def learn_step_fn(carry, _):
            agent, key = carry
            key, key_learn = jax.random.split(key, 2)
            new_agent, metrics = agent.learn(key_learn)
            return (new_agent, key), metrics

        (agent, _), metrics = jax.lax.scan(
            learn_step_fn,
            (agent, key_learn),
            None,
            self.train_iterations
        )

        avg_metrics = jax.tree.map(jnp.mean, metrics)
        return (agent, key), avg_metrics

    def evaluate(self, agent: Agent, key: PRNGKeyArray, num_envs: int = 1) -> jax.Array:
        transitions, _ = self.interact(agent, key, eval=True, num_envs=num_envs)
        masks = 1 - jnp.maximum.accumulate(transitions.done, axis=0)
        shifted_masks = jnp.concatenate([jnp.ones_like(masks[0:1]), masks[:-1]])
        cumulative_rewards = jnp.sum(transitions.reward * shifted_masks) # Return up to the first termination inclusively
        return cumulative_rewards

    def interact(
            self,
            agent: Agent,
            key: PRNGKeyArray,
            eval: bool = False,
            prefill: bool = False,
            num_envs: int | None = None
    ) -> Tuple[Transition, ...]:
        key_reset, key_init, key_step = jax.random.split(key, 3)
        if num_envs is not None:
            init_transition, info, env_state = self.env.reset(key_reset, num_envs=num_envs)
        else:
            init_transition, info, env_state = self.env.reset(key_reset)
            num_envs = self.env.num_envs
        init_latent_state = agent.init_state(key_init, batch_shape=(num_envs,))
        init_terminal_obs = init_transition.next_obs # Zeros: for consistency

        def random_act_branch(operand):
            last_latent_state, _, _, key = operand
            keys = jax.random.split(key, num_envs)
            action = jax.vmap(self.env.action_space.sample)(keys)
            return last_latent_state, action # For consistency

        def agent_act_branch(operand):
            last_latent_state, last_action, obs, key = operand
            key_act, key_noise = jax.random.split(key, 2)
            latent_state, action = agent.act(last_latent_state, last_action, obs, key=key_act, eval=eval)
            if not eval and self.action_noise > 0:
                if self.env.is_action_space_discrete:
                    key_idx, key_cond = jax.random.split(key_noise, 2)
                    random_idx = jax.random.randint(key_idx, action.shape[:-1], 0, self.env.action_size)
                    expl_action = jax.nn.one_hot(random_idx, self.env.action_size)

                    should_explore = jax.random.uniform(key_cond, (*action.shape[:-1], 1)) < self.action_noise
                    action = jnp.where(should_explore, expl_action, action)
                else:
                    noise = jax.random.normal(key_noise, shape=action.shape) * self.action_noise
                    action = jnp.clip(action + noise, -1.0, 1.0)

            return latent_state, action

        def interact_step_fn(carry, _):
            last_transition, last_terminal_obs, last_latent_state, env_state, key = carry
            key, key_action, key_step = jax.random.split(key, 3)

            mask = 1 - last_transition.done[..., None]
            last_latent_state = last_latent_state * mask
            last_action = last_transition.action * mask
            obs = last_transition.next_obs

            operand = (last_latent_state, last_action, obs, key_action)
            if prefill:
                latent_state, action = random_act_branch(operand)
            else:
                latent_state, action = agent_act_branch(operand)

            transition, info, next_env_state = self.env.step(key_step, env_state, action)
            return (transition, info.terminal_observation, latent_state, next_env_state, key), (last_transition, last_terminal_obs)

        _, (transitions, terminal_obs) = jax.lax.scan(
            interact_step_fn,
            (init_transition, init_terminal_obs, init_latent_state, env_state, key_step),
            None,
            self.episode_length // num_envs,
        )
        return transitions, terminal_obs


def resolve_agent_config(config: Config, env: Environment) -> Config:
    obs_shape = env.observation_space.shape
    action_size = env.action_size

    config.agent.world.transition.update({"action_size": action_size})
    config.agent.actor.update({"action_size": action_size})
    config.agent.world.perception.encoder.update({"shape": obs_shape})
    config.agent.world.perception.decoder.update({"shape": obs_shape})
    if config.agent.memory.device == "cpu":
        config.agent.memory.num_seeds = config.num_seeds # pre-allocate for all seeds upfront
    else:
        config.agent.memory.num_seeds = None             # vmap handles this

    return config.agent


def main(config):
    # Distribute RNG keys
    key = jax.random.PRNGKey(config.seed)
    num_devices = jax.device_count()
    if config.num_seeds % num_devices != 0:
        closest_lower = (config.num_seeds // num_devices) * num_devices
        closest_higher = closest_lower + num_devices
        raise ValueError(
            f"Mismatch: config.num_seeds ({config.num_seeds}) is not divisible by "
            f"num_devices ({num_devices}). \n"
            f"Please set --num_seeds to {closest_lower} or {closest_higher}."
        )
    seeds_per_device = config.num_seeds // num_devices
    keys = jax.random.split(key, config.num_seeds * 2)
    keys_agent, keys_train = keys.reshape(2, num_devices, seeds_per_device, -1)
    memory_ids = jnp.arange(num_devices * seeds_per_device).reshape(num_devices, seeds_per_device) # For anchoring cpu memory if enabled

    # Initialize trainer with environment
    trainer = Trainer.create(config)

    # Resolve agent config with environment-specific information
    agent_config = resolve_agent_config(config, trainer.env)

    # Spawn parallel agents
    def make_agent(key, memory_id):
        return Agent(agent_config, key=key, memory_id=memory_id)

    agents = jax.vmap(jax.vmap(make_agent))(keys_agent, memory_ids)

    # Ready to train
    @eqx.filter_pmap(donate="all") # shard across devices, donate buffer for memory efficiency
    @eqx.filter_vmap               # vectorise within each device
    def make_train(agent, key):
        return trainer(agent, key)

    final_agent, (metrics, evaluation) = make_train(agents, keys_train)
    final_eval_return = evaluation.reshape(config.num_seeds, -1)[:, -1]
    print(
        f"{config.num_seeds} agents/seeds training completed!\n"
        f"Achieved return:\n"
        f"{final_eval_return}"
    )


if __name__ == "__main__":
    # Grab the env id
    env_selector, _ = tyro.cli(
        EnvSelector,
        return_unknown_args=True
    )
    env_id = env_selector.env_id

    # Final CLI Pass
    config = tyro.cli(
        Config,
        default=get_config(env_id)
    )

    # Run
    main(config)
