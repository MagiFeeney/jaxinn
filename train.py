import tyro
from typing import Optional, Tuple, Literal, Any, Dict
from jaxtyping import PRNGKeyArray

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import equinox as eqx

from config import Config
from envs import make_env, Transition
from agent import Agent
from agent.models import LatentState, LatentStateWithParams


class Trainer(eqx.Module):
    agent: Agent
    env: Any = eqx.field(static=True)

    num_environment_steps: int = eqx.field(static=True)
    prefill_steps: int = eqx.field(static=True)
    eval_interval: int = eqx.field(static=True)
    train_interval: int = eqx.field(static=True)
    train_iterations: int = eqx.field(static=True)
    episode_length: int = eqx.field(static=True)
    num_eval_episodes: int = eqx.field(static=True)
    action_noise_std: float = eqx.field(static=True)

    def __init__(self, config: Config, *, key: PRNGKeyArray, memory_id: jax.Array):
        self.env = make_env(**config.env())
        # Update config with env particulars
        config.agent.world.transition.update({"action_size": self.env.action_size})
        config.agent.actor.update({"action_size": self.env.action_size})
        config.agent.world.perception.encoder.update({"shape": self.env.observation_space.shape})
        config.agent.world.perception.decoder.update({"shape": self.env.observation_space.shape})
        if config.agent.memory.device == "cpu":
            config.agent.memory.num_seeds = config.num_seeds # Manually allocate memory for all seeds
        else:
            config.agent.memory.num_seeds = None             # vmap handles this
        self.agent = Agent(config.agent, key=key, memory_id=memory_id)
        self.__dict__.update(config.exploration())

    def __call__(self, key: PRNGKeyArray) -> Tuple[Agent, Tuple[Dict[str, Any], jax.Array]]:
        key_prefill, key_interleaved = jax.random.split(key, 2)

        # Prefill
        agent = self.prefill(key_prefill)

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

    def prefill(self, key: PRNGKeyArray) -> Agent:
        # Start with the initial agent: self.agent
        num_prefill_episodes = self.prefill_steps // self.episode_length
        keys = jax.random.split(key, num_prefill_episodes)
        transitions, terminal_obs = jax.vmap(lambda k: self.interact(self.agent, k, prefill=True))(keys) # N x T x E
        agent = self.agent.add_experience(transitions, terminal_obs, source=2)
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
            if not eval and self.action_noise_std > 0:
                noise = jax.random.normal(key_noise, shape=action.shape) * self.action_noise_std
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


def main(args):
    @eqx.filter_pmap
    @eqx.filter_vmap
    def train(key, memory_id):
        key_agent, key_train = jax.random.split(key)
        trainer = Trainer(args, key=key_agent, memory_id=memory_id)
        return trainer(key_train)

    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, args.num_seeds)
    num_devices = jax.device_count()
    seeds_per_device = args.num_seeds // num_devices
    keys = keys.reshape(num_devices, seeds_per_device, -1)

    memory_ids = jnp.arange(num_devices * seeds_per_device).reshape(num_devices, seeds_per_device) # For anchoring cpu memory if enabled

    # Parallel agents
    final_agent, (metrics, evaluation) = train(keys, memory_ids)
    final_eval_return = evaluation.reshape(args.num_seeds, -1)[:, -1]
    print(
        f"{args.num_seeds} agents (multiple seeds) training completed!\n"
        f"Achieved return:\n"
        f"{final_eval_return}"
    )


if __name__ == "__main__":
    args = tyro.cli(Config)
    main(args)
