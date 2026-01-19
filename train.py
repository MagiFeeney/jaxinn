import tyro
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Any
from jaxtyping import PRNGKeyArray

import equinox as eqx
from gymnax.environments.environment import Environment

from agent import Agent, Transition
from envs import make_env
from config import Config

from agent.models import LatentState, LatentStateWithParams


class Trainer(eqx.Module):
    agent: eqx.Module
    env: Environment = eqx.field(static=True)
    env_params: Any = eqx.field(static=True)

    num_envs: int = eqx.field(static=True)
    num_environment_steps: int = eqx.field(static=True)
    eval_interval: int = eqx.field(static=True)
    train_interval: int = eqx.field(static=True)
    train_iterations: int = eqx.field(static=True)
    episode_length: int = eqx.field(static=True)
    num_eval_episodes: int = eqx.field(static=True)

    def __init__(self, config, *, key: PRNGKeyArray):
        self.env, self.env_params = make_env(**config.env)
        # Update config with env particulars
        config.agent.world.transition.update({"action_size": self.action_size})
        config.agent.actor.update({"action_size": self.action_size})
        observation_space = self.env.observation_space(self.env_params)
        config.agent.world.perception.encoder.update({"shape": observation_space.shape})
        config.agent.world.perception.decoder.update({"shape": observation_space.shape})
        self.agent = Agent(config.agent, key=key)
        self.__dict__.update(config.exploration())

    @property
    def action_size(self):
        action_space = self.env.action_space(self.env_params)
        if isinstance(action_space, gymnax.environments.spaces.Discrete):
            return action_space.n
        return jnp.prod(jnp.array(action_space.shape))

    def __call__(self, key: PRNGKeyArray):
        def interleaved_step_fn(carry, iteration): # Evaluation truck with unit being Training truck
            agent, key = carry
            key, key_train, key_evaluate = jax.random.split(key, 3)
            agent, metrics = jax.lax.scan(
                self.train,
                (agent, key_train),
                None,
                self.eval_interval // self.train_interval
            )
            episodic_returns = self.evaluate(agent, jax.random.split(key_evaluate, self.num_eval_episodes)) # Parallel evaluation
            evaluation = jnp.mean(episodic_returns)

            jax.debug.print(    # Callback
                """Step {k}: Train

            --- Model Loss ---
            reward:       {model/reward}
            observation:  {model/observation}
            kl:           {model/kl}
            total:        {model/total}

            --- Actor and Critic Loss ---
            actor:  {actor}
            critic: {critic}

            --- Evaluation ({self.num_eval_episodes} episodes) ---
            {e}
                """,
                k=(iteration + 1) * self.eval_interval,
                **metrics,
                e=evaluation,
            )

            return (agent, key), (metrics, evaluation)

        (final_agent, _), (metrics, evaluation) = jax.lax.scan(
            interleaved_step_fn,
            (self.agent, key),
            jnp.arange(self.num_environment_steps // self.eval_interval),
        )

        return final_agent, (metrics, evaluation)

    @eqx.filter_vmap(in_axis=(None, None, 0)) # TODO: more serious consideration of parallel train
    def train(self, agent: Agent, key: PRNGKeyArray):
        key_init, key_reset, key_interact, key_learn = jax.random.split(key, 4)
        obs, env_state = self.env.reset(key_reset, self.env_params)
        latent_state_init = self.agent.init_state(key_init)

        transition_init = Transition(
            action=jnp.zeros((self.action_size,)),
            next_obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
        )

        train_interact_step_fn = self.make_interact_step_fn(eval=False)

        _, transitions = jax.lax.scan(
            train_interact_step_fn,
            (transition_init, latent_state_init, env_state, key_interact),
            None,
            self.episode_length // self.num_envs,
        )

        transitions = jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y], axis=0), transitions_init, transitions) # insert the initial transition

        agent = self.agent.add_experience(transitions)

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
        return agent, avg_metrics

    @eqx.filter_vmap(in_axis=(None, None, 0))
    def evaluate(self, agent: Agent, key: PRNGKeyArray):
        key_init, key_reset, key_scan = jax.random.split(key, 3)
        obs, env_state = self.env.reset(key_reset, self.env_params)
        latent_state_init = self.agent.init_state(key_init)

        transition_init = Transition(
            action=jnp.zeros((self.action_size,)),
            next_obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
        )

        evaluate_interact_step_fn = self.make_interact_step_fn(eval=True)

        _, transitions = jax.lax.scan(
            evaluate_interact_step_fn,
            (transition_init, latent_state_init, env_state, key_scan),
            None,
            self.episode_length // self.num_envs,
        )

        masks = 1 - jnp.maximum.accumulate(transitions.done)
        cumulative_rewards = jnp.sum(transitions.reward * masks) # Return up to the first termination
        return cumulative_rewards

    def make_interact_step_fn(self, eval=False, prefill=False):
        def random_act_branch(operand):
            last_state, _, _, key = operand
            action = self.env.action_space(self.env_params).sample(key) # TODO: vmap for vectorization
            return last_state, action # For consistency required by branching

        def agent_act_branch(operand):
            last_state, last_action, obs, key = operand
            return self.agent.act(last_state, last_action, obs, key=key, eval=eval)

        def interact_step_fn(carry, _):
            transition, last_latent_state, env_state, key = carry
            last_action, obs, _, done = transition
            mask = 1 - done
            key, key_action, key_step = jax.random.split(key, 3)

            last_latent_state = last_latent_state * mask
            last_action = last_action * mask
            latent_state, action = jax.lax.cond(
                prefill,
                random_act_branch,
                agent_act_branch,
                (last_latent_state, last_action, obs, key_action)
            )
            key_step = jax.random.split(key_step, self.num_envs)
            next_obs, next_env_state, reward, done, info = self.env.step(key_step, env_state, action, self.env_params)

            transition = Transition(
                action=action,
                next_obs=next_obs,
                reward=reward,
                done=done,
            )

            return (transition, latent_state, next_env_state, key), transition
        return interact_step_fn


def main(args):
    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, args.num_seeds)

    @eqx.filter_jit
    @eqx.filter_vmap
    def train(key):
        key_agent, key_train = jax.random.split(key)
        trainer = Trainer(args, key=key_agent)
        return trainer(key_train)

    # Parallel agents
    final_agent, (metrics, evaluation) = train(keys)
    final_eval_return = evaluation[:, -1]
    print(
        f"{args.num_seeds} agents (multiple seeds) training completed!\n"
        f"Achieved return:\n"
        f"{final_eval_return}"
    )

    # TODO: plot figure or statistics logging


if __name__ == "__main__":
    args = tyro.cli(Config)
    main(args)
