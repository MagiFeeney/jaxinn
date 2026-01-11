import tyro
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Any
from jaxtyping import PRNGKeyArray

import equinox as eqx
from gymnax.environments.environment import Environment

from agent import Agent
from envs import make_env
from config import Config

from agent.models import LatentState, LatentStateWithParams


class Transition(eqx.Module):
    latent_state: jax.Array
    action: jax.Array
    next_obs: jax.Array
    reward: jax.Array
    done: jax.Array


class Trainer(eqx.Module):
    agent: eqx.Module
    env: Environment = eqx.field(static=True)
    env_params: Any = eqx.field(static=True)

    num_environment_steps: int = eqx.field(static=True)
    eval_interval: int = eqx.field(static=True)
    train_interval: int = eqx.field(static=True)
    train_iterations: int = eqx.field(static=True)
    episode_length: int = eqx.field(static=True)
    num_eval_episodes: int = eqx.field(static=True)

    def __init__(self, config, *, key: PRNGKeyArray):
        self.env, self.env_params = make_env(**config.env)
        # Update config with env particulars
        config.agent.transition.update({"action_size": self.action_dim})
        config.agent.encoder.update({"shape": self.observation_space.shape})
        config.agent.decoder.update({"shape": self.observation_space.shape})
        self.agent = Agent(config.agent, key=key)
        self.__dict__.update(config.exploration())

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

        transition_init = Transition(
            latent_state=self.agent.init_state(key_init),
            action=jnp.zeros((self.action_dim,)),
            next_obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
        )

        train_interact_step_fn = self.make_interact_step_fn(eval=False)

        _, transitions = jax.lax.scan(
            train_interact_step_fn,
            (transition_init, env_state, key_interact),
            None,
            self.episode_length,
        )

        transitions = jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y], axis=0), transitions_init, transitions) # insert the initial transition

        agent = self.agent.add_experience(
            action=transitions.action,
            reward=transitions.reward,
            next_obs=(
                transitions.observation_before_reset if transitions.done else transitions.next_observation # TODO: real observation after auto-reset? see jaxinn.org.
            ),
            done=transitions.done,
        )

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

        transition_init = Transition(
            latent_state=self.agent.init_state(key_init),
            action=jnp.zeros((self.action_dim,)),
            next_obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
        )

        evaluate_interact_step_fn = self.make_interact_step_fn(eval=True)

        _, transitions = jax.lax.scan(
            evaluate_interact_step_fn,
            (transition_init, env_state, key_scan),
            None,
            self.episode_length,
        )

        masks = 1 - jnp.maximum.accumulate(transitions.done)
        cumulative_rewards = jnp.sum(transitions.reward * masks) # Return up to the first termination
        return cumulative_rewards

    def make_interact_step_fn(self, eval=False):
        def interact_step_fn(carry, _)
            transition, env_state, key = carry
            last_latent_state, last_action, obs, *_ = transition
            key, key_action, key_step = jax.random.split(key, 3)

            latent_state, action = agent.act(last_latent_state, last_action, obs, key=key_action, eval=eval)
            next_obs, next_env_state, reward, done, info = self.env.step(key_step, env_state, action, self.env_params)

            transition = Transition(
                latent_state=latent_state,
                action=action,
                next_obs=next_obs,
                reward=reward,
                done=done,
            )

            return (transition, next_env_state, key), transition
        return interact_step_fn

    @property
    def action_dim(self):
        action_space = self.env.action_space(self.env_params)
        if isinstance(action_space, gymnax.environments.spaces.Discrete):
            return action_space.n
        return jnp.prod(jnp.array(action_space.shape))

    @property
    def action_space(self):
        return self.env.action_space(self.env_params)

    @property
    def observation_space(self):
        return self.env.observation_space(self.env_params)


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
    print(f"{args.num_seeds} num. of agents (multiple seeds) training done!\nAchieved return:\n {final_eval_return}")

    # TODO: plot figure or statistics logging


if __name__ == "__main__":
    args = tyro.cli(Config)
    main(args)
