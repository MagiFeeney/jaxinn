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
    next_env_state: jax.Array
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
    episode_length: int = eqx.field(static=True) # TODO: change name

    def __init__(self, config, *, key: PRNGKeyArray):
        self.env, self.env_params = make_env(**config.env)
        self.agent = Agent(config.agent, key=key) # TODO: determine useful env_params to pass in
        self.__dict__.update(config.exploration())

    def __call__(self, key: PRNGKeyArray):
        """
        eval_interval
          train_interval
            init_state()                        # s_0, a_0, h_0
            s_1 ← perceive(o_1)                 # o_1
            loop:
              a_t ← act(s_t)                    # a_t
              (o_t+1, r_t) ← env.step           # o_t+1, r_t
              s_t+1 ← perceive(o_t+1)           # s_t, a_t, h_t → h_t+1 + (o_t+1) → s_t+1
        """
        def main_step_fn(agent, _): # Evaluation truck with unit being Training truck
            agent, _ = jax.lax.scan(self.train, agent, None, self.eval_interval // self.train_interval)
            evaluation = self.evaluate(agent, key)
            return (agent, evaluation), None

        (final_agent, evaluation), _ = jax.lax.scan(main_step_fn, (self.agent, key), None, self.num_environment_steps // self.eval_interval)
        return final_agent, evaluation

    def train(self, agent: Agent, key: PRNGKeyArray):                # data + learn()
        key_init, key_reset, key_scan = jax.random.split(key, 3)
        obs, env_state = self.env.reset(key_reset, self.env_params)

        transition_init = Transition(
            latent_state=self.agent.init_state(key_init),
            action=jnp.zeros((self.action_dim,)),
            next_obs=obs,
            next_env_state=env_state,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
        )

        evaluate_step_fn = self.make_interact_step_fn(eval=False)

        _, transitions = jax.lax.scan(
            evaluate_step_fn,
            (transition_init, key_scan),
            None,
            self.episode_length, # TODO: while loop compatibility
        )

        agent = self.agent.add_experience(transitions)

        def learn_step_fn(carry, _):
            agent, key = carry
            key, key_learn = jax.random.split(key, 2)
            new_agent, metrics = agent.learn(key_learn)
            return (new_agent, key), metrics

        (agent, _), metrics = jax.lax.scan(
            learn_step_fn,
            agent,
            None,
            self.train_iterations
        )

        return agent, metrics   # TODO: aggregate metrics

    @eqx.filter_vmap
    def evaluate(self, agent: Agent, key: PRNGKeyArray):
        key_init, key_reset, key_scan = jax.random.split(key, 3)
        obs, env_state = self.env.reset(key_reset, self.env_params)

        transition_init = Transition(
            latent_state=self.agent.init_state(key_init),
            action=jnp.zeros((self.action_dim,)),
            next_obs=obs,
            next_env_state=env_state,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
        )

        evaluate_step_fn = self.make_interact_step_fn(eval=True)

        _, transitions = jax.lax.scan(
            evaluate_step_fn,
            (transition_init, key_scan),
            None,
            self.episode_length, # TODO: while loop compatibility
        )

        # cond_fn = lambda x: jnp.logical_and(
        #     x.length < self.episode_length, jnp.logical_not(x.done)
        # )

        # state = jax.lax.while_loop(
        #     cond_fn,
        #     step,
        #     state,
        # )

        return jnp.sum(transitions.reward, axis=-1) # TODO: Evaluate Datastructure

    def make_interact_step_fn(self, eval=False):
        def interact_step_fn(carry, _)
            transition, key = carry
            last_latent_state, last_action, obs, env_state, agent_state, *_ = transition
            key, key_action, key_step = jax.random.split(key, 3)

            # action, next_agent_state = agent.act(obs, latent_state, key=key_action)
            latent_state, action = agent.act(last_latent_state, last_action, obs, key=key_action, eval=eval)
            next_obs, next_env_state, reward, done, info = self.env.step(key_step, env_state, action, self.env_params)

            transition = Transition(
                latent_state=latent_state,
                action=action,
                next_obs=next_obs,
                next_env_state=next_env_state,
                reward=reward,
                done=done,
            )

            return (transition, key), transition
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


def main(args):                     # TODO: vectorize Trainer
    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, args.num_seeds)

    @jax.jit
    @jax.vmap
    def train(key):
        key_agent, key_train = jax.random.split(key)
        trainer = Trainer(args, key=key_agent)
        return trainer(key_train)

    # Parallel agents
    evaluations = train(keys)
    # evaluations = jax.jit(jax.vmap(make_trainer))(keys)
    # evaluations = eqx.filter_jit(eqx.filter_vmap(make_trainer))(keys) # equinox version

    # TODO: plot figure or statistics logging

if __name__ == "__main__":
    args = tyro.cli(Config)
    main(args)
