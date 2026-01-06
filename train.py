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


class TransitionState(eqx.Module):
    latent_state: LatentState
    action: jax.Array
    obs: jax.Array


class Transition(eqx.Module):
    action: jax.Array
    next_obs: jax.Array
    next_env_state              # TODO: inclusion of env_state
    reward: jax.Array
    done: jax.Array

    @classmethod
    def init_empty(cls, obs_shape, action_dim):
        return cls(
            action=jnp.zeros((action_dim,)),
            next_obs=jnp.zeros(obs_shape),
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
        )


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

    def __call__(self, key: PRNGKeyArray): # TODO: train loop
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

    @eqx.filter_vmap
    def evaluate(self, key: PRNGKeyArray):
        key_reset, key_scan = jax.random.split(key, 2)
        def step_fn(carry, _):
            obs, env_state, agent_state, key = carry
            key, key_action, key_step = jax.random.split(key, 3)

            action, agent_state = self.agent.act(obs, agent_state, key=key_action)
            next_obs, next_env_state, reward, done, info = self.env.step(key_step, env_state, action, self.env_params)
            transition = Transition()        # TODO: assign values

            return (obs, env_state, agent_state, key), transition

        obs, env_state = self.env.reset(key_reset, env_params)
        agent_state = TransitionState.init_empty()

        _, transitions = jax.lax.scan(
            step_fn,
            (obs, env_state, agent_state, key_scan),
            None,
            self.episode_length, # TODO: while loop compatibility
        )

        return jnp.sum(transitions.reward, axis=-1)


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
