import tyro
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Any
from jaxtyping import PRNGKeyArray

import equinox as eqx

from agent import Agent
from envs import make_env
from config import Config


class Transition(eqx.Module):
    action: jax.Array
    next_obs: jax.Array
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
    env: eqx.Module
    env_params: Any = eqx.field(static=True)

    def __init__(self, config, *, key: PRNGKeyArray):
        self.env, self.env_params = make_env(**config.env)
        self.agent = Agent(config.agent, key=key) # TODO: determine useful env_params to pass in

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


def main(args):                     # TODO: vectorize Trainer
    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, args.num_seeds)

    def make_trainer(key):
        key_agent, key_train = jax.random.split(key)
        trainer = Trainer(args, key=key_agent)
        return trainer(key_train)

    # Parallel agents
    evaluations = jax.jit(jax.vmap(make_trainer))(keys)
    # evaluations = eqx.filter_jit(eqx.filter_vmap(make_trainer))(keys) # equinox version

    # TODO: plot figure or statistics logging

if __name__ == "__main__":
    args = tyro.cli(Config)
    main(args)
