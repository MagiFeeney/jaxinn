import tyro
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Any
from jaxtyping import PRNGKeyArray

import equinox as eqx

from agent import Agent
from envs import make_env


class Trainer(eqx.Module):
    agent: eqx.Module
    env: eqx.Module
    env_params: Any = eqx.field(static=True)

    def __init__(self, config, *, key: PRNGKeyArray):
        self.env, self.env_params = make_env(**config.env)
        self.agent = Agent(**config.agent, key=key) # TODO: determine useful env_params to pass in

    def __call__(self, key: jax.random.PRNGKey): # TODO: train loop
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


@dataclass
class Args:
    # General
    agent_name: Literal['PSRL', 'EUBRL', 'QLearning', 'RMAX', 'SARSA', 'VBRB', 'BEB'] = 'EUBRL'
    """the agent you wish to choose"""


def main(args):                     # TODO: vectorize Trainer
    key = jax.random.PRNGKey(args.seed)
    key_agent, key_train = jax.random.split(key, 2)

    trainer = Trainer(args, key=key_agent)

    keys = jax.random.split(key_train, args.num_seeds) # vectorization

    vmap_train = jax.jit(jax.vmap(trainer, in_axes=0))
    ts, (_, returns) = vmap_train(keys)

    # TODO: plot figure or statistics logging

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
