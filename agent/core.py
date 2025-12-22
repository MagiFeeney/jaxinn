import jax
import jax.random as jr
import equinox as eqx
import jax.numpy as jnp


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


def perceive():
    pass

def learn():
    """
    memory()
    reasoning(): what is it? what might be missing? will it be?
    update:
      model
      planning()
        policy
        value
    """

def reasoning():
    """
    Reason about the relationship among data and to the goal with contexts, from model itself or memory given a fixed belief;
    Reasoning is on-demand learning, which creates new knowledge and will be offloaded to the offline learning stage, e.g. dreaming

    reward
    observation
    transition
    """
    pass

def planning():
    """
    imagine()
    return()
    """
    pass

def act():
    pass

def train():
    """
    eval_interval
      train_interval
        act()
        env.step
        perceive()
    """
