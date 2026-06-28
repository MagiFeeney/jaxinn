import math
from typing import Union, Dict, Tuple, Optional

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, PRNGKeyArray
import equinox as eqx
import distrax

from jaxinn.structs import LatentState
from jaxinn.configs.head import HeadConfig
from jaxinn.configs.model import TransitionConfig
from jaxinn.agent.models.utils import get_activation_fn, StaticCallable
from jaxinn.agent.models.common import ActionEncoder
from jaxinn.agent.models.heads import Head


# Transition
class Transition(eqx.Module):
    encoder: eqx.Module
    core: eqx.nn.GRUCell
    body: eqx.Module
    head: Head
    action_encoder: ActionEncoder

    @classmethod
    def create(cls, config: TransitionConfig, *, key: PRNGKeyArray):
        return cls(**config(), head=config.head, key=key)

    def __init__(
            self,
            belief_size: int,
            state_size: Union[int, Tuple[int, ...]],
            action_shape: PyTree[Tuple[int, ...]],
            hidden_size: int,
            head_config: HeadConfig,
            activation_function="elu",
            action_embedding_size: Optional[int] = None,
            *,
            key: PRNGKeyArray,
    ):
        key, key_encoder = jax.random.split(key, 2)
        self.action_encoder = ActionEncoder(action_shape, action_embedding_size, key=key_encoder)
        encoded_action_size = self.action_encoder.output_size

        self.head = Head.create(head_config, event_size=state_size)

        input_size = (math.prod(state_size) if isinstance(state_size, tuple) else state_size) + encoded_action_size

        activation = get_activation_fn(activation_function)

        keys = jax.random.split(key, 4)

        # p(c_{t - 1} | s_{t - 1}, a_{t - 1})
        self.encoder = eqx.nn.Sequential([
            eqx.nn.Linear(input_size, hidden_size, key=keys[0]),
            StaticCallable(activation),
        ])

        # p(h_t | c_{t - 1}, h_{t - 1})
        self.core = eqx.nn.GRUCell(hidden_size, belief_size, key=keys[1])

        # p(s_t | h_t)
        self.body = eqx.nn.Sequential([
            eqx.nn.Linear(belief_size, hidden_size, key=keys[2]),
            StaticCallable(activation),
            eqx.nn.Linear(hidden_size, self.head.param_size, key=keys[3]),
        ])

    def __call__(
            self,
            latent_state: LatentState,
            action: jax.Array,
    ) -> Tuple[
        Dict[str, jax.Array],
        jax.Array,
    ]:
        encoded_action = self.action_encoder(action)
        input_tensor = jnp.concatenate([latent_state.state, encoded_action], axis=-1)

        embedding = self.encoder(input_tensor)
        belief = self.core(embedding, latent_state.belief)
        out = self.body(belief)

        return self.head(out), belief

    def sample(
            self,
            dist: distrax.Distribution,
            key: PRNGKeyArray,
    ) -> jax.Array:
        state = self.head.sample(dist, key=key)
        return state
