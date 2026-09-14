import math
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import PyTree, PRNGKeyArray
import equinox as eqx

from jaxinn.common.structs import LatentState
from jaxinn.configs.head import HeadConfig
from jaxinn.configs.model import TransitionConfig

from ..base import Model
from ..perception import ActionEncoder
from ..recurrent import FusedGRUCell
from ..heads import Head
from ..distributions import DistributionLike
from ..utils import make_mlp


# Transition
class Transition(Model):
    encoder: eqx.Module
    action_encoder: ActionEncoder
    core: eqx.nn.GRUCell | FusedGRUCell | eqx.nn.LSTMCell
    body: eqx.Module
    head: Head

    core_arch: Literal["gru", "fused_gru", "lstm"] = eqx.field(static=True)

    @classmethod
    def create(cls, config: TransitionConfig, *, key: PRNGKeyArray):
        key_model, key_init = jax.random.split(key, 2)
        return cls(**config(), head_config=config.head, key=key_model).apply_init(config.initializer, key=key_init)

    def __init__(
            self,
            belief_size: int,
            state_size: int | tuple[int, ...],
            action_shape: PyTree[tuple[int, ...]],
            encoder_hidden_size: int | tuple[int, ...],
            body_hidden_size: int | tuple[int, ...],
            head_config: HeadConfig,
            core_arch: Literal["gru", "fused_gru", "lstm"] = "gru",
            activation_function = "elu",
            action_embedding_size: int | None = None,
            norm_type: Literal['layer', 'rms'] | None = None,
            norm_where: Literal['all', 'input', 'output', 'first', 'last'] | None = None,
            core_use_layernorm: bool = True,
            *,
            key: PRNGKeyArray,
    ):
        key_action_encoder, key_encoder, key_core, key_body = jax.random.split(key, 4)
        self.action_encoder = ActionEncoder(action_shape, action_embedding_size, key=key_action_encoder)
        encoded_action_size = self.action_encoder.output_size

        self.head = Head.create(head_config, event_size=state_size)

        input_size = (math.prod(state_size) if isinstance(state_size, tuple) else state_size) + encoded_action_size

        # p(c_{t - 1} | s_{t - 1}, a_{t - 1})
        encoder_norm_where = norm_where if norm_where in ['all', 'input', 'first'] else None
        self.encoder = make_mlp(
            input_size=input_size,
            hidden_size=encoder_hidden_size,
            output_size=None,
            activation=activation_function,
            norm_type=norm_type,
            norm_where=encoder_norm_where,
            key=key_encoder
        )

        # p(h_t | c_{t - 1}, h_{t - 1})
        core_input_size = encoder_hidden_size[-1]
        if core_arch == "gru":
            self.core = eqx.nn.GRUCell(core_input_size, belief_size, key=key_core)
        elif core_arch == "fused_gru":
            self.core = FusedGRUCell(core_input_size, belief_size, use_layernorm=core_use_layernorm or (norm_where == "all"), key=key_core)
        elif core_arch == "lstm":
            raise NotImplementedError("LSTM is planned for future support but is not yet implemented.")
        else:
            raise ValueError(f"Unknown core architecture: {core_arch}")

        # p(s_t | h_t)
        body_norm_where = norm_where if norm_where in ['all', 'output', 'last'] else None
        self.body = make_mlp(
            input_size=belief_size,
            hidden_size=body_hidden_size,
            output_size=self.head.param_size,
            activation=activation_function,
            norm_type=norm_type,
            norm_where=body_norm_where,
            key=key_body
        )

        self.core_arch = core_arch

    def __call__(
            self,
            latent_state: LatentState,
            action: jax.Array,
    ) -> tuple[
        DistributionLike,
        jax.Array,
    ]:
        encoded_action = self.action_encoder(action)
        input_tensor = jnp.concatenate([latent_state.state, encoded_action], axis=-1)

        embedding = self.encoder(input_tensor)
        belief = self.core(embedding, latent_state.belief)
        out = self.body(belief)

        return self.head(out), belief
