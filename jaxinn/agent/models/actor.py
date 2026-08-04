import math
from collections.abc import Callable

import jax
from jaxtyping import PRNGKeyArray, PyTree
import equinox as eqx

from jaxinn.structs import LatentState
from jaxinn.configs.model import ActorConfig
from jaxinn.configs.head import (
    HeadConfig,
    ComplexHeadConfig,
    HierarchicalHeadConfig
)

from .utils import make_mlp
from .heads import Head, TreeHead, HierarchicalHead
from .distributions import DistributionLike


class Actor(eqx.Module):
    net: eqx.Module
    head: TreeHead

    @classmethod
    def create(cls, config: ActorConfig, *, key: PRNGKeyArray):
        return cls(**config(), head_config=config.head, key=key)

    def __init__(
        self,
        belief_size: int,
        state_size: int | tuple[int, ...],
        hidden_size: list[int],
        action_size: PyTree[int],
        head_config: PyTree[HeadConfig],
        activation_function: str | Callable = "elu",
        *,
        key: PRNGKeyArray,
    ):
        if not isinstance(head_config, ComplexHeadConfig):
            self.head = Head.create(head_config, event_size=action_size)
        elif isinstance(head_config, HierarchicalHeadConfig):
            self.head = HierarchicalHead.create(head_config, event_size=action_size)
        else:
            self.head = TreeHead.create(head_config, event_size=action_size)

        if isinstance(state_size, tuple): # TODO: fix state_size when it is Categorical
            state_size = math.prod(state_size)

        # Build network
        self.net = make_mlp(
            input_size = belief_size + state_size,
            hidden_size = hidden_size,
            output_size = self.head.param_size,
            activation = activation_function,
            key = key
        )

    def __call__(
        self,
        latent_state: jax.Array | LatentState,
    ) -> DistributionLike:
        if isinstance(latent_state, LatentState):
            latent_state = latent_state.feature

        out = self.net(latent_state)

        return self.head(out)
