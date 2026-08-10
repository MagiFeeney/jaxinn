import abc
from typing import Any

import jax
from jaxtyping import PRNGKeyArray
import equinox as eqx

from jaxinn.configs.initializer import InitializerConfig

from .utils import apply_init
from .initializers import Initializer


class Model(eqx.Module, abc.ABC):
    _init_applied: bool = eqx.field(static=True, default=False)

    @abc.abstractmethod
    def __call__(self, x: Any) -> Any:
        pass

    def apply_init(self, config: InitializerConfig, *, key: PRNGKeyArray) -> "Model":
        if getattr(self, "_init_applied", False):
            return self

        if config is None or not any([
            getattr(config, "weight_init", None),
            getattr(config, "bias_init", None),
            getattr(config, "output_weight_init", None),
            getattr(config, "output_bias_init", None),
        ]):
            return self

        fused = getattr(config, "fused", False)

        is_child_model = lambda x: isinstance(x, Model) and x is not self

        if not fused:
            child_models = [
                x for x in jax.tree.leaves(self, is_leaf=is_child_model)
                if is_child_model(x)
            ]
            if child_models:
                key, *subkeys = jax.random.split(key, len(child_models) + 1)
                new_children = [
                    m.apply_init(config, k) for m, k in zip(child_models, subkeys)
                ]
                new_self = eqx.tree_at(
                    lambda m: [x for x in jax.tree.leaves(m, is_leaf=is_child_model) if is_child_model(x)],
                    self,
                    new_children,
                    is_leaf=is_child_model
                )

        is_target = lambda x: isinstance(x, (eqx.nn.Linear, eqx.nn.Conv2d))

        if fused:
            is_initialized_model = lambda x: is_child_model(x) and getattr(x, "_init_applied", False)
            is_leaf_primitive = lambda x: is_target(x) or is_initialized_model(x)
        else:
            is_leaf_primitive = lambda x: is_target(x) or is_child_model(x)

        key, subkey = jax.random.split(key)

        weight_init_cfg = getattr(config, "weight_init", None)
        bias_init_cfg = getattr(config, "bias_init", None)
        output_weight_init_cfg = getattr(config, "output_weight_init", None)
        output_bias_init_cfg = getattr(config, "output_bias_init", None)

        new_self = apply_init(
            model=new_self,
            weight_init=Initializer.create(weight_init_cfg) if weight_init_cfg else None,
            bias_init=Initializer.create(bias_init_cfg) if bias_init_cfg else None,
            output_weight_init=Initializer.create(output_weight_init_cfg) if output_weight_init_cfg else None,
            output_bias_init=Initializer.create(output_bias_init_cfg) if output_bias_init_cfg else None,
            is_leaf=is_leaf_primitive,
            key=subkey
        )

        return eqx.tree_at(lambda m: m._init_applied, new_self, True)
