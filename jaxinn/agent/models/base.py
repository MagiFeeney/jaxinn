import jax
from jaxtyping import PRNGKeyArray
import equinox as eqx

from jaxinn.configs.initializer import InitializerConfig

from .utils import apply_init
from .initializers import Initializer


class Model(eqx.Module):
    _init_applied: bool = eqx.field(static=True, default=False, init=False)

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

        new_self = self
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
                    m.apply_init(config, key=k) for m, k in zip(child_models, subkeys)
                ]

                iterator = iter(new_children)
                def maybe_replace(x):
                    return next(iterator) if is_child_model(x) else x

                new_self = jax.tree.map(maybe_replace, self, is_leaf=is_child_model)

        is_target = lambda x: isinstance(x, (eqx.nn.Linear, eqx.nn.Conv, eqx.nn.ConvTranspose))

        if fused:
            is_initialized_model = lambda x: is_child_model(x) and getattr(x, "_init_applied", False)
            is_leaf_primitive = lambda x: is_target(x) or is_initialized_model(x)
        else:
            is_leaf_primitive = lambda x: is_target(x) or is_child_model(x)

        if not fused and child_models:
            key, subkey = jax.random.split(key)
        else:
            subkey = key

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

        object.__setattr__(new_self, "_init_applied", True)
        return new_self
