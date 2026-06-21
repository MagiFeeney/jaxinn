from typing import Generic, TypeVar, Any

import jax
from jaxtyping import PRNGKeyArray, PyTree, PyTreeDef
import equinox as eqx
import optax

ModelType = TypeVar("ModelType", bound=eqx.Module)


class Learner(eqx.Module, Generic[ModelType]):
    dynamic_flatten: PyTree
    dynamic_treedef: PyTreeDef = eqx.field(static=True)
    static:  Any = eqx.field(static=True)
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    optimizer_state: optax.OptState

    def __init__(self, model: ModelType, optimizer: optax.GradientTransformation):
        dynamic, self.static = eqx.partition(model, eqx.is_inexact_array)
        self.dynamic_flatten, self.dynamic_treedef = jax.tree.flatten(dynamic)
        self.optimizer = optimizer
        self.optimizer_state = self.optimizer.init(self.dynamic_flatten)

    @classmethod
    def create(cls, model_cls, config, *, key: PRNGKeyArray) -> "Learner":
        if hasattr(model_cls, "create"): # nested / multiple routes
            model = model_cls.create(config, key=key)
        else:                            # primitive
            model = model_cls(**config(), key=key)
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.optimizer.max_norm),
            optax.adam(config.optimizer.lr, eps=config.optimizer.eps)
        )
        dynamic, static = eqx.partition(model, eqx.is_inexact_array)
        dynamic_flatten, dynamic_treedef = jax.tree.flatten(dynamic)
        optimizer_state = optimizer.init(dynamic_flatten)
        return cls(dynamic_flatten, dynamic_treedef, static, optimizer, optimizer_state)

    def update(self, grads) -> "Learner":
        if isinstance(grads, Learner):
            grads = grads.dynamic_flatten
        updates, new_optimizer_state = self.optimizer.update(grads, self.optimizer_state)
        new_dynamic_flatten = eqx.apply_updates(self.dynamic_flatten, updates)
        return eqx.tree_at(
            lambda x: (x.dynamic_flatten, x.optimizer_state),
            self,
            (new_dynamic_flatten, new_optimizer_state)
        )

    @property
    def model(self):
        dynamic = jax.tree.unflatten(self.dynamic_treedef, self.dynamic_flatten)
        return eqx.combine(dynamic, self.static)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
