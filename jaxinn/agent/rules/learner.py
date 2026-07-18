from typing import Generic, TypeVar, Any, Optional, Callable

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
    def create(cls, model_cls, config, lr_scheduler: Optional[Callable] = None, *, key: PRNGKeyArray) -> "Learner":
        if hasattr(model_cls, "create"): # nested / multiple routes
            model = model_cls.create(config, key=key)
        else:                            # primitive
            model = model_cls(**config(), key=key)

        has_scheduler_fn = lr_scheduler is not None
        has_scheduler_cfg = getattr(config.optimizer, "lr_scheduler", None) is not None

        if has_scheduler_fn and has_scheduler_cfg:
            lr = lr_scheduler(config.optimizer.lr, **config.optimizer.lr_scheduler())
        else:
            lr = config.optimizer.lr

        transforms = []
        max_norm = getattr(config.optimizer, "max_norm", None)
        if max_norm is not None:
            transforms.append(optax.clip_by_global_norm(max_norm))
        transforms.append(optax.adam(learning_rate=lr, eps=getattr(config.optimizer, "eps", 1e-8)))
        optimizer = optax.chain(*transforms)

        return cls(model, optimizer)

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
