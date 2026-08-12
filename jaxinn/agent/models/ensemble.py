from typing import Any, Generic, TypeVar

import jax
from jaxtyping import PRNGKeyArray
import equinox as eqx

T = TypeVar("T", bound=eqx.Module)


class Ensemble(eqx.Module, Generic[T]):
    nets: T

    @classmethod
    def create(cls, model_cls: type[T], num_ensembles: int, config: Any, *, key: PRNGKeyArray):
        keys = jax.random.split(key, num_ensembles)

        def make_model(k):
            if hasattr(model_cls, "create"): # nested / multiple routes
                return model_cls.create(config, key=k)
            else:                            # primitive
                return model_cls(**config(), key=k)

        nets = eqx.filter_vmap(make_model)(keys)
        return cls(
            nets=nets,
        )

    def __call__(self, *args, **kwargs) -> Any: # Unbatched call
        return eqx.filter_vmap(lambda net: net(*args, **kwargs))(self.nets)


def make_ensemble_cls(model_cls: type[T], num_ensembles: int) -> type[Ensemble[T]]:
    class BoundedEnsemble(Ensemble[model_cls]):
        @classmethod
        def create(cls, config: Any, *, key: PRNGKeyArray):
            return super().create(
                model_cls=model_cls,
                num_ensembles=num_ensembles,
                config=config,
                key=key
            )

    BoundedEnsemble.__name__ = f"{model_cls.__name__}Ensemble{num_ensembles}"
    return BoundedEnsemble
