import dataclasses
from typing import Any, ClassVar, TypeVar

import jax

ConfigT = TypeVar("ConfigT")
ModelT = TypeVar("ModelT")


class Registrable:
    _registry: ClassVar[dict[type[ConfigT], type[ModelT]]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Bind to the most direct subclass
        if Registrable in cls.__bases__:
            cls._registry = {}

        config_cls = getattr(cls, "config_cls", None)
        if config_cls is not None:
            config_cls = config_cls if isinstance(config_cls, tuple) else (config_cls,)
            for cfg in config_cls:
                cls._registry[cfg] = cls

    @classmethod
    def create(cls, config: Any, **kwargs):
        config_type = type(config)
        if config_type not in cls._registry:
            raise KeyError(f"No model registered for the config {config_type.__name__}")
        model_cls = cls._registry[config_type]

        if cls is not model_cls and hasattr(model_cls, "create"):
            return model_cls.create(config, **kwargs)  # nested / multiple routes

        if callable(config):
            config_kwargs = config()
        if dataclasses.is_dataclass(config):
            config_kwargs = dataclasses.asdict(config)
        else:
            config_kwargs = vars(config)

        key = kwargs.get("key", None)
        if key is not None:
            key_model, key_init = jax.random.split(key, 2)
            kwargs.update({"key": key_model})
        else:
            key_init = None

        model = model_cls(**config_kwargs, **kwargs)   # primitive

        if key_init is not None and hasattr(model, "apply_init"):
            model = model.apply_init(getattr(config, "initializer", None), key=key_init)
        return model
