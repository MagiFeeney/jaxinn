import dataclasses
from typing import Any, ClassVar, Dict, Type, TypeVar

ConfigT = TypeVar("ConfigT")
ModelT = TypeVar("ModelT")


class Registrable:
    _registry: ClassVar[Dict[Type[ConfigT], Type[ModelT]]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Bind to the most direct subclass
        if Registrable in cls.__bases__:
            cls._registry = {}

        config_cls = getattr(cls, "config_cls", None)
        if config_cls is not None:
            cls._registry[config_cls] = cls

    @classmethod
    def create(cls, config: Any, **kwargs):
        config_type = type(config)
        if config_type not in cls._registry:
            raise KeyError(f"No model registered for the config {config_type.__name__}")
        model_cls = cls._registry[config_type]

        if hasattr(model_cls, "create"):
            return model_cls.create(config, **kwargs)  # nested / multiple routes

        if callable(config):
            config_kwargs = config()
        if dataclasses.is_dataclass(config):
            config_kwargs = dataclasses.asdict(config)
        else:
            config_kwargs = vars(config)
        return model_cls(**config_kwargs, **kwargs)    # primitive
