from dataclasses import dataclass
import re
import importlib

from .wrapper import Batched, AutoReset
from .environment import Environment


@dataclass(frozen=True)
class EnvSpec:
    module: str
    cls_name: str
    native_batched: bool = False
    native_autoreset: bool = False


_FACTORY_REGISTRY = {
    "gymnax":    EnvSpec(".adapters.gymnax", "Gymnax", native_autoreset=True),
    "mjx":       EnvSpec(".adapters.playground", "Playground"),
    "brax":      EnvSpec(".adapters.brax", "Brax"),
    "navix":     EnvSpec(".adapters.navix", "Navix"),
    "craftax":   EnvSpec(".adapters.craftax", "Craftax"),
    "envpool":   EnvSpec(".adapters.envpool", "EnvPool", native_batched=True, native_autoreset=True),
}


def make_env(env_id: str, **kwargs) -> Environment:
    parts = re.split(r'[:/]', env_id, maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Invalid env_id '{env_id}'. Must be 'source/env_name'.")
    source, env_name = parts[0].lower(), parts[1]

    if source not in _FACTORY_REGISTRY:
        raise ValueError(f"Unknown source: '{source}'. Valid: {list(_FACTORY_REGISTRY.keys())}")

    spec = _FACTORY_REGISTRY[source]

    try:
        module = importlib.import_module(spec.module, package=__package__)
        cls = getattr(module, spec.cls_name)
    except (ImportError, TypeError) as e:
        raise ImportError(f"Failed to import adapter for '{source}'. Do you have '{spec.module}' installed?") from e

    num_envs = kwargs.get('num_envs', 1)

    if spec.native_batched:
        env = cls.create(env_name, **kwargs)
    else:
        create_kwargs = kwargs.copy()
        create_kwargs.pop('num_envs', None)
        env = cls.create(env_name, **create_kwargs)

    if not spec.native_autoreset:
        env = AutoReset(env)

    if not spec.native_batched:
        env = Batched(env, num_envs=num_envs)

    return env
