import abc
from typing import Any, ClassVar

import jax
from jaxtyping import PRNGKeyArray, PyTree
import equinox as eqx

from jaxinn.common.structs import Transition


class EnvInfo(eqx.Module):
    data: dict[str, Any]

    _lookup_key: ClassVar[str] = "info"

    def __init__(self, **kwargs):
        object.__setattr__(self, "data", kwargs)

    def __getattr__(self, item):
        # Check top level first
        if item in self.data:
            return self.data[item]

        # Look inside info otherwise
        inner = self.data.get(self._lookup_key)

        if inner is not None:
            if isinstance(inner, dict) and item in inner:
                return inner[item]
            elif hasattr(inner, item):
                return getattr(inner, item)

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{item}'")


class EnvState(EnvInfo):
    _lookup_key: ClassVar[str] = "state"


class Environment(eqx.Module):
    env: Any = eqx.field(static=True)
    env_params: Any = eqx.field(static=True)

    @abc.abstractmethod
    def reset(self, key: PRNGKeyArray) -> tuple[Transition, EnvInfo, EnvState]:
        pass

    @abc.abstractmethod
    def step(self, key: PRNGKeyArray, env_state: EnvState, action: jax.Array) -> tuple[Transition, EnvInfo, EnvState]:
        pass

    @property
    @abc.abstractmethod
    def observation_space(self):
        pass

    @property
    @abc.abstractmethod
    def action_space(self):
        pass

    @property
    @abc.abstractmethod
    def max_episode_length(self) -> int | None:
        pass

    @property
    def action_size(self) -> PyTree[int]:
        return self.action_space.size

    def __getattr__(self, name):
        if isinstance(self.env_params, dict):
            if name in self.env_params:
                return self.env_params[name]
        elif hasattr(self.env_params, name):
            return getattr(self.env_params, name)
        return getattr(self.env, name)
