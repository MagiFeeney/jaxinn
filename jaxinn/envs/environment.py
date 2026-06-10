import abc
import jax
from typing import Tuple, Any, Dict
from jaxtyping import PRNGKeyArray
import equinox as eqx

from jaxinn.structs import Transition


class EnvInfo(eqx.Module):
    data: Dict[str, Any]

    def __init__(self, **kwargs):
        object.__setattr__(self, "data", kwargs)

    def __getattr__(self, item):
        # Check top level first
        if item in self.data:
            return self.data[item]

        # Look inside info otherwise
        info = self.data.get("info")
        if info is not None and isinstance(info, dict) and item in info:
            return info[item]

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{item}'")


class EnvState(EnvInfo):
    pass


class Environment(eqx.Module):
    env: Any = eqx.field(static=True)
    env_params: Any = eqx.field(static=True)

    @abc.abstractmethod
    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, EnvState]:
        pass

    @abc.abstractmethod
    def step(self, key: PRNGKeyArray, env_state: EnvState, action: jax.Array) -> Tuple[Transition, EnvInfo, EnvState]:
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
    def is_action_space_discrete(self) -> bool:
        valid_names = ("Discrete", "OneHotDiscrete")
        return type(self.action_space).__name__ in valid_names

    def __getattr__(self, name):
        if isinstance(self.env_params, dict):
            if name in self.env_params:
                return self.env_params[name]
        elif hasattr(self.env_params, name):
            return getattr(self.env_params, name)
        return getattr(self.env, name)
