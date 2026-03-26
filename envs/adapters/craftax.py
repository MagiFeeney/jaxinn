import math
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from envs.environment import Transition, Environment, EnvInfo
from envs.adapters.gymnax import Gymnax

from gymnax.environments.spaces import Discrete
from gymnax.environments.environment import Environment as GymnaxEnvironment

from craftax.craftax.craftax_state import EnvParams as CraftaxEnvParams, EnvState as CraftaxEnvState
from craftax.craftax_env import make_craftax_env_from_name


class Craftax(Gymnax):
    @classmethod
    def create(cls, env_name: str, **kwargs) -> "Craftax":
        if "auto_reset" not in kwargs:
            kwargs["auto_reset"] = False # Delegated to jaxinn AutoReset wrapper
        env = make_craftax_env_from_name(env_name, **kwargs)
        env_params = env.default_params
        return cls(env, env_params)
