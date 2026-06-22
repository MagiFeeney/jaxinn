from typing import Tuple

import jax
from craftax.craftax_env import make_craftax_env_from_name

from .adapters.gymnax import Gymnax


class Craftax(Gymnax):
    @classmethod
    def create(cls, env_name: str, **kwargs) -> "Craftax":
        if "auto_reset" not in kwargs:
            kwargs["auto_reset"] = False # Delegated to jaxinn AutoReset wrapper
        env = make_craftax_env_from_name("Craftax-" + env_name, **kwargs)
        env_params = env.default_params
        return cls(env, env_params)

    @property
    def observation_space(self):
        space = super().observation_space
        new_shape = self.observation_size
        return type(space)(
            low=space.low,
            high=space.high,
            shape=new_shape,
            dtype=space.dtype
        )

    @property
    def observation_size(self) -> Tuple[int, ...]:
        key = jax.random.PRNGKey(0)
        obs, _ = jax.eval_shape(self.env.reset, key)
        return obs.shape
