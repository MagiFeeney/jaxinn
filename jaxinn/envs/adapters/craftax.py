from envs.adapters.gymnax import Gymnax
from craftax.craftax_env import make_craftax_env_from_name


class Craftax(Gymnax):
    @classmethod
    def create(cls, env_name: str, **kwargs) -> "Craftax":
        if "auto_reset" not in kwargs:
            kwargs["auto_reset"] = False # Delegated to jaxinn AutoReset wrapper
        env = make_craftax_env_from_name("Craftax-" + env_name, **kwargs)
        env_params = env.default_params
        return cls(env, env_params)
