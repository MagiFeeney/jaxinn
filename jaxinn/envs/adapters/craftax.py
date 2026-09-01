import jax
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.craftax_state import EnvState as CraftaxEnvState

from .gymnax import Gymnax


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
    def observation_size(self) -> tuple[int, ...]:
        key = jax.random.PRNGKey(0)
        obs, _ = jax.eval_shape(self.env.reset, key)
        return obs.shape

    @property
    def max_episode_length(self) -> int:
        return self.max_timesteps

    def _get_current_time(self, env_state: CraftaxEnvState) -> jax.Array:
        return env_state.timestep
