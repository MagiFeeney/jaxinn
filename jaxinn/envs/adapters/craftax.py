import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from craftax.craftax_env import make_craftax_env_from_name
from craftax.environment_base.environment_bases import EnvironmentNoAutoReset as CraftaxEnvironmentNoAutoReset
from craftax.craftax.craftax_state import EnvState as CraftaxEnvState, EnvParams as CraftaxEnvParams

from jaxinn.common.structs import Transition

from ..environment import Environment, EnvInfo


class Craftax(Environment):
    def __init__(
            self,
            env: CraftaxEnvironmentNoAutoReset,
            env_params: CraftaxEnvParams | None = None,
    ):
        super().__init__(env, env_params)

    @classmethod
    def create(cls, env_name: str, **kwargs) -> "Craftax":
        if "auto_reset" not in kwargs:
            kwargs["auto_reset"] = False # Delegated to jaxinn AutoReset wrapper
        env = make_craftax_env_from_name("Craftax-" + env_name, **kwargs)
        env_params = env.default_params
        return cls(env, env_params)

    def reset(self, key: PRNGKeyArray) -> tuple[Transition, EnvInfo, CraftaxEnvState]:
        obs, env_state = self.env.reset(key, self.env_params)
        transition = Transition(
            action=jnp.zeros(self.action_space.shape, dtype=self.action_space.dtype),
            next_obs=obs,
            reward=jnp.zeros(()),
            terminated=jnp.zeros((), dtype=bool),
            truncated=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo(boundary_obs=jnp.zeros_like(obs)) # dummy
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: CraftaxEnvState, action: jax.Array) -> tuple[Transition, EnvInfo, CraftaxEnvState]:
        next_obs, next_env_state, reward, done, info = self.env.step(key, env_state, action, self.env_params)
        current_time = self._get_current_time(next_env_state)
        truncated = current_time >= self.max_episode_length
        # Infer termination from the done and truncation flags since gymnax couples the two.
        # Note: Ambiguity remains when truncation is True. However, the probability of
        # false positives (i.e., actual termination coinciding exactly with truncation) is low.
        terminated = jnp.logical_and(done, jnp.logical_not(truncated))
        transition = Transition(
            action=action,
            next_obs=next_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        env_info = EnvInfo(info=info)
        return transition, env_info, next_env_state

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
