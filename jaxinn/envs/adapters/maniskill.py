from typing import Any, Dict, Optional, Tuple

import torch
from torch2jax import torch2jax
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import gymnasium as gym
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from jaxinn.structs import Transition

from .gymnasium import GymnasiumVmapMixIn, gymnasium_space_to_jaxinn_space
from ..environment import Environment, EnvInfo, EnvState


class Torch2JaxEnvWrapper:
    """Wraps a PyTorch RL environment of Gymnasium interface as a JAX-compiled XLA operation via torch2jax."""

    def __init__(self, env, seed: int | None = None):
        self.env = env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        def torch_step(action: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            obs, reward, terminated, truncated, info = self.env.step(action)
            final_obs = info.get("final_observation", jax.tree.map(torch.zeros_like, obs))
            return obs, reward, terminated, truncated, final_obs

        def torch_reset(seed: torch.Tensor) -> torch.Tensor:
            seed_val = int(seed.item())
            if seed_val < 0:
                obs, _ = self.env.reset()
            else:
                obs, _ = self.env.reset(seed=seed_val)
            return obs

        dummy_seed_torch = torch.tensor(-1 if seed is None else seed, dtype=torch.int32, device=self.device)
        dummy_action_torch = torch.zeros(env.action_space.shape, dtype=torch.float32, device=self.device)

        self._jax_reset = torch2jax(
            torch_reset,
            dummy_seed_torch
        )

        self._jax_step = torch2jax(
            torch_step,
            dummy_action_torch
        )

    def _reset(self, seed: int | None = None):
        seed_val = -1 if seed is None else seed
        seed_jax = jnp.array(seed_val, dtype=jnp.int32)
        return self._jax_reset(seed_jax)

    def _step(self, action: jax.Array):
        return self._jax_step(action)

    def __getattr__(self, name):
        return getattr(self.env, name)


class ManiSkill(GymnasiumVmapMixIn, Environment):
    def __init__(
            self,
            env: Torch2JaxEnvWrapper,
            env_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(env, env_params)

    @classmethod
    def create(
            cls,
            env_name: str,
            num_envs: int,
            obs_mode: str = "rgb",
            render_mode: str = "rgb_array",
            sim_backend: str = "physx_cuda",
            include_state: bool = False,
            seed: Optional[int] = None,
            **kwargs
    ) -> "ManiSkill":
        env = gym.make(
            env_name,
            num_envs=num_envs,
            obs_mode=obs_mode,
            render_mode=render_mode,
            sim_backend=sim_backend,
            **kwargs
        )

        rgb = obs_mode in {"rgb", "rgb+depth", "rgbd"}
        depth = obs_mode in {"depth", "rgb+depth", "rgbd"}

        if rgb or depth:        # TODO: concat rgb and depth into a single array for obs_encoder
            env = FlattenRGBDObservationWrapper(env, rgb=rgb, depth=depth, state=include_state)

        env = ManiSkillVectorEnv(env, num_envs)
        env = Torch2JaxEnvWrapper(env, seed=seed)

        return cls(env, env_params={
            "capacity": num_envs,
            "obs_mode": obs_mode,
            "rgb": rgb,
            "depth": depth,
            "include_state": include_state,
            "seed": seed,
            **kwargs,
        })

    def reset(self, key: PRNGKeyArray) -> Tuple[Transition, EnvInfo, jax.Array]:
        obs = self.v_reset(self, key)
        transition = Transition(
            action = jax.tree.map(
                lambda shape, dtype: jnp.zeros(shape, dtype=dtype),
                self.action_space.shape,
                self.action_space.dtype,
                is_leaf=lambda x: isinstance(x, tuple)
            ),
            next_obs=obs,
            reward=jnp.zeros(()),
            terminated=jnp.zeros((), dtype=bool),
            truncated=jnp.zeros((), dtype=bool),
        )
        env_info = EnvInfo(terminal_observation=jax.tree.map(jnp.zeros_like, obs))
        env_state = EnvState()
        return transition, env_info, env_state

    def step(self, key: PRNGKeyArray, env_state: jax.Array, action: jax.Array) -> Tuple[Transition, EnvInfo, jax.Array]:
        # VmapMixIn → torch2jax
        next_obs, reward, terminated, truncated, final_obs = self.v_step(self, key, action)
        transition = Transition(
            action=action,
            next_obs=next_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        env_info = EnvInfo(terminal_observation=final_obs)
        next_env_state = EnvState()
        return transition, env_info, next_env_state

    @property
    def observation_space(self):
        observation_space = self.env.single_observation_space if self.capacity is not None and hasattr(self.env, "single_observation_space") else self.env.observation_space
        return gymnasium_space_to_jaxinn_space(observation_space)

    @property
    def action_space(self):
        action_space = self.env.single_action_space if (self.capacity is not None) and hasattr(self.env, "single_action_space") else self.env.action_space
        return gymnasium_space_to_jaxinn_space(action_space)

    @property
    def max_episode_length(self) -> int:
        return gym_utils.find_max_episode_steps_value(self.env)
