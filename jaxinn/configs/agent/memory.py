import math
import warnings
from dataclasses import dataclass, field
from typing import Literal

from jaxtyping import PyTree, DTypeLike

from .base import Base, Resolvable


@dataclass
class MemoryConfig(Resolvable, Base):
    capacity: int | tuple[int, ...] = 1000000
    device: Literal['cpu', 'gpu'] = field(default='gpu', metadata={"transient": True})

    num_seeds: int | None = field(default=None, init=False)
    obs_shape: PyTree[tuple[int, ...]] | None = field(default=None, init=False)
    obs_dtype: PyTree[DTypeLike] | None = field(default=None, init=False)
    action_shape: PyTree[tuple[int, ...]] | None = field(default=None, init=False)
    action_dtype: PyTree[DTypeLike] | None = field(default=None, init=False)

    def _resolve(self, ctx: dict) -> None:
        # If device is cpu, pre-allocate for all seeds upfront
        # Otherwise vmap handles this
        if self.device == "cpu":
            self.num_seeds = ctx.get("num_seeds", 1)

        # Set obs and action shape and dtype
        try:
            self.obs_shape = ctx["observation_space"].shape
            self.obs_dtype = ctx["observation_space"].dtype
            self.action_shape = ctx["action_space"].shape
            self.action_dtype = ctx["action_space"].dtype
        except KeyError as e:
            raise ValueError(f"Creating Memory requires knowing observation and action metadata. Missing key: {e}")

        self._resolve_capacity(ctx)

    def _resolve_capacity(self, ctx: dict) -> None:
        raise NotImplementedError("Subclasses must implement _resolve_capacity()")

    def _warn_if_overcapacity(self, original_capacity: int, rectified_capacity: int, action_repeat: int) -> None:
        if 0 < rectified_capacity < original_capacity:
            warnings.warn(
                f"Memory is overcapacity ({original_capacity}). "
                f"Truncated to actual number of environment steps: {rectified_capacity} "
                f"(relative to action_repeat={action_repeat}).",
                category=UserWarning,
                stacklevel=3,
            )


@dataclass
class FlattenedMemoryConfig(MemoryConfig):
    def _resolve_capacity(self, ctx: dict) -> None:
        if isinstance(self.capacity, tuple):
            raise ValueError(
                f"Flattened memory requires an integer capacity, "
                f"but received a tuple: {self.capacity}"
            )

        num_environment_steps = ctx.get("num_environment_steps", 0)
        prefill_steps = ctx.get("num_prefill_episodes", 0) * ctx.get("episode_length", 0)
        action_repeat = ctx.get("action_repeat", 1)
        max_steps = (num_environment_steps + prefill_steps) // action_repeat

        original_capacity = self.capacity
        rectified_capacity = min(original_capacity, max_steps)

        self.capacity = rectified_capacity
        self._warn_if_overcapacity(original_capacity, rectified_capacity, action_repeat)


@dataclass
class BatchedMemoryConfig(MemoryConfig):
    needs_boundary_obs: bool | None = field(default=None, init=False)

    def _resolve(self, ctx: dict) -> None:
        super()._resolve(ctx)
        self.needs_boundary_obs = not ctx.get("next_step_autoreset", False)

    def _resolve_capacity(self, ctx: dict) -> None:
        num_envs = ctx.get("num_envs", 1)
        action_repeat = ctx.get("action_repeat", 1)
        episode_length = ctx.get("episode_length", 0)
        train_interval = ctx.get("train_interval", 0)
        num_environment_steps = ctx.get("num_environment_steps", 0)

        if not isinstance(self.capacity, tuple) or len(self.capacity) != 2 or self.capacity[1] != num_envs:
            num_episodes_per_learn = train_interval // episode_length
            actual_episode_length = num_episodes_per_learn * (episode_length // num_envs // action_repeat) + 1
            self.capacity = (actual_episode_length, num_envs)

        original_capacity = math.prod(self.capacity)
        max_steps = num_environment_steps // action_repeat

        rectified_capacity = min(original_capacity, max_steps)
        self.capacity = (rectified_capacity // num_envs, num_envs)

        self._warn_if_overcapacity(original_capacity, rectified_capacity, action_repeat)


@dataclass
class UniformMemoryConfig(FlattenedMemoryConfig):
    pass


@dataclass
class PrioritizedMemoryConfig(FlattenedMemoryConfig):
    pass


@dataclass
class EpisodicMemoryConfig(FlattenedMemoryConfig):
    max_sequence_length: int = 0
    prioritize_ends: bool = True


MemoryUnion = UniformMemoryConfig | BatchedMemoryConfig | PrioritizedMemoryConfig | EpisodicMemoryConfig
