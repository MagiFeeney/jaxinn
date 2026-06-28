import math
import warnings
from dataclasses import dataclass, field
from typing import Tuple, Union, Literal, Optional

from jaxtyping import PyTree, DTypeLike

from .base import Base, Resolvable


# Memory
@dataclass
class Memory(Resolvable, Base):
    capacity: Union[int, Tuple[int, ...]] = 1000000
    device: Literal['cpu', 'gpu'] = 'gpu'
    type: Literal['uniform', 'prioritized', 'batched'] = 'uniform'

    num_seeds: Optional[int] = field(default=None, init=False)
    obs_shape: Optional[PyTree[Tuple[int, ...]]] = field(default=None, init=False)
    obs_dtype: Optional[PyTree[DTypeLike]] = field(default=None, init=False)
    action_shape: Optional[PyTree[Tuple[int, ...]]] = field(default=None, init=False)
    action_dtype: Optional[PyTree[DTypeLike]] = field(default=None, init=False)

    def _resolve(self, ctx: dict) -> None:
        # If device is cpu, pre-allocate for all seeds upfront
        # Otherwise vmap handles this
        if self.device == "cpu":
            self.num_seeds = ctx.get("num_seeds", 1)

        # Set obs and action shape and dtype
        try:
            self.obs_shape = ctx["obs_shape"]
            self.obs_dtype = ctx["obs_dtype"]
            self.action_shape = ctx["action_shape"]
            self.action_dtype = ctx["action_dtype"]
        except KeyError as e:
            raise ValueError(f"Creating Memory requires knowing observation and action metadata. Missing key: {e}")

        # Handle capacity
        num_envs = ctx.get("num_envs", 1)
        action_repeat = ctx.get("action_repeat", 1)
        episode_length = ctx.get("episode_length", 0)
        num_environment_steps = ctx.get("num_environment_steps", 0)

        is_tuple = isinstance(self.capacity, tuple)

        if self.type == "batched":
            if not is_tuple or len(self.capacity) != 2 or self.capacity[1] != num_envs:
                actual_episode_length = (episode_length // num_envs // action_repeat) + 1
                self.capacity = (actual_episode_length, num_envs)
                is_tuple = True
        elif is_tuple:
            raise ValueError(
                f"Memory type '{self.type}' requires an integer capacity, "
                f"but received a tuple: {self.capacity}"
            )

        original_capacity = math.prod(self.capacity) if is_tuple else self.capacity
        max_steps = num_environment_steps // action_repeat

        rectified_capacity = min(original_capacity, max_steps)

        if self.type == "batched":
            self.capacity = (rectified_capacity // num_envs, num_envs)
        else:
            self.capacity = rectified_capacity

        if 0 < rectified_capacity < original_capacity:
            warnings.warn(
                f"Memory is overcapacity ({original_capacity}). "
                f"Truncated to actual number of environment steps: {rectified_capacity} "
                f"(relative to action_repeat={action_repeat}).",
                category=UserWarning,
                stacklevel=2,
            )
