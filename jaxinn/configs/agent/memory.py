import warnings
from dataclasses import dataclass
from typing import Literal

from .base import Base, Resolvable


# Memory
@dataclass
class Memory(Resolvable, Base):
    capacity: int = 1000000
    device: Literal['cpu', 'gpu'] = 'gpu'
    type: Literal['uniform', 'prioritized', 'batched'] = 'uniform'

    def _resolve(self, ctx: dict) -> None:
        if self.device == "cpu":
            self.num_seeds = ctx["num_seeds"] # pre-allocate for all seeds upfront
        else:
            self.num_seeds = None             # vmap handles this

        num_environment_steps = ctx.get("num_environment_steps", 0)
        original_capacity = self.capacity
        self.capacity = min(original_capacity, num_environment_steps)

        if 0 < self.capacity < original_capacity:
            warnings.warn(
                f"Memory is overcapacity ({original_capacity}). "
                f"Truncated to actual number of environment steps: {self.capacity}.",
                category=UserWarning,
                stacklevel=2,
            )
