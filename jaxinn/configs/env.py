from dataclasses import dataclass, field
from typing import Any, Literal

from .base import Base


# Environment
@dataclass
class Wrapper(Base):
    num_envs: int = 1                # num. of envs for collecting data
    action_repeat: int = 2
    use_one_hot_action: bool = False
    target_shape: tuple[int, int] | None = None
    normalize_obs: bool = False
    normalize_reward: bool = False
    reward_transform: Literal["tanh", "sign", "symlog"] | None = None


@dataclass
class Env(Base):
    env_id: str = "gymnax/DeepSea-bsuite"
    creation: dict[str, int | float | bool | str] = field(default_factory=dict)
    wrapper: Wrapper = field(default_factory=Wrapper)
    separated: bool = False
    prefill_mode: Literal['batched', 'serial', 'external'] = "serial" # TODO: handle external dataset


@dataclass
class EnvSelector:
    """The entry for selecting the env specific config, which requires knowing the env_name in the first place."""
    env: Env = field(default_factory=Env)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
