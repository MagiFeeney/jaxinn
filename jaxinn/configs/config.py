from dataclasses import dataclass, field
from typing import Literal

from .base import Base
from .env import Env
from .agent import AgentUnion, DreamerAgentConfig


# Exploration
@dataclass
class Exploration(Base):
    num_environment_steps: int = 1000000
    num_prefill_episodes: int = 5
    eval_interval: int = 10000
    train_interval: int = 1000
    train_iterations: int = 100
    pretrain_iterations: int = 0
    episode_length: int = 1000
    num_eval_episodes: int = 10
    action_noise: float = 0.3
    restart: bool = True


# Logger
@dataclass
class Logger(Base):
    log_dir: str | None = None
    backend: str = "tensorboard"
    shaded_method: Literal["std", "se", "ci", "iqr"] = "std"
    aggregate_keywords: tuple[str, ...] = ("eval",)


# console
@dataclass
class Config(Base):
    agent: AgentUnion = field(default_factory=DreamerAgentConfig)
    env: Env = field(default_factory=Env)
    exploration: Exploration = field(default_factory=Exploration)    # Trainer particulars
    logger: Logger = field(default_factory=Logger)

    axis_name: str = "p"             # pmap axis name
    seed: int = 42                   # master seed
    num_seeds: int = 50              # num. of agents

    save_model_path: str = ""
    load_model_path: str = ""
