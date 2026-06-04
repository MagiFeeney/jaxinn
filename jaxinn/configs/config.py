from dataclasses import dataclass, field
from .base import Base, Env, Exploration, Logger
from .agent import AgentUnion, DreamerAgent


# console
@dataclass
class Config(Base):
    agent: AgentUnion = field(default_factory=DreamerAgent)
    env: Env = field(default_factory=Env)
    exploration: Exploration = field(default_factory=Exploration)    # Trainer particulars
    logger: Logger = field(default_factory=Logger)

    axis_name: str = "p"             # pmap axis name
    seed: int = 42                   # master seed
    num_seeds: int = 50              # num. of agents

    save_model_path: str = ""
    load_model_path: str = ""
