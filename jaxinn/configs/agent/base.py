from dataclasses import dataclass, field

from jaxinn.configs.base import Resolvable, Base, Memory, _sync_statics


# Base class
@dataclass
class Agent(Resolvable, Base):
    memory: Memory = field(default_factory=Memory)

    def _resolve(self, ctx: dict) -> None:
        _sync_statics(self)
