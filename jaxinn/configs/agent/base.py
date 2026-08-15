from dataclasses import dataclass, field

from jaxinn.configs.base import Resolvable, Base, _sync_statics

from .memory import MemoryUnion, UniformMemoryConfig


# Base class
@dataclass
class AgentConfig(Resolvable, Base):
    memory: MemoryUnion = field(default_factory=UniformMemoryConfig)

    def _resolve(self, ctx: dict) -> None:
        _sync_statics(self)
