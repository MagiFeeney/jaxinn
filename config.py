from dataclasses import dataclass, field, fields, asdict, is_dataclass
from typing import Tuple, Any, Optional, Literal, Dict, Union
from types import SimpleNamespace


class ConfigNamespace(SimpleNamespace):
    def keys(self):
        return vars(self).keys()

    def items(self):
        return vars(self).items()

    def __contains__(self, key):
        return key in vars(self)

    def __iter__(self):
        return iter(vars(self))

    def __getitem__(self, key):
        return vars(self)[key]

    def __repr__(self):
        items = [f"{k}={v}" for k, v in self.items()]
        return f"Namespace({', '.join(items)})"


@dataclass
class Resolvable:
    """Depth-first recursive resolution."""

    def resolve(self, ctx: dict) -> "Resolvable":
        """Check the child nodes."""
        for f in fields(self):
            child = getattr(self, f.name)
            if isinstance(child, Resolvable):
                child.resolve(ctx)
        self._resolve(ctx)
        return self

    def _resolve(self, ctx: dict) -> None:
        """Resolve current level."""
        pass


@dataclass
class Base:
    """
    Nested configuration management.

    If there is sub-config, return the arguments at the current level excluding those from sub-configs.
    """
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)     # Preserve hierarchy

    def __call__(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not is_dataclass(v)} # Sub-node is considered as next phase

    def update(self, updates: dict[str, Any] = None, **kwargs) -> None:
        """Update or add config attributes. Allows adding new attributes."""
        if updates is None:
            updates = kwargs
        else:
            updates.update(kwargs)

        for key, value in updates.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if is_dataclass(attr) and isinstance(value, dict):
                    attr.update(value)
                else:
                    setattr(self, key, value)
            else:
                # Add new attribute that doesn't exist
                setattr(self, key, value)


@dataclass
class Model(Base):
    def __call__(self):
        models = {k: v for k, v in vars(self).items() if isinstance(v, Model)}
        if len(models) == 0:
            return super().__call__()
        return ConfigNamespace(**models) # Dot notation compatibility


@dataclass
class ModelShared(Model):
    """Shared parameters across different models."""
    belief_size: int = 200
    state_size: int = 30


@dataclass
class PerceptionShared(Model):
    """Shared parameters across perception modules."""
    shape: Optional[Tuple[int, ...]] = None
    activation_function: str = "elu"


@dataclass
class OptimizerShared(Base):
    """Shared parameters across optimizers."""
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8


# World Model
## For pixel-based tasks
@dataclass
class CNNEncoder(PerceptionShared):
    embedding_size: int = 1024
    dtype: str = "bfloat16"


@dataclass
class CNNDecoder(ModelShared, PerceptionShared):
    dtype: str = "bfloat16"


## For state-based tasks
@dataclass
class LinearEncoder(PerceptionShared):
    hidden_size: Optional[int] = None
    embedding_size: Optional[int] = None
    num_layers: Optional[int] = None


@dataclass
class LinearDecoder(ModelShared, PerceptionShared):
    hidden_size: int = 300
    num_layers: int = 3


CONFIG_REGISTRY = {
    "cnn": (CNNEncoder, CNNDecoder),
    "linear": (LinearEncoder, LinearDecoder)
}


PIXEL_ARCHS = {"cnn"}
STATE_ARCHS = {"linear"}


def arch_to_domain(arch):
    if arch in PIXEL_ARCHS:
        return "pixel"
    elif arch in STATE_ARCHS:
        return "state"
    else:
        raise NotImplementedError(f"Architecture {arch} is currently not supported or classified.")


def make_perception_class(name, encoder_cls, decoder_cls):
    @dataclass
    class Perception(Resolvable, Model):
        encoder: encoder_cls = field(default_factory=encoder_cls)
        decoder: decoder_cls = field(default_factory=decoder_cls)
        domain: str = field(init=False, default=arch_to_domain(name))
        type: str = name

        def _resolve(self, ctx: dict) -> None:
            self.encoder.shape = ctx["obs_shape"]
            self.decoder.shape = ctx["obs_shape"]

    Perception.__name__ = f"{name.capitalize()}Perception"
    Perception.__qualname__ = Perception.__name__
    return Perception


PERCEPTION_REGISTRY = {
    name: make_perception_class(name, enc, dec)
    for name, (enc, dec) in CONFIG_REGISTRY.items()
}


PerceptionUnion = Union[tuple(PERCEPTION_REGISTRY.values())]


@dataclass
class Representation(Resolvable, ModelShared):
    embedding_size: Optional[int] = None
    hidden_size: int = 200
    activation_function: str = "elu"
    head_type: Literal['Normal', 'Categorical'] = "Normal"

    def _resolve(self, ctx: dict) -> None:
        perception_type = ctx.get("perception_type")
        if "embedding_size" not in ctx or perception_type is None:
            return

        obs_shape = ctx["obs_shape"]
        embedding_size = ctx["embedding_size"]

        if embedding_size:
            self.embedding_size = embedding_size
        elif perception_type == "linear" and len(obs_shape) == 1:
            self.embedding_size = obs_shape[0]
        else:
            raise NotImplementedError(
                f"Cannot infer embedding_size for perception={perception_type!r} "
                f"with obs_shape={obs_shape}"
            )


@dataclass
class Transition(Resolvable, ModelShared):
    hidden_size: int = 200
    activation_function: str = "elu"
    head_type: Literal['Normal', 'Categorical'] = "Normal"

    def _resolve(self, ctx: dict) -> None:
        self.action_size = ctx["action_size"]


@dataclass
class Reward(ModelShared):
    hidden_size: int = 300
    activation_function: str = "elu"
    head_type: str = "Isotropic Normal"
    min_std: float = 0.0


@dataclass
class WorldOptimizer(OptimizerShared):
    """Optimizer for world model."""
    lr: float = 6e-4
    max_norm: int = 100


@dataclass
class World(Resolvable, Model):
    perception: Optional[PerceptionUnion] = field(default=None)
    representation: Representation = field(default_factory=Representation)
    transition: Transition = field(default_factory=Transition)
    reward: Reward = field(default_factory=Reward)
    optimizer: WorldOptimizer = field(default_factory=WorldOptimizer)

    def _resolve(self, ctx: dict) -> None:
        obs_shape = ctx["obs_shape"]

        # If perception is not given, fall back to defaults
        if self.perception is None:
            name = "cnn" if len(obs_shape) > 1 else "linear"
            self.perception = PERCEPTION_REGISTRY[name]()
            self.perception.resolve(ctx)

        if len(obs_shape) == 1:
            assert self.perception.domain == "state", (
                f"1D obs requires a state module, got {self.perception.type!r}"
            )
        else:
            assert self.perception.domain == "pixel", (
                f"2D+ obs requires a pixel module, got {self.perception.type!r}"
            )

        ctx["embedding_size"] = self.perception.encoder.embedding_size
        ctx["perception_type"] = self.perception.type
        self.representation._resolve(ctx)


# Actor
@dataclass
class ActorOptimizer(OptimizerShared):
    """Optimizer for actor."""
    lr: float = 8e-5
    max_norm: int = 100


@dataclass
class Actor(Resolvable, ModelShared):
    hidden_size: int = 300
    activation_function: str = "elu"
    action_size: Optional[int] = None # Pass from the env params
    min_std: float = 0.0
    head_type: Literal['Tanh Normal', 'Beta', 'Categorical'] = "Tanh Normal"

    optimizer: ActorOptimizer = field(default_factory=ActorOptimizer)

    def _resolve(self, ctx: dict) -> None:
        self.action_size = ctx["action_size"]


# Critic
@dataclass
class CriticOptimizer(OptimizerShared):
    """Optimizer for critic."""
    lr: float = 8e-5
    max_norm: int = 100


@dataclass
class Critic(ModelShared):
    hidden_size: int = 300
    activation_function: str = "elu"
    action_size: Optional[int] = None
    min_std: float = 0.0
    head_type: Literal['Isotropic Normal', 'Normal'] = "Isotropic Normal"

    optimizer: CriticOptimizer = field(default_factory=CriticOptimizer)


# Memory
@dataclass
class Memory(Resolvable, Base):
    capacity: int = 1000000
    device: Literal['cpu', 'gpu'] = 'gpu'
    type: Literal['uniform', 'prioritized'] = "uniform"

    def _resolve(self, ctx: dict) -> None:
        if self.device == "cpu":
            self.num_seeds = ctx["num_seeds"] # pre-allocate for all seeds upfront
        else:
            self.num_seeds = None             # vmap handles this


# Environment
@dataclass
class Wrapper(Base):
    num_envs: int = 1                # num. of envs for collecting data
    target_shape: Optional[Tuple[int, int]] = None


@dataclass
class Env(Base):
    env_id: str = "gymnax/DeepSea-bsuite"
    creation: Dict[str, Union[int, float, bool, str]] = field(default_factory=dict)
    wrapper: Wrapper = field(default_factory=Wrapper)
    separated: bool = False


# Exploration
@dataclass
class Exploration(Base):
    num_environment_steps: int = 1000000
    num_prefill_episodes: int = 5
    eval_interval: int = 10000
    train_interval: int = 1000
    train_iterations: int = 100
    episode_length: int = 1000
    num_eval_episodes: int = 10
    action_noise: float = 0.3


# Optimization
@dataclass
class Optimization(Base):
    planning_horizon: int = 15
    discount_factor: float = 0.99
    uae_lambda: float = 0.95
    batch_size: int = 50
    chunk_size: int = 50


# All about agent
@dataclass
class Agent(Resolvable, Base):
    world: World = field(default_factory=World)
    actor: Actor = field(default_factory=Actor)
    critic: Critic = field(default_factory=Critic)
    memory: Memory = field(default_factory=Memory)
    optimization: Optimization = field(default_factory=Optimization)
    random_init: bool = False   # Whether to initialize the state by following a simple distribution


# console
@dataclass
class Config(Base):
    agent: Agent = field(default_factory=Agent)
    env: Env = field(default_factory=Env)
    exploration: Exploration = field(default_factory=Exploration)    # Trainer particulars

    seed: int = 42                   # master seed
    num_seeds: int = 50              # num. of agents
