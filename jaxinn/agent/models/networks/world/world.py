import jax
from jaxtyping import PRNGKeyArray
import equinox as eqx

from jaxinn.configs.model import WorldConfig

from .perception import Perception
from .representation import Representation
from .transition import Transition
from .reward import Reward


class World(eqx.Module):
    perception: Perception
    representation: Representation
    transition: Transition
    reward: Reward

    @classmethod
    def create(cls, config: WorldConfig, *, key: PRNGKeyArray):
        key_perception, key_representation, key_transition, key_reward = jax.random.split(key, 4)
        perception = Perception.create(config.perception, key=key_perception)
        representation = Representation(**config.representation(), key=key_representation)
        transition = Transition(**config.transition(), key=key_transition)
        reward = Reward(**config.reward(), key=key_reward)
        return cls(
            perception=perception,
            representation=representation,
            transition=transition,
            reward=reward
        )
