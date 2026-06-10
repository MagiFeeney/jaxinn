import abc

from jaxtyping import PRNGKeyArray
import equinox as eqx

from .utils import differentiable


class Loss(eqx.Module, abc.ABC):
    pass


class WorldLoss:
    @eqx.filter_value_and_grad(has_aux=True)
    @differentiable(['world'])
    def world_loss_fn(self, *args, key: PRNGKeyArray):
        pass


class ActorLoss:
    @eqx.filter_value_and_grad(has_aux=True)
    @differentiable(['actor'])
    def actor_loss_fn(self, *args, key: PRNGKeyArray):
        pass


class CriticLoss:
    @eqx.filter_value_and_grad(has_aux=True)
    @differentiable(['critic'])
    def critic_loss_fn(self, *args, key: PRNGKeyArray):
        pass


class ActorCriticLoss:
    @eqx.filter_value_and_grad(has_aux=True)
    @differentiable(['actor_critic'])
    def actor_critic_loss_fn(self, *args, key: PRNGKeyArray):
        pass
