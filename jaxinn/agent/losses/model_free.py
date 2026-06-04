from .base import Loss, ActorLoss, CriticLoss


class PPOLossMixIn(Loss, ActorLoss, CriticLoss):
    pass


class SACLossMixIn(Loss, ActorLoss, CriticLoss):
    pass
