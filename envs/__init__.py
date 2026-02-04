from .wrapper import Batched, AutoReset
from .environment import Transition, Environment


def make_env(env_name, **kwargs):
    import re
    domain, task = re.split(r'[-_/]', env_name, maxsplit=1)
    num_envs = kwargs.pop('num_envs', 1)
    env, env_params = create(domain, task, **kwargs)
    return Batched(env, env_params, num_envs)


__all__ = [
    'Batched',
    'AutoReset',
    'Transition',
    'Environment',
]
