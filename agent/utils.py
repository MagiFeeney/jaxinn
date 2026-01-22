import jax
import equinox as eqx
import functools

from typing import Union, List


def differentiable(fields: Union[str, List[str]]):
    """
    Decorator that partitions the first argument (the module/agent)
    so that only the specified sub-modules (diff_names) are differentiable.
    """
    if isinstance(fields, str):
        fields = [fields]

    def wrapper(func):
        @functools.wraps(func)
        def body(agent, *args, **kwargs):
            diff, static = partition(agent)
            static = jax.lax.stop_gradient(static)
            return call(diff, static, *args, **kwargs)

        def call(diff_part, static_part, *args, **kwargs):
            agent = eqx.combine(diff_part, static_part)
            return func(agent, *args, **kwargs)

        def partition(agent):
            mask = jax.tree.map(lambda _: False, agent)
            def selector(a):
                return tuple(getattr(a, field) for field in fields)
            mask = eqx.tree_at(selector, mask, replace_fn=lambda _: True)
            return eqx.partition(agent, mask)
        return body
    return wrapper
