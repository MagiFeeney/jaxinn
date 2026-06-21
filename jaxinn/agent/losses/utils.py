import jax
import equinox as eqx
import functools

from typing import Union, Sequence


def differentiable(fields: Union[str, Sequence[str]]):
    """
    Decorator that selects differentiable sub-modules by input names.
    """
    if isinstance(fields, str):
        fields = (fields,)
    fields = tuple(fields)

    def selector(a):
        return tuple(getattr(a, field) for field in fields)

    def wrapper(func):
        @functools.wraps(func)
        def body(agent, *args, **kwargs):
            stopped_agent = jax.lax.stop_gradient(agent)
            mixed_agent = eqx.tree_at(
                selector,
                stopped_agent,
                selector(agent)
            )
            return func(mixed_agent, *args, **kwargs)
        return body
    return wrapper
