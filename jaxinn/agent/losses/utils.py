import jax
import equinox as eqx
import functools

from typing import Union, List, Tuple


def differentiable(fields: Union[str, List[str]]):
    """
    Decorator that selects differentiable sub-modules by input names.
    """
    if isinstance(fields, str):
        fields = [fields]

    def wrapper(func):
        @functools.wraps(func)
        def body(agent, *args, **kwargs):
            stopped_agent = jax.tree.map(
                lambda x: jax.lax.stop_gradient(x) if eqx.is_inexact_array(x) else x,
                agent
            )

            def selector(a):
                return tuple(getattr(a, field) for field in fields)

            mixed_agent = eqx.tree_at(
                selector,
                stopped_agent,
                selector(agent)
            )
            return func(mixed_agent, *args, **kwargs)
        return body
    return wrapper
