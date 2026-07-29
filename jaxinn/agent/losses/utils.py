import jax
import equinox as eqx
import functools

from collections.abc import Sequence, Callable


def differentiable(fields: str | Sequence[str]):
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


def value_and_grad(
        loss_fn: Callable | None = None,
        *,
        fields: str | Sequence[str] | None = None,
        **grad_kwargs
):

    """
    A wrapper around `eqx.filter_value_and_grad` that uses `differentiable`
    to determine which arguments participate in differentiation.

    This function can be used either directly or as a decorator. When used
    as a decorator, all arguments must be specified explicitly. In contrast to `differentiable`, it additionally extracts gradients for the selected modules.
    """

    def wrapper(loss_fn):
        if fields is not None:
            _fields = (fields,) if isinstance(fields, str) else tuple(fields)
            def selector(a):
                return tuple(getattr(a, f) for f in _fields)
        else:
            selector = None

        diff_fn = differentiable(_fields)(loss_fn) if selector is not None else loss_fn
        grad_fn = eqx.filter_value_and_grad(diff_fn, **grad_kwargs)

        @functools.wraps(loss_fn)
        def selected_grad_fn(model, *args, **kwargs):
            out, grads = grad_fn(model, *args, **kwargs)
            grads = selector(grads)
            if len(grads) == 1:
                grads = grads[0]
            return out, grads

        return selected_grad_fn

    if loss_fn is not None:
        return wrapper(loss_fn)
    else:
        return wrapper
