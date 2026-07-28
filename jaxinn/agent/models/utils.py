import math
from typing import Any, Callable, Dict, Union, Optional, Literal, Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree
import equinox as eqx
from equinox._module import Static
import distrax

RegisteredItem = Union[str, Callable]
Activation = RegisteredItem
Dtype = RegisteredItem


ACTIVATIONS = {
    "relu": jnn.relu,
    "elu": jnn.elu,
    "silu": jnn.silu,
    "gelu": jnn.gelu,
    "mish": jnn.mish,
    "tanh": jnn.tanh,
    "sigmoid": jnn.sigmoid,
}


DTYPES = {
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
    "float64": jnp.float64,
}


def _create_getter(registry: Dict[str, Any], entity_name: str) -> Callable[[RegisteredItem], Callable]:
    """Generates a getter function for a specific registry."""
    def getter(name_or_fn: RegisteredItem) -> Callable:
        if isinstance(name_or_fn, str):
            try:
                return registry[name_or_fn.lower()]
            except KeyError:
                raise ValueError(f"Unknown {entity_name}: {name_or_fn}")
        elif callable(name_or_fn):
            return name_or_fn
        else:
            raise TypeError(f"Expected str or callable, got {type(name_or_fn)}")
    return getter


get_activation_fn = _create_getter(ACTIVATIONS, "activation")
get_precision_fn = _create_getter(DTYPES, "dtype")


is_shape_leaf = lambda x: isinstance(x, tuple) and all(isinstance(i, int) for i in x)


def get_flatten_size(shape):
    return sum(math.prod(l) for l in jax.tree.leaves(shape, is_leaf=is_shape_leaf))


class StaticCallable(eqx.Module):
    fn: Callable[[Array], Array] = eqx.field(static=True)

    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array:
        return self.fn(x)


class FixedDistrax(eqx.Module):
    cls: Callable = eqx.field(static=True)
    args: PyTree[Any]
    kwargs: PyTree[Any]

    def __init__(self, cls: Callable, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def _resolve(self, x):
        return jax.tree.map(
            lambda leaf: (
                leaf.dist if isinstance(leaf, FixedDistrax)
                else leaf.value if isinstance(leaf, Static)
                else leaf
            ),
            x,
            is_leaf=lambda leaf: isinstance(leaf, (FixedDistrax, Static)),
        )

    @property
    def dist(self):
        resolved_args = self._resolve(self.args)
        resolved_kwargs = self._resolve(self.kwargs)
        return self.cls(*resolved_args, **resolved_kwargs)

    def __getattr__(self, name):
        if hasattr(self.dist, name):
            return getattr(self.dist, name)
        return getattr(self.dist.distribution, name)


class Composer(eqx.Module):
    cls: Callable = eqx.field(static=True)
    factory_args: tuple
    factory_kwargs: dict

    def __call__(self, *runtime_args, **runtime_kwargs):
        def _resolve_factory(leaf):
            if _is_factory(leaf):
                return leaf(*runtime_args, **runtime_kwargs)
            return leaf

        resolved_args, resolved_kwargs = jax.tree.map(
            _resolve_factory,
            (self.factory_args, self.factory_kwargs),
            is_leaf=_is_factory
        )
        return FixedDistrax(self.cls, *resolved_args, **resolved_kwargs)


class FixedFactory(eqx.Module):
    cls: Callable = eqx.field(static=True)

    def __call__(self, *args, **kwargs):
        leaves, _ = jax.tree.flatten((args, kwargs), is_leaf=_is_factory)
        if any(_is_factory(leaf) for leaf in leaves):
            return Composer(self.cls, args, kwargs)        # Delay instantiation
        else:
            return FixedDistrax(self.cls, *args, **kwargs) # Primitive fires with parameters


FactoryLike = Composer | FixedFactory


def _is_factory(x: Any) -> bool:
    return isinstance(x, FactoryLike)


class ProxyDistrax:
    def __call__(self, module):
        """
        Wrap a custom distrax-compatible callable (function or class).
        """
        if not callable(module):
            raise TypeError("ProxyDistrax can only wrap callables")

        return FixedFactory(module)

    def __getattr__(self, name):
        """
        Wrap a distrax method directly.
        """
        attr = getattr(distrax, name)

        if callable(attr):
            return FixedFactory(attr)

        return attr


dx = ProxyDistrax()


def make_mlp(
        input_size: int,
        hidden_size: Union[int, list[int]],
        output_size: int,
        activation: Union[str, Callable, StaticCallable],
        num_layers: Optional[int] = None,
        layer_norm: Optional[Literal['all', 'input', 'output', 'first', 'last']] = None,
        *,
        key: PRNGKeyArray
) -> eqx.nn.Sequential:
    if isinstance(hidden_size, int):
        assert num_layers is not None and num_layers >= 0, (
            "When hidden_size is an integer, num_layers must be specified "
            "and non-negative."
        )
        hidden_size = [hidden_size] * num_layers

    sizes = [input_size] + hidden_size + [output_size]
    layer_norms = [False] * len(sizes)

    if layer_norm == "all":
        layer_norms[1:-1] = [True] * (len(sizes) - 2)
    elif layer_norm == "input":
        layer_norms[0] = True
    elif layer_norm == "output":
        layer_norms[-1] = True
    elif layer_norm == "first":
        layer_norms[1] = True
    elif layer_norm == "last":
        layer_norms[-2] = True

    layers = []
    keys = jax.random.split(key, len(sizes) - 1)

    if isinstance(activation, str):
        activation = get_activation_fn(activation)
    if not isinstance(activation, StaticCallable):
        activation = StaticCallable(activation)

    if layer_norms[0]:          # Pre-norm
        layers.append(eqx.nn.LayerNorm(sizes[0]))

    for i in range(len(sizes) - 1):
        layers.append(eqx.nn.Linear(sizes[i], sizes[i+1], key=keys[i]))

        if layer_norms[i+1]:    # Pre-activation norm
            layers.append(eqx.nn.LayerNorm(sizes[i+1]))

        if i < len(sizes) - 2:
            layers.append(activation)

    return eqx.nn.Sequential(layers)


def make_cnn(
        in_channels: int,
        num_spatial_dims: int,
        activation: str | Callable | StaticCallable,
        kernel_size: int | Sequence[int] = 4,
        depth: int | Sequence[int] = 32,
        depth_factor: int | None = 2,
        stride: int | Sequence[int] = 2,
        padding: str | int | Sequence[int] | Sequence[tuple[int, int]] = "SAME",
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = 'ZEROS',
        dtype: str = "float32",
        num_layers: Optional[int] = None,
        *,
        key: PRNGKeyArray
) -> eqx.nn.Sequential:
    potential_sequences = [
        ("kernel_size", kernel_size),
        ("depth", depth),
        ("stride", stride),
        ("padding", padding),
        ("dilation", dilation)
    ]

    sequence_args = {name: val for name, val in potential_sequences if isinstance(val, (list, tuple))}

    if sequence_args:
        seq_lengths = {name: len(val) for name, val in sequence_args.items()}
        unique_lengths = set(seq_lengths.values())

        if len(unique_lengths) > 1:
            raise ValueError(
                f"Conflicting sequence lengths provided! Your sequences imply "
                f"different numbers of layers: {seq_lengths}"
            )

        num_layers = unique_lengths.pop()
    else:
        if num_layers is None:
            raise ValueError(
                "`num_layers` must be specified when kernel_size, depth, stride, "
                "etc., are all scalar values."
            )

    assert num_layers >= 0, "`num_layers` must be non-negative."

    kernel_size = [kernel_size] * num_layers if isinstance(kernel_size, int) else kernel_size
    stride = [stride] * num_layers if isinstance(stride, int) else stride
    dilation = [dilation] * num_layers if isinstance(dilation, int) else dilation
    padding = [padding] * num_layers if isinstance(padding, (str, int)) else padding

    if isinstance(depth, int):
       if depth_factor is None:
           raise ValueError("When `depth` is an integer, the `depth_factor` cannot be None")
       depth = [depth * depth_factor**i for i in range(num_layers)]

    sizes = [in_channels] + depth

    if isinstance(activation, str):
        activation = get_activation_fn(activation)
    if not isinstance(activation, StaticCallable):
        activation = StaticCallable(activation)

    layers = []

    # Conv layers
    keys = jax.random.split(key, len(sizes) - 1)
    for i in range(len(sizes) - 1):
        layers.append(
            eqx.nn.Conv(
                num_spatial_dims,
                in_channels=sizes[i],
                out_channels=sizes[i+1],
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i],
                dilation=dilation[i],
                groups=groups,
                use_bias=use_bias,
                padding_mode=padding_mode,
                dtype=dtype,
                key=keys[i]
            )
        )

        layers.append(activation)

    # Flatten it
    layers.append(StaticCallable(jnp.ravel))

    return eqx.nn.Sequential(layers)


def make_cnn_transposed(
        in_channels: int,
        out_channels: int,
        num_spatial_dims: int,
        activation: str | Callable | StaticCallable,
        kernel_size: int | Sequence[int] = 4,
        depth: int | Sequence[int] = 32,
        depth_factor: int | None = 2,
        stride: int | Sequence[int] = 2,
        padding: str | int | Sequence[int] | Sequence[tuple[int, int]] = "SAME",
        output_padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = 'ZEROS',
        dtype: str = "float32",
        num_layers: Optional[int] = None,
        *,
        key: PRNGKeyArray
) -> eqx.nn.Sequential:
    potential_sequences = [
        ("kernel_size", kernel_size),
        ("depth", depth),
        ("stride", stride),
        ("padding", padding),
        ("output_padding", output_padding),
        ("dilation", dilation)
    ]

    sequence_args = {name: val for name, val in potential_sequences if isinstance(val, (list, tuple))}

    if sequence_args:
        seq_lengths = {name: len(val) for name, val in sequence_args.items()}
        unique_lengths = set(seq_lengths.values())

        if len(unique_lengths) > 1:
            raise ValueError(
                f"Conflicting sequence lengths provided! Your sequences imply "
                f"different numbers of layers: {seq_lengths}"
            )

        num_layers = unique_lengths.pop()
    else:
        if num_layers is None:
            raise ValueError(
                "`num_layers` must be specified when kernel_size, depth, stride, "
                "etc., are all scalar values."
            )

    assert num_layers >= 0, "`num_layers` must be non-negative."

    kernel_size = [kernel_size] * num_layers if isinstance(kernel_size, int) else kernel_size
    stride = [stride] * num_layers if isinstance(stride, int) else stride
    dilation = [dilation] * num_layers if isinstance(dilation, int) else dilation
    padding = [padding] * num_layers if isinstance(padding, (str, int)) else padding
    output_padding = [output_padding] * num_layers if isinstance(output_padding, (str, int)) else output_padding

    if isinstance(depth, int):
       if depth_factor is None:
           raise ValueError("When `depth` is an integer, the `depth_factor` cannot be None")
       depth = [depth * depth_factor**i for i in reversed(range(num_layers - 1))]

    sizes = [in_channels] + depth + [out_channels]

    if isinstance(activation, str):
        activation = get_activation_fn(activation)
    if not isinstance(activation, StaticCallable):
        activation = StaticCallable(activation)

    layers = []

    # Conv layers
    keys = jax.random.split(key, len(sizes) - 1)
    for i in range(len(sizes) - 1):
        layers.append(
            eqx.nn.ConvTranspose(
                num_spatial_dims,
                in_channels=sizes[i],
                out_channels=sizes[i+1],
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i],
                output_padding=output_padding[i],
                dilation=dilation[i],
                groups=groups,
                use_bias=use_bias,
                padding_mode=padding_mode,
                dtype=dtype,
                key=keys[i]
            )
        )

        if i < len(sizes) - 2:
            layers.append(activation)

    return eqx.nn.Sequential(layers)


def apply_init(
        model: eqx.Module,
        weight_init: Callable,
        bias_init: Callable = jax.nn.initializers.constant(0.0),
        output_weight_init: Optional[Callable] = None,
        output_bias_init: Optional[Callable] = None,
        *,
        key: PRNGKeyArray
) -> eqx.Module:
    is_target = lambda x: isinstance(x, (eqx.nn.Linear, eqx.nn.Conv2d))
    get_layers = lambda m: [x for x in jax.tree.leaves(m, is_leaf=is_target) if is_target(x)]

    if output_weight_init is None:
        output_weight_init = weight_init

    if output_bias_init is None:
        output_bias_init = bias_init

    def init_layer(layer: eqx.Module, k: PRNGKeyArray, is_output: bool) -> eqx.Module:
        kw, kb = jax.random.split(k)

        w_fn = output_weight_init if is_output else weight_init
        b_fn = output_bias_init if is_output else bias_init

        # Init weight
        weight = layer.weight
        if isinstance(layer, eqx.nn.Conv2d):
            out_c, in_c, height, width = weight.shape
            hwio = w_fn(kw, (height, width, in_c, out_c), weight.dtype) # JAX's expected layout: HWIO
            new_w = jnp.transpose(hwio, (3, 2, 0, 1))                   # Equinox's: OIHW
        else:
            new_w = w_fn(kw, weight.shape, weight.dtype)

        # Init bias
        bias = layer.bias
        new_b = b_fn(kb, bias.shape, bias.dtype)

        return eqx.tree_at(lambda l: (l.weight, l.bias), layer, (new_w, new_b))

    layers = get_layers(model)
    keys = jax.random.split(key, len(layers))
    new_layers = [
        init_layer(l, k, is_output=(i == len(layers) - 1))
        for i, (l, k) in enumerate(zip(layers, keys))
    ]

    return eqx.tree_at(get_layers, model, new_layers)
