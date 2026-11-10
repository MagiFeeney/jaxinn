"""A check that the device computes a linear layer as written.

Some jax releases compute `W @ x + b` wrongly on NVIDIA GPUs once the product is rewritten into a cuBLAS call: the result is silently off, on the GPU only (jax 0.10.0 to 0.11.0; see also https://github.com/jax-ml/jax/issues/39336).
A model trained that way differs from the one that acts and is evaluated. `check_device` scores a small encoder and decoder on the device and in numpy and raises if the two disagree.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

T, B, OBS, HID, N = 16, 16, 100, 256, 2048
WORKAROUND = "--xla_gpu_gemm_rewrite_size_threshold=100000000"


def _elu(z, xp):
    return xp.where(z > 0, z, xp.expm1(xp.minimum(z, 0.0)))


def _loss(p, obs):                                  # obs: [T, B, OBS]
    e = jax.vmap(jax.vmap(lambda o: _elu(p["W2"] @ _elu(p["W1"] @ o + p["b1"], jnp) + p["b2"], jnp)))(obs)
    out = jax.vmap(jax.vmap(lambda z: p["W3"] @ z + p["b3"]))(e)
    return ((out - obs) ** 2).mean()


def _batch(buffer, key):
    starts = jax.random.randint(key, (B,), 0, N - T)
    return buffer[starts[None, :] + jnp.arange(T)[:, None]]


def _reference(p, obs) -> float:
    p = {k: np.asarray(v, np.float64) for k, v in p.items()}
    obs = np.asarray(obs, np.float64)
    e = _elu(_elu(obs @ p["W1"].T + p["b1"], np) @ p["W2"].T + p["b2"], np)
    return float((((e @ p["W3"].T + p["b3"]) - obs) ** 2).mean())


def device_error() -> tuple[float, float]:
    """The loss of a fixed encoder and decoder on a fixed batch: on the device (jitted), and in numpy."""
    key_params, key_buffer, key_batch = jax.random.split(jax.random.PRNGKey(0), 3)
    k = jax.random.split(key_params, 8)
    weight = lambda key, o, i: jax.random.normal(key, (o, i)) / jnp.sqrt(i)
    bias = lambda key, o: 0.5 * jax.random.normal(key, (o,))
    p = dict(
        W1=weight(k[0], HID, OBS), b1=bias(k[1], HID),
        W2=weight(k[2], HID, HID), b2=bias(k[3], HID),
        W3=weight(k[4], OBS, HID), b3=bias(k[5], OBS),
        Wh=weight(k[6], HID, HID), bh=bias(k[7], HID)
    )
    buffer = jax.nn.one_hot(jax.random.randint(key_buffer, (N,), 0, OBS), OBS)
    obs = jax.jit(_batch)(buffer, jax.random.split(key_batch)[1])
    return float(jax.jit(_loss)(p, obs)), _reference(p, obs)


@functools.cache
def check_device(tolerance: float = 5e-3) -> None:
    """Raise if this device's linear layers disagree with numpy's (once per process; the CPU is not checked)."""
    if jax.default_backend() == "cpu":
        return
    got, want = device_error()
    if abs(got - want) > tolerance * abs(want):
        raise RuntimeError(
            f"This device computes linear layers wrongly with jax {jax.__version__}: a test loss is {got:.6f} on "
            f"{jax.devices()[0]} and {want:.6f} in numpy. Training would silently use a different model from the one "
            f"that acts. Upgrade jax, or set XLA_FLAGS={WORKAROUND} before starting Python."
        )
