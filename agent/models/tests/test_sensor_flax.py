import time
from typing import Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import lecun_normal

# Activation function helper
def get_activation_fn(name: str):
    if name.lower() == "elu":
        return nn.elu
    elif name.lower() == "relu":
        return nn.relu
    else:
        raise ValueError(f"Unsupported activation: {name}")

# Flax Decoder
class Decoder(nn.Module):
    belief_size: int
    state_size: int
    shape: Tuple[int, int, int]
    kernel_size: int = 4
    depth: int = 48
    stride: int = 2
    activation_function: str = "elu"
    embedding_size: int = 1024

    @nn.compact
    def __call__(self, x):
        activation = get_activation_fn(self.activation_function)

        # Linear embedding
        x = nn.Dense(self.embedding_size)(x)
        x = activation(x)

        # Add dummy spatial dimensions
        x = x[..., None, None, :]

        # ConvTranspose stack
        x = nn.ConvTranspose(features=4*self.depth, kernel_size=(5,5), strides=(self.stride,self.stride))(x)
        x = activation(x)
        x = nn.ConvTranspose(features=2*self.depth, kernel_size=(5,5), strides=(self.stride,self.stride))(x)
        x = activation(x)
        x = nn.ConvTranspose(features=1*self.depth, kernel_size=(6,6), strides=(self.stride,self.stride))(x)
        x = activation(x)
        x = nn.ConvTranspose(features=self.shape[0], kernel_size=(6,6), strides=(self.stride,self.stride))(x)

        dist = distrax.Normal(x, jnp.ones_like(x))
        return distrax.Independent(dist, reinterpreted_batch_ndims=len(self.shape))


belief_size = 200
state_size = 30
shapes = {'S': (3, 64, 64), 'M': (3, 84, 84), 'L': (3, 96, 96)}

batch_size = 100
test_input_tensor = jnp.ones((batch_size, belief_size + state_size))

# Initialize model
key = jax.random.PRNGKey(0)
decoder = Decoder(belief_size, state_size, shape=shapes['S'])
params = decoder.init(key, test_input_tensor)

# JIT compile
dec_jit = jax.jit(lambda x: decoder.apply(params, x))

# Warm-up (compilation + first run)
start = time.time()
# _ = dec_jit(test_input_tensor).block_until_ready()
_ = dec_jit(test_input_tensor).mean().block_until_ready()
compilation_time = time.time() - start
print(f"First JIT run (Compilation + Exec): {compilation_time:.6f}s")

# Post-compilation benchmark
iters = 100
start = time.time()
for _ in range(iters):
    # _ = dec_jit(test_input_tensor).block_until_ready()
    _ = dec_jit(test_input_tensor).mean().block_until_ready()
jit_time = (time.time() - start) / iters
print(f"JIT (Post-compilation) average time: {jit_time:.6f}s")
