import jax
from ..sensor import Encoder, Decoder

import time

"""Encoder Test."""


shapes = {
    'S': (3, 64, 64),
    'M': (3, 84, 84),
    'L': (3, 96, 96),
}

belief_size = 200
state_size = 30

batch_size = 100
iters = 1000


def test_encoder_shape():
    key = jax.random.PRNGKey(42)

    # Vary depth and not provide embedding_size
    print(f"cnn depth {32}")
    for k, v in shapes.items():
        key, subkey = jax.random.split(key)
        cnn = Encoder(shape=v, key=subkey)
        print(f"{k} embed size {cnn.embedding_size}")


    print(f"\ncnn depth {48}")
    for k, v in shapes.items():
        key, subkey = jax.random.split(key)
        cnn = Encoder(depth=48, shape=v, key=subkey)
        print(f"{k} embed size {cnn.embedding_size}")


    # Given embedding_size
    embedding_size = 1024
    print(f"\ncnn depth {32} with embedding_size {embedding_size}")
    for k, v in shapes.items():
        key, subkey = jax.random.split(key)
        cnn = Encoder(shape=v, embedding_size=embedding_size, key=subkey)
        print(f"{k} embed size {cnn.embedding_size}")


    print(f"\ncnn depth {48} with embedding_size {embedding_size}")
    for k, v in shapes.items():
        key, subkey = jax.random.split(key)
        cnn = Encoder(depth=48, shape=v, embedding_size=embedding_size, key=subkey)
        print(f"{k} embed size {cnn.embedding_size}")


"""Decoder Test."""


def test_decoder_shape():

    key = jax.random.PRNGKey(42)
    test_input_tensor = jnp.ones((belief_size + state_size, ))

    print(f"\ncnn depth {32}")
    for k, v in shapes.items():
        key, subkey = jax.random.split(key)
        dec = Decoder(belief_size, state_size, shape=v, key=subkey)
        dist = dec(test_input_tensor)
        assert isinstance(dist, distrax.Distribution)
        key, subkey = jax.random.split(key)
        sample = dist.sample(seed=subkey)
        print(f"{k} output size {sample.shape}")

    print(f"\ncnn depth {48}")
    for k, v in shapes.items():
        key, subkey = jax.random.split(key)
        dec = Decoder(belief_size, state_size, shape=v, key=subkey)
        dist = dec(test_input_tensor)
        assert isinstance(dist, distrax.Distribution)
        key, subkey = jax.random.split(key)
        sample = dist.sample(seed=subkey)
        print(f"{k} output size {sample.shape}")


def test_jit_encoder():
    key = jax.random.PRNGKey(42)

    key, subkey = jax.random.split(key)
    enc = Encoder(shape=shapes['S'], key=subkey)

    test_input_tensor = jnp.ones((100, *shapes['S']))
    print(f"\ntest input tensor shape {test_input_tensor.shape}")

    print("\nEncoder testing vmap ... ")
    vmap_enc = eqx.filter_vmap(enc)
    out = vmap_enc(test_input_tensor)
    print(f"vmap out size {out.shape}")

    print("\nEncoder testing jit ... ")
    jit_enc = eqx.filter_jit(vmap_enc)

    # warm-up
    start = time.time()
    _ = jit_enc(test_input_tensor).block_until_ready()
    compilation_time = time.time() - start
    print(f"First JIT run (Compilation + Exec): {compilation_time:.6f}s")

    # post-compilation
    start = time.time()
    for _ in range(iters):
        _ = jit_enc(test_input_tensor).block_until_ready()
    jit_time = (time.time() - start) / iters
    print(f"JIT (Post-compilation) average time: {jit_time:.6f}s")


def test_jit_decoder():
    key = jax.random.PRNGKey(42)

    key, subkey = jax.random.split(key)
    dec = Decoder(belief_size, state_size, shape=shapes['S'], key=subkey)

    test_input_tensor = jnp.ones((batch_size, belief_size + state_size))
    test_image_tensor = jnp.ones((batch_size, *shapes['S']))
    print(f"\ntest input tensor shape {test_input_tensor.shape}")

    print("\nDecoder testing vmap ... ")
    # vmap_dec = eqx.filter_vmap(dec)
    vmap_dec = jax.vmap(dec)
    out = vmap_dec(test_input_tensor)
    # key, subkey = jax.random.split(key)
    # log_prob = out.log_prob(test_image_tensor)
    # out_sample = out.sample(seed=subkey)
    # print(f"vmap out sample sizei  {out_sample.shape}")
    # print(f"vmap out log prob size {log_prob.shape}")

    print("\nDecoder testing jit ... ")
    # jit_dec = eqx.filter_jit(vmap_dec)
    jit_dec = jax.jit(vmap_dec)
    # jit_dec = get_dist

    # warm-up
    start = time.time()
    # _ = jit_dec(dec, test_input_tensor).block_until_ready()
    _ = jit_dec(test_input_tensor).mean().block_until_ready()
    # _ = jit_dec(test_input_tensor).block_until_ready()
    compilation_time = time.time() - start
    print(f"First JIT run (Compilation + Exec): {compilation_time:.6f}s")

    # post-compilation
    start = time.time()
    for _ in range(iters):
        # _ = jit_dec(dec, test_input_tensor).block_until_ready()
        _ = jit_dec(test_input_tensor).mean().block_until_ready()
        # _ = jit_dec(test_input_tensor).block_until_ready()
    jit_time = (time.time() - start) / iters
    print(f"JIT (Post-compilation) average time: {jit_time:.6f}s")


if __name__ == "__main__":
    test_encoder_shape()
    test_decoder_shape()
    test_jit_encoder()
    test_jit_decoder()
