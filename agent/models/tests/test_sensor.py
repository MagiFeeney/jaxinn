import jax
from ..sensor import Encoder, Decoder


"""Test Encoder."""


shapes = {
    'S': (3, 64, 64),
    'M': (3, 84, 84),
    'L': (3, 96, 96),
}


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
