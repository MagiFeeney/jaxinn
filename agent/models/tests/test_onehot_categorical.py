import jax
import jax.numpy as jnp
import distrax

N, K = 3, 10
B = 20

out = jnp.ones((B, N * K))

out = out.reshape(*out.shape[:-1], N, K)
print(f"logit shape {out.shape}")

dist = distrax.OneHotCategorical(logits=out)
s = dist.sample(seed=0)
print(f"sample {s.shape}")
logp = dist.log_prob(s)
print(f"logp {logp.shape}")
ind_dist = distrax.Independent(dist, reinterpreted_batch_ndims=1)
s = ind_dist.sample(seed=0)
print(f"ind sample {s.shape}")

dx_dist = dx.OneHotCategorical(logits=out)
dx_s = dx_dist.sample(seed=0)
print(f"dx sample {dx_s.shape}")
dx_logp = dx_dist.log_prob(dx_s)
print(f"dx_logp {dx_logp.shape}")
ind_dx_dist = dx.Independent(dx_dist, reinterpreted_batch_ndims=1)
dx_s = ind_dx_dist.sample(seed=0)
print(f"ind dx sample {dx_s.shape}")
