import time

import jax
import jax.numpy as jnp
from ..actor import ActorModel


# Config
batch_size = 4
belief_size = 3
state_size = 2
hidden_size = 16
action_size = 3  # action_size > 1


def test_actor_model_batch():
    key = jax.random.PRNGKey(0)

    # Create dummy batch input
    input_tensor = jax.random.normal(key, (batch_size, belief_size + state_size))
    print(f"Input tensor shape {input_tensor.shape}")

    for head_type in ["Tanh Normal", "Beta"]:
        print(f"\nTesting head type: {head_type}")

        # Initialize actor
        key, subkey = jax.random.split(key)
        actor = ActorModel(
            belief_size=belief_size,
            state_size=state_size,
            hidden_size=hidden_size,
            action_size=action_size,
            head_type=head_type,
            key=subkey
        )

        # Forward pass to get distribution
        vmap_actor = jax.vmap(actor)
        # vmap_actor = eqx.filter_vmap(actor)
        # dist = jax.vmap(actor)(input_tensor)
        dist = vmap_actor(input_tensor)
        print("Distribution type:", type(dist))

        # Sample actions
        key, subkey = jax.random.split(key)
        actions = eqx.filter_vmap(
            lambda x: actor.get_action(x, det=False, key=subkey),
            in_axes=0
        )(input_tensor)
        print("Sampled actions shape:", actions.shape)
        assert actions.shape == (batch_size, action_size)

        # Deterministic actions (mode)
        key, subkey = jax.random.split(key)
        det_actions = eqx.filter_vmap(
            lambda x: actor.get_action(x, det=True, key=subkey),
            in_axes=0
        )(input_tensor)
        print("Deterministic actions shape:", det_actions.shape)
        assert det_actions.shape == (batch_size, action_size)

        # Test batch shape
        print(f"batch shape {dist.distribution.distribution.batch_shape}")

        # Test log_prob
        log_probs = dist.log_prob(actions)
        print("Log probs shape:", log_probs.shape)
        assert log_probs.shape == (batch_size,)

        # Test entropy via sampling
        sample_dist = SampleDist(dist, num_samples=50)
        key, subkey = jax.random.split(key)
        entropy = sample_dist.entropy(seed=subkey)
        print("Entropy estimate shape:", entropy.shape)
        assert entropy.shape == (batch_size,)

        # Test mean via sampling
        key, subkey = jax.random.split(key)
        mean_estimate = sample_dist.mean(seed=subkey)
        print("Mean estimate shape:", mean_estimate.shape)
        assert mean_estimate.shape == (batch_size, action_size)

        # Test mode via sampling
        key, subkey = jax.random.split(key)
        mode_estimate = sample_dist.mode(seed=subkey)
        print("Mode estimate shape:", mode_estimate.shape)
        assert mode_estimate.shape == (batch_size, action_size)


def test_actor_model_jit():
    key = jax.random.PRNGKey(0)

    # Create dummy batch input
    input_tensor = jax.random.normal(key, (batch_size, belief_size + state_size))
    print(f"Input tensor shape {input_tensor.shape}")

    iters = 100

    for head_type in ["Tanh Normal", "Beta"]:
        print(f"\nTesting head type: {head_type}")

        # Initialize actor
        key, subkey = jax.random.split(key)
        model = ActorModel(
            belief_size=belief_size,
            state_size=state_size,
            hidden_size=hidden_size,
            action_size=action_size,
            head_type=head_type,
            key=subkey
        )

        get_action_non_jit = eqx.filter_vmap(
            lambda x, k: model.get_action(x, det=False, key=k),
            in_axes=(0, None),
        )

        get_action = eqx.filter_jit(get_action_non_jit)

        # Prepare inputs
        obs = jnp.array(input_tensor)

        # --- 1. Measure Un-JITed (Eager) Performance ---
        # Note: We call the function directly without the @eqx.filter_jit wrapper
        start = time.time()
        for _ in range(iters):
            # Using the model's __call__ or raw logic
            # _ = model.get_action(obs, key=key)
            _ = get_action_non_jit(obs, key)
        eager_time = (time.time() - start) / iters
        print(f"Eager (Un-JITed) average time: {eager_time:.6f}s")

        # --- 2. Measure First JIT Run (Compilation Overhead) ---
        start = time.time()
        # This triggers the JAX tracer and compiler
        _ = get_action(obs, key).block_until_ready()
        compilation_time = time.time() - start
        print(f"First JIT run (Compilation + Exec): {compilation_time:.6f}s")

        # --- 3. Measure Subsequent JIT Runs (The Real Gains) ---
        start = time.time()
        for _ in range(iters):
            _ = get_action(obs, key).block_until_ready()
        jit_time = (time.time() - start) / iters
        print(f"JIT (Post-compilation) average time: {jit_time:.6f}s")

        # --- Summary ---
        speedup = eager_time / jit_time
        print(f"\nSpeedup: {speedup:.2f}x")


if __name__ == "__main__":
    test_actor_model_batch()
    test_actor_model_jit()
