from ..memory import Uniform, Prioritized


CAPACITY = 20
OBS_SHAPE = (4,)
ACT_DIM = 2
BATCH_ADD_SIZE = 5
BATCH_SAMPLE_SIZE = 20
CHUNK_SIZE = 4


def test_uniform():
    print(" --- Testing Uniform Memory --- ")
    key = jax.random.PRNGKey(0)

    memory = Uniform(CAPACITY, OBS_SHAPE, ACT_DIM)
    print(f"1. Init: Size={memory.size}, Ptr={memory.ptr}, Capacity={memory.capacity}")

    # Generate dummy batch data
    def make_dummy_batch(seed, size):
        k = jax.random.PRNGKey(seed)
        return Transition(
            action=jax.random.uniform(k, (size, ACT_DIM)),
            next_obs=jax.random.randint(k, (size, *OBS_SHAPE), 0, 255).astype(jnp.uint8),
            reward=jnp.ones((size,)),
            done=jnp.zeros((size,))
        )

    for i in range(3):
        batch = make_dummy_batch(i, BATCH_ADD_SIZE)
        memory = memory.add(batch) # MUST reassign memory
        print(f"2. Added batch {i+1}: Size={memory.size}, Ptr={memory.ptr}")

    assert memory.size == 15
    assert memory.ptr == 15

    # Test Sampling
    sample_key, key = jax.random.split(key)
    sample_shape = (BATCH_SAMPLE_SIZE, CHUNK_SIZE)

    # We JIT the sample function to ensure XLA compatibility
    @jax.jit
    def jit_sample(mem, k):
        return mem.sample(sample_shape, k)

    traj = jit_sample(memory, sample_key)

    # T x B (Chunk x Batch)
    T, B = sample_shape[1], sample_shape[0]

    print("\n3. Sampling Checks:")
    print(f"   Expected shape (T, B, ...): ({T}, {B}, ...)")
    print(f"   Actual Action shape: {traj.action.shape}")
    print(f"   Actual Reward shape: {traj.reward.shape}")

    assert traj.action.shape == (T, B, ACT_DIM)
    assert traj.reward.shape == (T, B)

    # Test Wrapping
    # Add 2 more batches.
    print("\n4. Testing Wrap-around:")
    for i in range(2):
        batch = make_dummy_batch(i+10, BATCH_ADD_SIZE)
        memory = memory.add(batch)
        print(f"4. Added batch {i+1}: Size={memory.size}, Ptr={memory.ptr}")

    print("\n5. Testing sampling after wrap-around:")
    traj = jit_sample(memory, sample_key)

    print("\n5. Sampling Checks:")
    print(f"   Expected shape (T, B, ...): ({T}, {B}, ...)")
    print(f"   Actual Action shape: {traj.action.shape}")
    print(f"   Actual Reward shape: {traj.reward.shape}")

    print(f"\n   Final State: Size={memory.size} (Expect 20), Ptr={memory.ptr} (Expect 5)")
    print(f"   Is Full? {memory.full}")

    assert memory.size == CAPACITY
    assert memory.ptr == 5
    assert memory.full

    print("\nSUCCESS: Memory passed basic functional tests.")


def test_prioritized():
    print("\n\n --- Testing Prioritized Memory --- ")
    key = jax.random.PRNGKey(0)

    memory = Prioritized(CAPACITY, OBS_SHAPE, ACT_DIM, chunk_size=CHUNK_SIZE)
    print(f"1. Init: Size={memory.size}, Ptr={memory.ptr}, Capacity={memory.capacity}\n   chunk size={memory.sumtree.chunk_size}, alpha={memory.alpha}, beta={memory.beta}")

    # Generate dummy batch data
    def make_dummy_batch(seed, size):
        k = jax.random.PRNGKey(seed)
        return Transition(
            action=jax.random.uniform(k, (size, ACT_DIM)),
            next_obs=jax.random.randint(k, (size, *OBS_SHAPE), 0, 255).astype(jnp.uint8),
            reward=jnp.ones((size,)),
            done=jnp.zeros((size,))
        )

    for i in range(3):
        batch = make_dummy_batch(i, BATCH_ADD_SIZE)
        priorities = jax.random.uniform(jax.random.PRNGKey(i), (BATCH_ADD_SIZE,))
        memory = memory.add(batch, priorities)
        print(f"2. Added batch {i+1}: Size={memory.size}, Ptr={memory.ptr}")

        last_idx = (memory.ptr - 1) % CAPACITY
        tree_stored_val = memory.sumtree.tree[last_idx]
        masked_stored_val = memory.sumtree.masked_priorities[-1]
        input_val = priorities[-1] ** memory.alpha

        # Simple float check
        assert jnp.allclose(masked_stored_val, input_val), "SumTree priority mismatch!"
        assert tree_stored_val == 0, "tree last not zero!"


    assert memory.size == 15
    assert memory.ptr == 15

    # Test Sampling
    sample_key, key = jax.random.split(key)
    sample_shape = (BATCH_SAMPLE_SIZE, CHUNK_SIZE)

    # We JIT the sample function to ensure XLA compatibility
    @jax.jit
    def jit_sample(mem, k):
        return mem.sample(sample_shape, k)

    traj = jit_sample(memory, sample_key)

    # T x B (Chunk x Batch)
    T, B = sample_shape[1], sample_shape[0]

    print("\n3. Sampling Checks:")
    print(f"   Expected shape (T, B, ...): ({T}, {B}, ...)")
    print(f"   Actual Action shape: {traj.action.shape}")
    print(f"   Actual Reward shape: {traj.reward.shape}")

    assert traj.action.shape == (T, B, ACT_DIM)
    assert traj.reward.shape == (T, B)

    # Test Wrapping
    # Add 2 more batches.
    print("\n4. Testing Wrap-around:")
    for i in range(2):
        batch = make_dummy_batch(i+10, BATCH_ADD_SIZE)
        priorities = jax.random.uniform(jax.random.PRNGKey(i), (BATCH_ADD_SIZE,))
        memory = memory.add(batch, priorities)
        print(f"4. Added batch {i+1}: Size={memory.size}, Ptr={memory.ptr}")

    print("\n5. Testing sampling after wrap-around:")
    traj = jit_sample(memory, sample_key)

    print("\n5. Sampling Checks:")
    print(f"   Expected shape (T, B, ...): ({T}, {B}, ...)")
    print(f"   Actual Action shape: {traj.action.shape}")
    print(f"   Actual Reward shape: {traj.reward.shape}")

    print(f"\n   Final State: Size={memory.size} (Expect 20), Ptr={memory.ptr} (Expect 5)")
    print(f"   Is Full? {memory.full}")

    assert memory.size == CAPACITY
    assert memory.ptr == 5
    assert memory.full

    print("\nSUCCESS: Memory passed basic functional tests.")


if __name__ == "__main__":
    test_uniform()
    test_prioritized()
