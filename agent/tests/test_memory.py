from ..memory import Uniform, Prioritized


def test_memory():
    CAPACITY = 20
    OBS_SHAPE = (4,)
    ACT_DIM = 2
    BATCH_ADD_SIZE = 5
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
    sample_shape = (20, 4)

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

if __name__ == "__main__":
    test_memory()
