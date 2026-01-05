import jax
import jax.numpy as jnp
import chex

def test_uae_reduction():
    # 1. Setup Hyperparameters
    gamma = 0.99
    uae_lambda = 0.95
    steps = 5

    # 2. Mock Data
    rewards = jnp.array([1.0, 0.5, 2.0, 0.0, 1.0])
    values = jnp.array([0.5, 1.2, 1.0, 2.0, 0.8])
    last_value = jnp.array(0.9)
    baselines = jnp.arange(len(rewards))  # Baseline = 0
    dones = jnp.zeros_like(rewards)

    # 3. Your UAE Implementation (Refactored for the test)
    def uae_step_fn(carry, inputs):
        uae, next_value = carry
        reward, value, baseline, done = inputs

        delta = reward + gamma * next_value * (1 - done) - baseline
        z = value - baseline
        discounted_uae = gamma * uae_lambda * (1 - done) * uae

        return_prediction = delta + discounted_uae + baseline
        uae = (delta - z) + discounted_uae
        return (uae, value), return_prediction

    _, uae_returns = jax.lax.scan(
        uae_step_fn,
        (jnp.zeros_like(last_value), last_value),
        (rewards, values, baselines, dones),
        reverse=True,
    )

    # 4. Standard Lambda-Return Implementation for comparison
    # G(t) = r(t) + gamma * ((1 - lambda) * V(t+1) + lambda * G(t+1))
    def lambda_return_ref(rewards, values, last_value, gamma, lmbda):
        returns = jnp.zeros_like(rewards)
        next_ret = last_value
        next_val = last_value
        for i in reversed(range(len(rewards))):
            # The standard TD(lambda) target
            target = rewards[i] + gamma * ((1 - lmbda) * next_val + lmbda * next_ret)
            returns = returns.at[i].set(target)
            next_ret = target
            next_val = values[i]
        return returns

    ref_returns = lambda_return_ref(rewards, values, last_value, gamma, uae_lambda)

    # 5. Assertion
    try:
        chex.assert_trees_all_close(uae_returns, ref_returns, rtol=1e-5)
        print("✅ Test Passed: UAE reduces to Lambda-return when baseline=0")
    except AssertionError as e:
        print(f"❌ Test Failed: \nUAE: {uae_returns}\nRef: {ref_returns}")

if __name__ == "__main__":
    test_uae_reduction()
