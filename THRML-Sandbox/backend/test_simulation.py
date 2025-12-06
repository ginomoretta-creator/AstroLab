import jax
import jax.numpy as jnp
import numpy as np
from physics import batch_propagate
from generative import generate_thrust_schedules

def test_simulation():
    print("Testing Simulation Pipeline...")
    
    # 1. Test Generative Model
    print("1. Testing Generative Model (THRML)...")
    try:
        key = jax.random.PRNGKey(0)
        schedules = generate_thrust_schedules(key, num_steps=10, batch_size=5)
        print(f"   Schedules generated. Shape: {schedules.shape}")
        assert schedules.shape == (5, 10)
        assert not jnp.any(jnp.isnan(schedules))
        print("   Generative Model OK.")
    except Exception as e:
        print(f"   Generative Model FAILED: {e}")
        return

    # 2. Test Physics Engine
    print("2. Testing Physics Engine (JAX)...")
    try:
        initial_state = jnp.array([0.11, 0.0, 0.0, 2.9])
        thrust_vectors = jnp.stack([schedules, jnp.zeros_like(schedules)], axis=-1)
        trajectories = batch_propagate(initial_state, thrust_vectors, 0.01, 10)
        print(f"   Trajectories propagated. Shape: {trajectories.shape}")
        assert trajectories.shape == (5, 11, 4)
        assert not jnp.any(jnp.isnan(trajectories))
        print("   Physics Engine OK.")
    except Exception as e:
        print(f"   Physics Engine FAILED: {e}")
        return

    print("All Tests Passed!")

if __name__ == "__main__":
    test_simulation()
