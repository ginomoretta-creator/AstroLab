import jax
import jax.numpy as jnp
import numpy as np
import time
from generative import generate_thrust_schedules, create_time_series_model
from physics import batch_propagate

def random_sampler(key, num_steps, batch_size):
    """Generates random binary schedules."""
    return jax.random.bernoulli(key, p=0.5, shape=(batch_size, num_steps)).astype(jnp.float32)

def calculate_ising_energy(schedules, coupling_strength, external_field):
    """
    Calculates the Ising energy for a batch of schedules.
    E = -J * sum(s_i * s_{i+1}) - sum(h_i * s_i)
    Mapped from 0/1 to -1/1 spins.
    """
    # Map 0 -> -1, 1 -> 1
    spins = 2 * schedules - 1
    
    # Coupling term
    # sum(s_i * s_{i+1})
    interaction = jnp.sum(spins[:, :-1] * spins[:, 1:], axis=1)
    energy_coupling = -coupling_strength * interaction
    
    # Field term
    # sum(h_i * s_i)
    # external_field is (num_steps,)
    energy_field = -jnp.dot(spins, external_field)
    
    return energy_coupling + energy_field

def run_benchmark(num_steps=100, batch_size=50):
    print(f"Running Benchmark: {num_steps} steps, {batch_size} samples")
    
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    
    # Parameters
    J = 0.5
    # Create a synthetic field (e.g., eclipse in the middle)
    field = jnp.zeros((num_steps,))
    field = field.at[40:60].set(-5.0) # Eclipse
    
    # 1. THRML Generation
    start_time = time.time()
    thrml_schedules = generate_thrust_schedules(
        k1, num_steps, batch_size, coupling_strength=J, eclipse_indices=list(range(40, 60))
    )
    # Ensure binary
    thrml_schedules = jnp.round(thrml_schedules)
    thrml_time = time.time() - start_time
    
    # 2. Random Generation
    start_time = time.time()
    random_schedules = random_sampler(k2, num_steps, batch_size)
    random_time = time.time() - start_time
    
    # 3. Evaluate Energies
    thrml_energies = calculate_ising_energy(thrml_schedules, J, field)
    random_energies = calculate_ising_energy(random_schedules, J, field)
    
    # 4. Report
    print("\n--- Benchmark Results ---")
    print(f"THRML Time: {thrml_time:.4f}s")
    print(f"Random Time: {random_time:.4f}s")
    
    print("\nEnergy Statistics (Lower is Better):")
    print(f"THRML Mean Energy: {jnp.mean(thrml_energies):.4f}")
    print(f"Random Mean Energy: {jnp.mean(random_energies):.4f}")
    print(f"THRML Min Energy: {jnp.min(thrml_energies):.4f}")
    print(f"Random Min Energy: {jnp.min(random_energies):.4f}")
    
    # 5. Physics Check (Propagate one)
    # Just to ensure physics loop works with these schedules
    # Initial state: GEO-like orbit
    # x = 42164 km / 384400 km ~= 0.11
    initial_state = jnp.array([0.11, 0.0, 0.0, 3.07/1.024]) # Approx
    
    # Expand schedules to 2D thrust (tangential assumption for now: just x component)
    # In reality we need a guidance law. For benchmark, let's just apply thrust in X.
    thrml_thrust = jnp.stack([thrml_schedules, jnp.zeros_like(thrml_schedules)], axis=-1)
    
    # Propagate
    # batch_propagate(initial_state, thrust_schedule, dt, num_steps)
    # We need to broadcast initial_state
    # Actually my batch_propagate takes (None, 0, None, None) so it handles single initial state.
    
    trajectories = batch_propagate(initial_state, thrml_thrust, 0.01, num_steps)
    print(f"\nPhysics Propagation Check: {trajectories.shape}")
    print("Benchmark Complete.")

if __name__ == "__main__":
    run_benchmark()
