import pytest
import sys
import os
import jax.numpy as jnp

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))

from quantum_solver import SimulatedQuantumAnnealer

def test_annealer_initialization():
    annealer = SimulatedQuantumAnnealer()
    assert annealer is not None

def test_generate_schedules_shape():
    annealer = SimulatedQuantumAnnealer()
    num_steps = 50
    batch_size = 10
    schedules = annealer.generate_thrust_schedules(num_steps, batch_size)
    
    assert schedules.shape == (batch_size, num_steps)
    # Check values are binary (0 or 1)
    assert jnp.all((schedules == 0.0) | (schedules == 1.0))

def test_coupling_effect():
    # With high coupling, neighbors should be same
    annealer = SimulatedQuantumAnnealer()
    schedules = annealer.generate_thrust_schedules(num_steps=20, batch_size=10, coupling_strength=10.0)
    
    # Calculate number of switches
    switches = jnp.abs(schedules[:, 1:] - schedules[:, :-1])
    avg_switches = jnp.mean(jnp.sum(switches, axis=1))
    
    # With high coupling, switches should be low
    assert avg_switches < 5.0 # Expect very few switches
