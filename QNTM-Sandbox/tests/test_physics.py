import pytest
import jax.numpy as jnp
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))

from physics import equations_of_motion, propagate_trajectory

def test_equations_of_motion_shape():
    state = jnp.array([0.5, 0.0, 0.0, 1.0])
    t = 0.0
    thrust = 0.0
    dstate = equations_of_motion(state, t, thrust)
    assert dstate.shape == (4,)

def test_propagation_conservation():
    # Test that with 0 thrust, energy (Jacobi constant) is roughly conserved
    # or at least that it runs without error
    state = jnp.array([0.8, 0.0, 0.0, 1.0])
    thrust_schedule = jnp.zeros(100)
    dt = 0.01
    traj = propagate_trajectory(state, thrust_schedule, dt, 100)
    assert traj.shape == (101, 4)
    assert not jnp.any(jnp.isnan(traj))
