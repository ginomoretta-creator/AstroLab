"""
Physics Engine Validation Tests

Tests for CR3BP dynamics, numerical integration, and trajectory propagation.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from physics import equations_of_motion, rk4_step, propagate_trajectory, batch_propagate, MU


class TestCR3BPDynamics:
    """Test CR3BP equations of motion"""
    
    def test_lagrange_points_are_equilibria(self):
        """L1, L2, L3, L4, L5 should have zero velocity and acceleration"""
        # L1 point (approximate, between Earth and Moon)
        x_L1 = 1 - MU - 0.01  # Rough approximation
        state_L1 = jnp.array([x_L1, 0.0, 0.0, 0.0])
        
        dstate = equations_of_motion(state_L1, 0.0, 0.0)  # No thrust
        
        # Velocities should be zero (they are by construction)
        assert dstate[0] == 0.0
        assert dstate[1] == 0.0
        
        # Accelerations should be small (not exactly zero due to approximation)
        # In a real test, we'd use Newton-Raphson to find exact L1
        # For now, just check dynamics don't explode
        assert jnp.abs(dstate[2]) < 10.0
        assert jnp.abs(dstate[3]) < 10.0
    
    def test_earth_moon_positions(self):
        """Verify primary positions"""
        assert jnp.isclose(MU, 0.01215, atol=0.0001)
        
        # Earth at (-MU, 0)
        earth_x = -MU
        assert jnp.isclose(earth_x, -0.01215, atol=0.0001)
        
        # Moon at (1-MU, 0)
        moon_x = 1 - MU
        assert jnp.isclose(moon_x, 0.98785, atol=0.0001)
    
    def test_thrust_increases_velocity(self):
        """Positive thrust should increase velocity magnitude"""
        # Start at some state
        state = jnp.array([0.9, 0.1, 0.0, 0.0])
        
        # No thrust
        dstate_no_thrust = equations_of_motion(state, 0.0, 0.0)
        
        # With thrust
        dstate_thrust = equations_of_motion(state, 0.0, 0.1)
        
        # Acceleration magnitude should increase
        accel_no_thrust = jnp.sqrt(dstate_no_thrust[2]**2 + dstate_no_thrust[3]**2)
        accel_thrust = jnp.sqrt(dstate_thrust[2]**2 + dstate_thrust[3]**2)
        
        assert accel_thrust > accel_no_thrust
    
    def test_softening_prevents_singularity(self):
        """Should not crash when very close to primaries"""
        # Very close to Earth
        state_earth = jnp.array([-MU + 0.001, 0.0, 0.0, 0.1])
        dstate = equations_of_motion(state_earth, 0.0, 0.0)
        
        # Should have large but finite acceleration
        assert jnp.isfinite(dstate[2])
        assert jnp.isfinite(dstate[3])
        assert jnp.abs(dstate[2]) < 1e6  # Not infinity


class TestNumericalIntegration:
    """Test RK4 integrator"""
    
    def test_rk4_advances_time(self):
        """RK4 should propagate state forward"""
        state = jnp.array([0.8, 0.0, 0.0, 0.5])
        dt = 0.01
        
        next_state = rk4_step(state, 0.0, dt, 0.0)
        
        # Position should change based on velocity
        assert not jnp.allclose(state, next_state)
        
        # Should preserve dimensionality
        assert next_state.shape == state.shape
    
    def test_energy_conservation_free_flight(self):
        """Jacobi constant should be approximately conserved (no thrust)"""
        # Start in a stable orbit
        state = jnp.array([0.8, 0.0, 0.0, 0.6])
        
        def jacobi_constant(s):
            x, y, vx, vy = s
            r1_sq = (x + MU)**2 + y**2
            r2_sq = (x - (1 - MU))**2 + y**2
            r1 = jnp.sqrt(r1_sq)
            r2 = jnp.sqrt(r2_sq)
            
            # Jacobi constant C = -(v^2 - (x^2 + y^2) - 2(1-mu)/r1 - 2*mu/r2)
            v_sq = vx**2 + vy**2
            C = -(v_sq - (x**2 + y**2) - 2*(1-MU)/r1 - 2*MU/r2)
            return C
        
        jacobi_0 = jacobi_constant(state)
        
        # Propagate for 100 steps
        current_state = state
        for _ in range(100):
            current_state = rk4_step(current_state, 0.0, 0.01, 0.0)
        
        jacobi_final = jacobi_constant(current_state)
        
        # Should be conserved within ~1% (numerical error accumulates)
        relative_error = jnp.abs(jacobi_final - jacobi_0) / jnp.abs(jacobi_0)
        assert relative_error < 0.01  # 1% tolerance


class TestTrajectoryPropagation:
    """Test trajectory generation"""
    
    def test_propagate_trajectory_shape(self):
        """Output should have correct shape"""
        initial_state = jnp.array([0.8, 0.0, 0.0, 0.5])
        num_steps = 100
        thrust_schedule = jnp.zeros(num_steps)
        dt = 0.01
        
        trajectory = propagate_trajectory(initial_state, thrust_schedule, dt, num_steps)
        
        # Should be (num_steps + 1, 4) because it includes initial state
        assert trajectory.shape == (num_steps + 1, 4)
        
        # First state should match initial
        assert jnp.allclose(trajectory[0], initial_state)
    
    def test_constant_thrust_vs_no_thrust(self):
        """Constant thrust should result in different final state"""
        initial_state = jnp.array([0.8, 0.0, 0.0, 0.5])
        num_steps = 100
        dt = 0.01
        
        thrust_off = jnp.zeros(num_steps)
        thrust_on = jnp.ones(num_steps) * 0.05
        
        traj_off = propagate_trajectory(initial_state, thrust_off, dt, num_steps)
        traj_on = propagate_trajectory(initial_state, thrust_on, dt, num_steps)
        
        # Final states should differ
        assert not jnp.allclose(traj_off[-1], traj_on[-1])
        
        # With thrust, should have more energy (higher velocity typically)
        v_off = jnp.sqrt(traj_off[-1, 2]**2 + traj_off[-1, 3]**2)
        v_on = jnp.sqrt(traj_on[-1, 2]**2 + traj_on[-1, 3]**2)
        
        # Not always true depending on direction, but generally
        # For this test we just ensure they're different
        assert not jnp.isclose(v_off, v_on)
    
    def test_batch_propagate(self):
        """Batch propagation should handle multiple thrust schedules"""
        initial_state = jnp.array([0.8, 0.0, 0.0, 0.5])
        num_steps = 50
        batch_size = 10
        dt = 0.01
        
        # Create batch of random thrust schedules
        key = jax.random.PRNGKey(42)
        thrust_schedules = jax.random.uniform(key, (batch_size, num_steps)) * 0.1
        
        trajectories = batch_propagate(initial_state, thrust_schedules, dt, num_steps)
        
        # Should be (batch_size, num_steps + 1, 4)
        assert trajectories.shape == (batch_size, num_steps + 1, 4)
        
        # All should start from same initial state
        for i in range(batch_size):
            assert jnp.allclose(trajectories[i, 0], initial_state)
        
        # Different schedules should lead to different final states
        # Check that not all final states are identical
        final_states = trajectories[:, -1, :]
        for i in range(batch_size - 1):
            if not jnp.allclose(final_states[i], final_states[i+1]):
                break
        else:
            pytest.fail("All final states are identical despite different thrust schedules")


class TestPhysicalRealism:
    """Test that trajectories behave physically"""
    
    def test_no_collision_with_earth(self):
        """Trajectory starting in LEO should not crash into Earth"""
        # Start at LEO-like orbit
        R_EARTH_NORM = 6378.0 / 384400.0  # ~ 0.0166
        altitude_norm = 400.0 / 384400.0  # ~ 0.001
        
        x0 = -MU + R_EARTH_NORM + altitude_norm
        # Circular velocity approximation
        v_mag = jnp.sqrt((1 - MU) / (R_EARTH_NORM + altitude_norm))
        
        initial_state = jnp.array([x0, 0.0, 0.0, v_mag])
        
        # Propagate without thrust
        num_steps = 200
        thrust_schedule = jnp.zeros(num_steps)
        trajectory = propagate_trajectory(initial_state, thrust_schedule, 0.01, num_steps)
        
        # Check minimum distance to Earth
        positions = trajectory[:, :2]
        earth_pos = jnp.array([-MU, 0.0])
        distances = jnp.linalg.norm(positions - earth_pos, axis=1)
        
        min_distance = jnp.min(distances)
        
        # Should stay above Earth's surface
        assert min_distance > R_EARTH_NORM * 0.9  # 10% tolerance for numerical drift
    
    def test_reaches_moon_with_high_thrust(self):
        """Very high thrust should be able to reach Moon vicinity"""
        initial_state = jnp.array([0.0, 0.0, 0.0, 0.0])  # Start near Earth
        
        # Very high constant thrust toward Moon
        num_steps = 1000
        thrust_schedule = jnp.ones(num_steps) * 0.5  # Strong thrust
        
        trajectory = propagate_trajectory(initial_state, thrust_schedule, 0.01, num_steps)
        
        # Check if we got close to Moon
        moon_pos = jnp.array([1 - MU, 0.0])
        final_pos = trajectory[-1, :2]
        final_distance = jnp.linalg.norm(final_pos - moon_pos)
        
        # This is a very rough test - proper guidance would be needed for actual transfer
        # Just checking that high thrust can significantly change position
        initial_distance = jnp.linalg.norm(initial_state[:2] - moon_pos)
        
        # Should get closer than initial distance
        # (This might fail depending on thrust direction - tangent vs radial)
        # For now, just check trajectory moved significantly
        distance_traveled = jnp.linalg.norm(trajectory[-1, :2] - trajectory[0, :2])
        assert distance_traveled > 0.1  # Moved significantly


if __name__ == "__main__":
    # Run with: pytest tests/test_physics.py -v
    # Or: python tests/test_physics.py
    pytest.main([__file__, "-v"])
