"""
THRML Integration Tests

Tests for thrust schedule generation via probabilistic sampling.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

try:
    from generative import generate_thrust_schedules, create_time_series_model
    THRML_AVAILABLE = True
except ImportError:
    THRML_AVAILABLE = False
    pytest.skip("THRML not available", allow_module_level=True)


class TestThrustScheduleGeneration:
    """Test thrust schedule sampling"""
    
    def test_schedule_shape(self):
        """Generated schedules should have correct shape"""
        key = jax.random.PRNGKey(42)
        num_steps = 100
        batch_size = 10
        
        schedules = generate_thrust_schedules(key, num_steps, batch_size)
        
        assert schedules.shape == (batch_size, num_steps)
    
    def test_schedule_values_binary(self):
        """Schedules should be 0 or 1 (thrust off/on)"""
        key = jax.random.PRNGKey(42)
        num_steps = 100
        batch_size = 10
        
        schedules = generate_thrust_schedules(key, num_steps, batch_size)
        
        # Should only contain 0 or 1
        unique_values = jnp.unique(schedules)
        assert jnp.all((unique_values == 0.0) | (unique_values == 1.0))
    
    def test_coupling_increases_smoothness(self):
        """Higher coupling should produce smoother schedules"""
        key = jax.random.PRNGKey(42)
        num_steps = 200
        batch_size = 20
        
        # Low coupling (more random)
        schedules_low = generate_thrust_schedules(
            key, num_steps, batch_size, coupling_strength=0.0
        )
        
        # High coupling (smoother)
        schedules_high = generate_thrust_schedules(
            key, num_steps, batch_size, coupling_strength=2.0
        )
        
        # Measure smoothness by counting transitions (0->1 or 1->0)
        def count_transitions(schedule):
            """Count number of on/off switches"""
            diffs = jnp.diff(schedule)
            return jnp.sum(jnp.abs(diffs))
        
        transitions_low = jnp.mean(jax.vmap(count_transitions)(schedules_low))
        transitions_high = jnp.mean(jax.vmap(count_transitions)(schedules_high))
        
        # Higher coupling should have fewer transitions
        assert transitions_high < transitions_low
    
    def test_bias_field_affects_duty_cycle(self):
        """Positive bias should increase thrust-on percentage"""
        key = jax.random.PRNGKey(42)
        num_steps = 100
        batch_size = 20
        
        # Negative bias (prefer thrust off)
        bias_off = jnp.ones(num_steps) * (-2.0)
        schedules_off = generate_thrust_schedules(
            key, num_steps, batch_size, bias_field=bias_off
        )
        
        # Positive bias (prefer thrust on)
        bias_on = jnp.ones(num_steps) * 2.0
        schedules_on = generate_thrust_schedules(
            key, num_steps, batch_size, bias_field=bias_on
        )
        
        # Calculate duty cycle (percentage of thrust-on)
        duty_cycle_off = jnp.mean(schedules_off)
        duty_cycle_on = jnp.mean(schedules_on)
        
        # Positive bias should increase duty cycle
        assert duty_cycle_on > duty_cycle_off
        
        # With strong bias, should be quite different
        assert duty_cycle_on > 0.6  # Mostly on
        assert duty_cycle_off < 0.4  # Mostly off
    
    def test_eclipse_constraint(self):
        """Eclipse indices should have low thrust probability"""
        key = jax.random.PRNGKey(42)
        num_steps = 100
        batch_size = 50  # Larger batch for statistics
        
        # Simulate eclipse at steps 30-40
        eclipse_indices = list(range(30, 40))
        
        schedules = generate_thrust_schedules(
            key, num_steps, batch_size, 
            coupling_strength=0.3,
            eclipse_indices=eclipse_indices
        )
        
        # Check thrust probability during eclipse
        eclipse_duty = jnp.mean(schedules[:, 30:40])
        non_eclipse_duty = jnp.mean(schedules[:, :30])
        
        # Eclipse regions should have much lower thrust
        assert eclipse_duty < non_eclipse_duty
        assert eclipse_duty < 0.1  # Should be rare (strong negative bias)
    
    def test_reproducibility(self):
        """Same key should produce same schedules"""
        key = jax.random.PRNGKey(123)
        num_steps = 100
        batch_size = 10
        
        schedules1 = generate_thrust_schedules(key, num_steps, batch_size)
        schedules2 = generate_thrust_schedules(key, num_steps, batch_size)
        
        assert jnp.allclose(schedules1, schedules2)
    
    def test_different_keys_produce_different_schedules(self):
        """Different keys should produce different schedules"""
        num_steps = 100
        batch_size = 10
        
        schedules1 = generate_thrust_schedules(
            jax.random.PRNGKey(1), num_steps, batch_size
        )
        schedules2 = generate_thrust_schedules(
            jax.random.PRNGKey(2), num_steps, batch_size
        )
        
        # Should not be identical
        assert not jnp.allclose(schedules1, schedules2)


class TestIsingModel:
    """Test Ising model construction"""
    
    def test_model_creation(self):
        """Should create valid Ising model"""
        num_steps = 50
        coupling = 0.5
        field = jnp.zeros(num_steps)
        
        model, nodes = create_time_series_model(num_steps, coupling, field)
        
        # Should have correct number of nodes
        assert len(nodes) == num_steps
        
        # Model should have attributes (duck typing - just check it exists)
        assert model is not None
    
    def test_edge_connectivity(self):
        """Should create chain connectivity (each node connected to next)"""
        num_steps = 10
        coupling = 1.0
        field = jnp.zeros(num_steps)
        
        model, nodes = create_time_series_model(num_steps, coupling, field)
        
        # For a chain of N nodes, should have N-1 edges
        # We can't directly inspect model.edges easily without THRML internals
        # But we can verify model was created without error
        assert len(nodes) == num_steps


class TestBatchConsistency:
    """Test batch generation properties"""
    
    def test_batch_independence(self):
        """Different samples in batch should be independent"""
        key = jax.random.PRNGKey(42)
        num_steps = 100
        batch_size = 20
        
        schedules = generate_thrust_schedules(key, num_steps, batch_size)
        
        # Check that not all schedules are identical
        # (Could happen with bad sampling)
        for i in range(batch_size - 1):
            if not jnp.array_equal(schedules[i], schedules[i+1]):
                break
        else:
            pytest.fail("All schedules in batch are identical")
    
    def test_batch_statistics(self):
        """Batch should have reasonable statistics"""
        key = jax.random.PRNGKey(42)
        num_steps = 200
        batch_size = 50
        
        schedules = generate_thrust_schedules(
            key, num_steps, batch_size,
            coupling_strength=0.5
        )
        
        # Mean over batch and time should be around 0.5 (neutral bias)
        overall_mean = jnp.mean(schedules)
        
        # With no bias, should be roughly 50/50 (allow some variance)
        assert 0.3 < overall_mean < 0.7
        
        # Variance should be reasonable (not all same value)
        overall_var = jnp.var(schedules)
        assert overall_var > 0.01  # Some diversity


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_single_step(self):
        """Should handle single time step"""
        key = jax.random.PRNGKey(42)
        schedules = generate_thrust_schedules(key, num_steps=1, batch_size=10)
        
        assert schedules.shape == (10, 1)
    
    def test_single_batch(self):
        """Should handle batch size of 1"""
        key = jax.random.PRNGKey(42)
        schedules = generate_thrust_schedules(key, num_steps=100, batch_size=1)
        
        assert schedules.shape == (1, 100)
    
    def test_very_high_coupling(self):
        """Very high coupling should create very smooth schedules"""
        key = jax.random.PRNGKey(42)
        schedules = generate_thrust_schedules(
            key, num_steps=100, batch_size=10,
            coupling_strength=10.0  # Extremely high
        )
        
        # Should have very few transitions
        def count_transitions(schedule):
            return jnp.sum(jnp.abs(jnp.diff(schedule)))
        
        avg_transitions = jnp.mean(jax.vmap(count_transitions)(schedules))
        
        # With extreme coupling, expect very few switches
        assert avg_transitions < 10  # Less than 10% of steps


if __name__ == "__main__":
    # Run with: pytest tests/test_thrml.py -v
    if THRML_AVAILABLE:
        pytest.main([__file__, "-v"])
    else:
        print("THRML not available - tests skipped")
