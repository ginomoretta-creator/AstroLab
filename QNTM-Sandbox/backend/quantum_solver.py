"""
Quantum-Inspired Schedule Generation using Simulated Annealing
==============================================================

This module provides binary thrust schedule generation using D-Wave's
Neal simulated annealing sampler, which simulates quantum annealing behavior.

Key Features:
- Physics-aware external fields (bias toward optimal orbital phases)
- 1D Ising chain with ferromagnetic coupling (smooth schedules)
- Fuel budget filtering
- Energy-based trajectory evaluation

The Ising formulation:
    E = -J * Σ s_i * s_{i+1} - h_i * Σ s_i
    
Where:
    - s_i ∈ {-1, +1} (spin = thrust on/off)
    - J > 0 encourages smooth thrust arcs
    - h_i encodes physics priors (periapsis boost, arrival coast, etc.)

Author: ASL-Sandbox Team
"""

import sys
import os
import neal
import dimod
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, Optional, List, Tuple

# Add core to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
asl_root = os.path.dirname(project_root)
if asl_root not in sys.path:
    sys.path.insert(0, asl_root)

# Import physics-aware components from core
try:
    from core import (
        compute_physics_bias_field,
        compute_reference_trajectory_for_bias,
        update_bias_from_elite_samples,
        filter_schedules_by_fuel_budget,
        MU, EARTH_POS, MOON_POS,
        get_initial_state_4d
    )
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import core: {e}")
    CORE_AVAILABLE = False


class SimulatedQuantumAnnealer:
    """
    Quantum-inspired sampler for thrust schedule optimization.
    
    Uses D-Wave's Neal simulated annealing to find low-energy
    configurations of a 1D Ising model representing thrust decisions.
    """
    
    def __init__(
        self,
        num_reads: int = 100,
        num_sweeps: int = 1000,
        beta_range: Tuple[float, float] = None
    ):
        """
        Initialize the simulated annealer.
        
        Args:
            num_reads: Number of samples per annealing run
            num_sweeps: Number of Monte Carlo sweeps per read
            beta_range: (beta_min, beta_max) for annealing schedule
        """
        self.sampler = neal.SimulatedAnnealingSampler()
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
    
    def generate_thrust_schedules(
        self,
        num_steps: int,
        batch_size: int,
        coupling_strength: float = 1.0,
        bias: float = 0.0,
        physics_bias_field: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate binary thrust schedules using simulated annealing.
        
        The Ising model energy is:
            E = -J * Σ s_i * s_{i+1} - Σ h_i * s_i
        
        Args:
            num_steps: Number of time steps (qubits)
            batch_size: Number of samples to generate
            coupling_strength: J parameter (positive = smooth)
            bias: Global bias (positive = more thrust)
            physics_bias_field: (num_steps,) array of per-step biases
            
        Returns:
            Dictionary with:
                - schedules: (batch_size, num_steps) binary {0, 1}
                - energies: (batch_size,) Ising energies
                - h: External field dict
                - J: Coupling dict
        """
        # Build external field h
        if physics_bias_field is not None:
            h = {i: float(physics_bias_field[i]) for i in range(num_steps)}
        else:
            h = {i: bias for i in range(num_steps)}
        
        # Build coupling J (ferromagnetic chain)
        # Negative J in dimod convention encourages alignment
        J = {(i, i+1): -coupling_strength for i in range(num_steps - 1)}
        
        # Create BQM
        bqm = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.SPIN)
        
        # Sample
        sample_kwargs = {'num_reads': batch_size, 'num_sweeps': self.num_sweeps}
        if self.beta_range is not None:
            sample_kwargs['beta_range'] = self.beta_range
        
        response = self.sampler.sample(bqm, **sample_kwargs)
        
        # Extract samples
        samples_matrix = response.record.sample
        energies = response.record.energy
        
        # Map {-1, +1} to {0, 1}
        binary_schedules = (samples_matrix + 1) / 2.0
        
        return {
            "schedules": jnp.array(binary_schedules, dtype=jnp.float32),
            "energies": jnp.array(energies, dtype=jnp.float32),
            "h": h,
            "J": {str(k): v for k, v in J.items()}
        }
    
    def generate_physics_guided_schedules(
        self,
        num_steps: int,
        batch_size: int,
        coupling_strength: float = 1.0,
        initial_state: Optional[np.ndarray] = None,
        thrust_accel: float = 0.01,
        dt: float = 0.01,
        fuel_budget_fraction: float = 0.4,
        enable_3d: bool = False  # NEW: 3D support
    ) -> Dict[str, Any]:
        """
        Generate physics-aware thrust schedules with fuel budget filtering.

        Supports both 2D (4-state) and 3D (6-7 state) trajectories.

        Args:
            num_steps: Schedule length
            batch_size: Number of valid schedules to return
            coupling_strength: Ising smoothness
            initial_state: Initial orbit state (4D for 2D, 6-7D for 3D)
            thrust_accel: Thrust acceleration
            dt: Time step
            fuel_budget_fraction: Target thrust duty cycle
            enable_3d: If True, use 3D physics (NEW)

        Returns:
            Dictionary with schedules, energies, metadata
        """
        # Get initial state
        if initial_state is None:
            if CORE_AVAILABLE:
                initial_state = np.array(get_initial_state_4d(200.0))
            else:
                initial_state = np.array([0.017, 0, 0, 1.0])

        # Detect dimensionality from state size
        state_dim = len(initial_state)
        is_3d = enable_3d or (state_dim >= 6)

        # Compute physics-aware bias field
        if CORE_AVAILABLE and not is_3d:
            # 2D physics-aware bias (only available for 2D)
            ref_traj = compute_reference_trajectory_for_bias(
                num_steps, dt, thrust_accel,
                jnp.array(initial_state[:4]),
                fuel_budget_fraction
            )
            bias_field = compute_physics_bias_field(
                num_steps, ref_traj, fuel_budget_fraction
            )
            physics_bias = np.array(bias_field)
        else:
            # Simplified bias for 3D or when core unavailable
            # Linear decay: Start with higher thrust, decrease over time
            # This encourages early orbit raising
            physics_bias = np.linspace(0.5, -1.0, num_steps)
            physics_bias += (fuel_budget_fraction - 0.5) * 2.0

            if is_3d:
                print("[ISING-3D] Using simplified bias field (physics-aware bias is 2D only)")
        
        # Oversample for filtering
        oversample_factor = 3
        raw_result = self.generate_thrust_schedules(
            num_steps,
            batch_size * oversample_factor,
            coupling_strength,
            physics_bias_field=physics_bias
        )
        
        raw_schedules = raw_result['schedules']
        raw_energies = raw_result['energies']
        
        # Filter by fuel budget
        if CORE_AVAILABLE:
            valid_schedules, valid_mask = filter_schedules_by_fuel_budget(
                raw_schedules,
                max_thrust_fraction=fuel_budget_fraction + 0.2,
                min_thrust_fraction=max(0.05, fuel_budget_fraction - 0.2)
            )
            valid_energies = raw_energies[valid_mask]
        else:
            thrust_fractions = jnp.mean(raw_schedules, axis=1)
            valid_mask = (thrust_fractions >= 0.1) & (thrust_fractions <= 0.7)
            valid_schedules = raw_schedules[valid_mask]
            valid_energies = raw_energies[valid_mask]
        
        # Take requested batch size
        if len(valid_schedules) >= batch_size:
            output_schedules = valid_schedules[:batch_size]
            output_energies = valid_energies[:batch_size]
        else:
            # Pad if needed
            n_valid = len(valid_schedules)
            if n_valid > 0:
                repeats = (batch_size // n_valid) + 1
                output_schedules = jnp.tile(valid_schedules, (repeats, 1))[:batch_size]
                output_energies = jnp.tile(valid_energies, repeats)[:batch_size]
            else:
                output_schedules = raw_schedules[:batch_size]
                output_energies = raw_energies[:batch_size]
        
        return {
            "schedules": output_schedules,
            "energies": output_energies,
            "h": raw_result['h'],
            "J": raw_result['J'],
            "metadata": {
                "physics_bias_field": physics_bias.tolist(),
                "coupling_strength": coupling_strength,
                "fuel_budget_fraction": fuel_budget_fraction,
                "valid_fraction": float(jnp.sum(valid_mask) / len(valid_mask)),
                "physics_guided": CORE_AVAILABLE and not is_3d,
                "dimension": "3D" if is_3d else "2D",  # NEW
                "state_size": state_dim  # NEW
            }
        }


class IterativeQuantumOptimizer:
    """
    Iterative optimizer using quantum-inspired annealing.
    
    Similar to Cross-Entropy Method but using annealing samples:
    1. Sample from current Ising model
    2. Evaluate trajectory costs
    3. Update bias field toward elite patterns
    """
    
    def __init__(
        self,
        num_steps: int,
        coupling_strength: float = 1.0,
        elite_fraction: float = 0.1,
        learning_rate: float = 0.3,
        fuel_budget_fraction: float = 0.4,
        initial_state: Optional[np.ndarray] = None,
        thrust_accel: float = 0.01,
        dt: float = 0.01,
        enable_3d: bool = False  # NEW: 3D support
    ):
        """Initialize optimizer."""
        self.num_steps = num_steps
        self.coupling_strength = coupling_strength
        self.elite_fraction = elite_fraction
        self.learning_rate = learning_rate
        self.fuel_budget_fraction = fuel_budget_fraction
        self.thrust_accel = thrust_accel
        self.dt = dt
        self.enable_3d = enable_3d  # NEW

        # Initialize state
        if initial_state is None:
            if CORE_AVAILABLE:
                self.initial_state = np.array(get_initial_state_4d(200.0))
            else:
                self.initial_state = np.array([0.017, 0, 0, 1.0])
        else:
            # Keep full state for 3D (6-7 components)
            if enable_3d or len(initial_state) >= 6:
                self.initial_state = initial_state
                self.enable_3d = True
            else:
                self.initial_state = initial_state[:4] if len(initial_state) > 4 else initial_state

        # Detect dimensionality
        state_dim = len(self.initial_state)
        is_3d = self.enable_3d or (state_dim >= 6)

        # Initialize bias field
        if CORE_AVAILABLE and not is_3d:
            # Physics-aware bias for 2D
            ref_traj = compute_reference_trajectory_for_bias(
                num_steps, dt, thrust_accel,
                jnp.array(self.initial_state[:4]),
                fuel_budget_fraction
            )
            self.bias_field = np.array(compute_physics_bias_field(num_steps, ref_traj, fuel_budget_fraction))
        else:
            # Simplified bias for 3D
            self.bias_field = np.linspace(0.2, -0.5, num_steps)
            if is_3d:
                print("[ISING-3D-OPTIMIZER] Initialized with 3D state, using simplified bias")

        self.annealer = SimulatedQuantumAnnealer()
        self.iteration = 0
        self.best_cost = float('inf')
        self.best_schedule = None
        self.best_energy = None
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample schedules from current distribution."""
        return self.annealer.generate_thrust_schedules(
            self.num_steps,
            batch_size,
            self.coupling_strength,
            physics_bias_field=self.bias_field
        )
    
    def update(
        self,
        schedules: jnp.ndarray,
        costs: jnp.ndarray,
        energies: jnp.ndarray
    ):
        """Update distribution based on evaluated schedules."""
        # Select elite
        k_elite = max(1, int(len(schedules) * self.elite_fraction))
        elite_indices = jnp.argsort(costs)[:k_elite]
        elite_schedules = schedules[elite_indices]
        
        # Update bias field
        if CORE_AVAILABLE:
            self.bias_field = np.array(update_bias_from_elite_samples(
                elite_schedules, 
                jnp.array(self.bias_field), 
                self.learning_rate
            ))
        else:
            elite_mean = np.mean(np.array(elite_schedules), axis=0)
            target_bias = (elite_mean - 0.5) * 4.0
            self.bias_field = (1 - self.learning_rate) * self.bias_field + self.learning_rate * target_bias
        
        # Track best
        best_idx = elite_indices[0]
        if costs[best_idx] < self.best_cost:
            self.best_cost = float(costs[best_idx])
            self.best_schedule = np.array(schedules[best_idx])
            self.best_energy = float(energies[best_idx])
        
        self.iteration += 1
    
    def get_state(self) -> Dict[str, Any]:
        """Get current optimizer state."""
        return {
            'iteration': self.iteration,
            'bias_field': self.bias_field.tolist(),
            'best_cost': self.best_cost,
            'best_energy': self.best_energy,
            'best_schedule': self.best_schedule.tolist() if self.best_schedule is not None else None,
            'coupling_strength': self.coupling_strength,
            'fuel_budget_fraction': self.fuel_budget_fraction
        }


# =============================================================================
# Comparison Utilities
# =============================================================================

def generate_random_schedules(
    num_steps: int,
    batch_size: int,
    thrust_probability: float = 0.5
) -> jnp.ndarray:
    """
    Generate random baseline schedules (Bernoulli).
    
    Args:
        num_steps: Schedule length
        batch_size: Number of schedules
        thrust_probability: Probability of thrust at each step
        
    Returns:
        schedules: (batch_size, num_steps) binary
    """
    key = jax.random.PRNGKey(int(np.random.randint(0, 100000)))
    return jax.random.bernoulli(key, thrust_probability, (batch_size, num_steps)).astype(jnp.float32)


def compare_methods(
    num_steps: int,
    batch_size: int,
    coupling_strength: float = 1.0,
    fuel_budget_fraction: float = 0.4
) -> Dict[str, Dict[str, Any]]:
    """
    Compare quantum-annealing vs random schedule generation.
    
    Returns statistics for both methods.
    """
    annealer = SimulatedQuantumAnnealer()
    
    # Quantum-guided
    quantum_result = annealer.generate_physics_guided_schedules(
        num_steps, batch_size, coupling_strength,
        fuel_budget_fraction=fuel_budget_fraction
    )
    
    # Random baseline
    random_schedules = generate_random_schedules(
        num_steps, batch_size, thrust_probability=fuel_budget_fraction
    )
    
    # Statistics
    quantum_thrust_fracs = jnp.mean(quantum_result['schedules'], axis=1)
    random_thrust_fracs = jnp.mean(random_schedules, axis=1)
    
    return {
        'quantum': {
            'mean_thrust_fraction': float(jnp.mean(quantum_thrust_fracs)),
            'std_thrust_fraction': float(jnp.std(quantum_thrust_fracs)),
            'mean_energy': float(jnp.mean(quantum_result['energies'])),
            'physics_guided': quantum_result['metadata']['physics_guided']
        },
        'random': {
            'mean_thrust_fraction': float(jnp.mean(random_thrust_fracs)),
            'std_thrust_fraction': float(jnp.std(random_thrust_fracs)),
            'mean_energy': None  # No energy model
        }
    }


# =============================================================================
# Module Exports
# =============================================================================

# Import JAX for random schedules
import jax

__all__ = [
    'SimulatedQuantumAnnealer',
    'IterativeQuantumOptimizer',
    'generate_random_schedules',
    'compare_methods',
    'CORE_AVAILABLE'
]


if __name__ == "__main__":
    # Test
    annealer = SimulatedQuantumAnnealer()
    
    print("Testing basic schedule generation...")
    result = annealer.generate_thrust_schedules(num_steps=20, batch_size=5, coupling_strength=2.0, bias=0.1)
    print(f"Schedules shape: {result['schedules'].shape}")
    print(f"Mean energy: {jnp.mean(result['energies']):.2f}")
    
    print("\nTesting physics-guided generation...")
    physics_result = annealer.generate_physics_guided_schedules(
        num_steps=100, batch_size=10, coupling_strength=1.0, fuel_budget_fraction=0.4
    )
    print(f"Schedules shape: {physics_result['schedules'].shape}")
    print(f"Mean thrust fraction: {jnp.mean(physics_result['schedules']):.3f}")
    print(f"Physics guided: {physics_result['metadata']['physics_guided']}")
    
    print("\nComparing methods...")
    comparison = compare_methods(100, 50)
    print(f"Quantum mean thrust: {comparison['quantum']['mean_thrust_fraction']:.3f}")
    print(f"Random mean thrust: {comparison['random']['mean_thrust_fraction']:.3f}")
