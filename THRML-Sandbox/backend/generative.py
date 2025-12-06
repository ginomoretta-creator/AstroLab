"""
Generative Model for Thrust Schedule Sampling
==============================================

This module provides physics-guided thrust schedule generation using
THRML (a JAX-based probabilistic graphical model library) for Gibbs
sampling on 1D Ising chains.

Key Features:
- Physics-aware bias fields (periapsis/apoapsis, arrival coast, fuel budget)
- Cross-Entropy Method style iterative refinement
- Fuel constraint filtering
- Fallback to random sampling if THRML unavailable

Author: ASL-Sandbox Team
"""

import sys
import os
import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional, Dict, Any

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
thrml_path = os.path.join(project_root, 'thrml-main')

if thrml_path not in sys.path:
    sys.path.append(thrml_path)

# Add core to path
asl_root = os.path.dirname(project_root)
if asl_root not in sys.path:
    sys.path.insert(0, asl_root)

# Import physics-aware energy model from core
try:
    from core import (
        compute_physics_bias_field,
        compute_reference_trajectory_for_bias,
        update_bias_from_elite_samples,
        filter_schedules_by_fuel_budget,
        PhysicsGuidedScheduleGenerator,
        MU, EARTH_POS, MOON_POS,
        get_initial_state_4d
    )
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import core energy model: {e}")
    CORE_AVAILABLE = False

# Import THRML
try:
    from thrml import SpinNode, Block, SamplingSchedule, sample_states
    from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
    THRML_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import thrml: {e}")
    THRML_AVAILABLE = False
    # Mock classes for development
    class SpinNode: pass
    class Block: pass
    class SamplingSchedule: pass
    class IsingEBM: pass
    class IsingSamplingProgram: pass


# =============================================================================
# Core Ising Model Construction
# =============================================================================

def create_ising_chain_model(
    num_steps: int, 
    coupling_strength: float, 
    external_field: jnp.ndarray,
    beta: float = 1.0
):
    """
    Creates an Ising model representing a time series of thrust decisions.
    
    The energy function is:
        E = -β * [Σ J_ij s_i s_j + Σ h_i s_i]
    
    Where:
        - s_i ∈ {-1, +1} (spin variables, mapped to thrust on/off)
        - J > 0 encourages alignment (smooth thrust arcs)
        - h_i is the external field (physics-aware bias)
        - β is inverse temperature (controls sharpness)
    
    Args:
        num_steps: Number of time steps (nodes)
        coupling_strength: J parameter (positive = smooth schedules)
        external_field: h parameter array, shape (num_steps,)
        beta: Inverse temperature (default: 1.0)
        
    Returns:
        model: IsingEBM instance
        nodes: List of SpinNodes
    """
    if not THRML_AVAILABLE:
        raise ImportError("THRML is not available. Cannot create Ising model.")
    
    # Create spin nodes
    nodes = [SpinNode() for _ in range(num_steps)]
    
    # 1D Chain edges (nearest neighbors)
    edges = [(nodes[i], nodes[i+1]) for i in range(num_steps - 1)]
    
    # Coupling weights (ferromagnetic for smoothness)
    weights = jnp.ones((len(edges),)) * coupling_strength
    
    # External field (bias)
    biases = external_field
    
    # Create model
    model = IsingEBM(nodes, edges, biases, weights, jnp.array(beta))
    
    return model, nodes


# =============================================================================
# THRML-Based Schedule Generation
# =============================================================================

def generate_thrust_schedules_thrml(
    key: jax.random.PRNGKey,
    num_steps: int,
    batch_size: int,
    coupling_strength: float = 0.5,
    bias_field: Optional[jnp.ndarray] = None,
    n_warmup: int = 50,
    n_samples_per_chain: int = 1,
    steps_per_sample: int = 2
) -> jnp.ndarray:
    """
    Generate thrust schedules using THRML Gibbs sampling.
    
    Args:
        key: JAX random key
        num_steps: Length of schedule
        batch_size: Number of schedules to generate
        coupling_strength: Ising coupling J (higher = smoother)
        bias_field: External field h (shape: num_steps)
        n_warmup: Gibbs warmup iterations
        n_samples_per_chain: Samples to take per chain
        steps_per_sample: Gibbs steps between samples
        
    Returns:
        schedules: (batch_size, num_steps) binary array {0, 1}
    """
    if not THRML_AVAILABLE:
        # Fallback to random
        print("THRML not available, using random sampling fallback")
        return jax.random.bernoulli(key, 0.5, (batch_size, num_steps)).astype(jnp.float32)
    
    # Default bias field if not provided
    if bias_field is None:
        bias_field = jnp.zeros(num_steps)
    
    # Create Ising model
    model, nodes = create_ising_chain_model(num_steps, coupling_strength, bias_field)
    
    # Setup sampling program (checkerboard for 1D chain)
    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    
    # Run sampling for each chain in batch
    def run_single_chain(chain_key):
        k_init, k_samp = jax.random.split(chain_key)
        init_state = hinton_init(k_init, model, free_blocks, ())
        schedule = SamplingSchedule(n_warmup=n_warmup, n_samples=n_samples_per_chain, steps_per_sample=steps_per_sample)
        samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
        # Return last sample (or first if n_samples=1)
        return samples[0][-1] if n_samples_per_chain > 1 else samples[0][0]
    
    # Split keys for batch
    keys = jax.random.split(key, batch_size)
    
    # Run batch (vmap)
    batch_samples = jax.vmap(run_single_chain)(keys)
    
    # Map spins {-1, +1} to {0, 1}
    # -1 -> 0 (coast), +1 -> 1 (thrust)
    thrust_schedules = (batch_samples + 1) / 2
    
    return thrust_schedules.astype(jnp.float32)


# =============================================================================
# Physics-Guided Schedule Generation
# =============================================================================

def generate_physics_guided_schedules(
    key: jax.random.PRNGKey,
    num_steps: int,
    batch_size: int,
    coupling_strength: float = 0.5,
    initial_state: Optional[jnp.ndarray] = None,
    thrust_accel: float = 0.01,
    dt: float = 0.01,
    fuel_budget_fraction: float = 0.4,
    method: str = "thrml"
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Generate physics-guided thrust schedules with fuel budget filtering.
    
    This is the main entry point for schedule generation that:
    1. Computes a reference trajectory for orbital phase detection
    2. Creates physics-aware bias field
    3. Samples from Ising model with THRML
    4. Filters schedules by fuel budget
    
    Args:
        key: JAX random key
        num_steps: Length of schedule
        batch_size: Number of valid schedules to return
        coupling_strength: Ising smoothness parameter
        initial_state: Initial orbit state (defaults to 200km LEO)
        thrust_accel: Thrust acceleration magnitude (normalized)
        dt: Time step (normalized)
        fuel_budget_fraction: Target thrust duty cycle
        method: "thrml" or "random"
        
    Returns:
        schedules: (batch_size, num_steps) binary schedules
        metadata: Dictionary with bias field and statistics
    """
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Get initial state
    if initial_state is None:
        initial_state = get_initial_state_4d(200.0) if CORE_AVAILABLE else jnp.array([0.017, 0, 0, 1.0])
    
    # Compute physics-aware bias field
    if CORE_AVAILABLE:
        # Generate reference trajectory
        ref_traj = compute_reference_trajectory_for_bias(
            num_steps, dt, thrust_accel, initial_state, fuel_budget_fraction
        )
        
        # Compute bias field
        bias_field = compute_physics_bias_field(
            num_steps,
            ref_traj,
            fuel_budget_fraction=fuel_budget_fraction,
            arrival_coast_fraction=0.15,
            periapsis_boost=2.0,
            apoapsis_penalty=-1.0,
            arrival_coast_strength=-3.0
        )
    else:
        # Simple linear bias (more thrust early, coast late)
        bias_field = jnp.linspace(0.5, -1.0, num_steps)
        bias_field = bias_field + (fuel_budget_fraction - 0.5) * 2.0
    
    # Generate more schedules than needed (for filtering)
    oversample_factor = 3
    oversample_size = batch_size * oversample_factor
    
    if method == "thrml" and THRML_AVAILABLE:
        raw_schedules = generate_thrust_schedules_thrml(
            k2, num_steps, oversample_size, coupling_strength, bias_field
        )
    else:
        # Random sampling with bias
        # Use bias field to modulate probability
        probs = jax.nn.sigmoid(bias_field)  # Convert bias to probability
        raw_schedules = jax.random.bernoulli(k2, probs, (oversample_size, num_steps)).astype(jnp.float32)
    
    # Filter by fuel budget
    if CORE_AVAILABLE:
        valid_schedules, valid_mask = filter_schedules_by_fuel_budget(
            raw_schedules,
            max_thrust_fraction=fuel_budget_fraction + 0.2,
            min_thrust_fraction=max(0.05, fuel_budget_fraction - 0.2)
        )
    else:
        # Simple filter
        thrust_fractions = jnp.mean(raw_schedules, axis=1)
        valid_mask = (thrust_fractions >= 0.1) & (thrust_fractions <= 0.7)
        valid_schedules = raw_schedules[valid_mask]
    
    # Take requested batch size
    if len(valid_schedules) >= batch_size:
        output_schedules = valid_schedules[:batch_size]
    else:
        # Not enough valid schedules, pad with what we have
        n_valid = len(valid_schedules)
        if n_valid > 0:
            # Repeat valid schedules to fill batch
            repeats = (batch_size // n_valid) + 1
            output_schedules = jnp.tile(valid_schedules, (repeats, 1))[:batch_size]
        else:
            # Fallback to unfiltered (shouldn't happen)
            output_schedules = raw_schedules[:batch_size]
    
    # Compute statistics
    metadata = {
        'bias_field': bias_field,
        'coupling_strength': coupling_strength,
        'mean_thrust_fraction': float(jnp.mean(output_schedules)),
        'valid_fraction': float(jnp.sum(valid_mask) / len(valid_mask)),
        'method': method,
        'physics_guided': CORE_AVAILABLE
    }
    
    return output_schedules, metadata


# =============================================================================
# Legacy API (Backwards Compatible)
# =============================================================================

def generate_thrust_schedules(
    key: jax.random.PRNGKey,
    num_steps: int,
    batch_size: int,
    coupling_strength: float = 0.5,
    eclipse_indices: List[int] = [],
    perigee_indices: List[int] = [],
    bias_field: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Legacy API for thrust schedule generation.
    
    Maintained for backwards compatibility with existing code.
    Internally uses physics-guided generation if core is available.
    
    Args:
        key: JAX random key
        num_steps: Length of schedule
        batch_size: Number of schedules
        coupling_strength: Ising coupling
        eclipse_indices: [DEPRECATED] Use physics-aware bias instead
        perigee_indices: [DEPRECATED] Use physics-aware bias instead
        bias_field: Optional override bias field
        
    Returns:
        schedules: (batch_size, num_steps) binary schedules
    """
    # If explicit bias field provided, use it directly
    if bias_field is not None:
        field = bias_field.copy()
    else:
        # Build bias field from legacy parameters
        field = jnp.zeros(num_steps)
        
        if eclipse_indices:
            eclipse_arr = jnp.array(eclipse_indices)
            field = field.at[eclipse_arr].set(-10.0)  # Strong coast during eclipse
        
        if perigee_indices:
            perigee_arr = jnp.array(perigee_indices)
            field = field.at[perigee_arr].add(2.0)  # Encourage thrust at perigee
    
    # Generate using THRML or random
    if THRML_AVAILABLE:
        schedules = generate_thrust_schedules_thrml(
            key, num_steps, batch_size, coupling_strength, field
        )
    else:
        # Random with bias
        probs = jax.nn.sigmoid(field)
        schedules = jax.random.bernoulli(key, probs, (batch_size, num_steps)).astype(jnp.float32)
    
    return schedules


# =============================================================================
# Iterative Refinement (Cross-Entropy Method)
# =============================================================================

class IterativeThrustOptimizer:
    """
    Cross-Entropy Method style optimizer for thrust schedules.
    
    This class manages iterative refinement where:
    1. Sample schedules from current distribution
    2. Evaluate costs (via trajectory propagation)
    3. Select elite (best) schedules
    4. Update distribution toward elite patterns
    """
    
    def __init__(
        self,
        num_steps: int,
        coupling_strength: float = 0.5,
        elite_fraction: float = 0.1,
        learning_rate: float = 0.3,
        fuel_budget_fraction: float = 0.4,
        initial_state: Optional[jnp.ndarray] = None,
        thrust_accel: float = 0.01,
        dt: float = 0.01
    ):
        """
        Initialize optimizer.
        
        Args:
            num_steps: Schedule length
            coupling_strength: Ising smoothness
            elite_fraction: Fraction of best schedules to use for update
            learning_rate: How fast to update distribution
            fuel_budget_fraction: Target thrust duty cycle
            initial_state: Initial orbit state
            thrust_accel: Thrust magnitude
            dt: Time step
        """
        self.num_steps = num_steps
        self.coupling_strength = coupling_strength
        self.elite_fraction = elite_fraction
        self.learning_rate = learning_rate
        self.fuel_budget_fraction = fuel_budget_fraction
        self.thrust_accel = thrust_accel
        self.dt = dt
        
        # Initialize state
        if initial_state is None:
            self.initial_state = get_initial_state_4d(200.0) if CORE_AVAILABLE else jnp.array([0.017, 0, 0, 1.0])
        else:
            self.initial_state = initial_state[:4] if len(initial_state) > 4 else initial_state
        
        # Initialize bias field
        if CORE_AVAILABLE:
            ref_traj = compute_reference_trajectory_for_bias(
                num_steps, dt, thrust_accel, self.initial_state, fuel_budget_fraction
            )
            self.bias_field = compute_physics_bias_field(num_steps, ref_traj, fuel_budget_fraction)
        else:
            self.bias_field = jnp.linspace(0.2, -0.5, num_steps)
        
        self.iteration = 0
        self.best_cost = float('inf')
        self.best_schedule = None
    
    def sample(self, key: jax.random.PRNGKey, batch_size: int) -> jnp.ndarray:
        """Sample schedules from current distribution."""
        if THRML_AVAILABLE:
            return generate_thrust_schedules_thrml(
                key, self.num_steps, batch_size, self.coupling_strength, self.bias_field
            )
        else:
            probs = jax.nn.sigmoid(self.bias_field)
            return jax.random.bernoulli(key, probs, (batch_size, self.num_steps)).astype(jnp.float32)
    
    def update(
        self,
        schedules: jnp.ndarray,
        costs: jnp.ndarray,
        trajectories: Optional[jnp.ndarray] = None
    ):
        """
        Update distribution based on evaluated schedules.
        
        Args:
            schedules: (batch_size, num_steps) evaluated schedules
            costs: (batch_size,) cost for each schedule
            trajectories: Optional (batch_size, N, 4) trajectories for orbital analysis
        """
        # Select elite
        k_elite = max(1, int(len(schedules) * self.elite_fraction))
        elite_indices = jnp.argsort(costs)[:k_elite]
        elite_schedules = schedules[elite_indices]
        
        # Update bias field
        if CORE_AVAILABLE:
            elite_trajs = trajectories[elite_indices] if trajectories is not None else None
            self.bias_field = update_bias_from_elite_samples(
                elite_schedules, self.bias_field, self.learning_rate
            )
        else:
            # Simple mean-based update
            elite_mean = jnp.mean(elite_schedules, axis=0)
            target_bias = (elite_mean - 0.5) * 4.0
            self.bias_field = (1 - self.learning_rate) * self.bias_field + self.learning_rate * target_bias
        
        # Track best
        best_idx = elite_indices[0]
        if costs[best_idx] < self.best_cost:
            self.best_cost = float(costs[best_idx])
            self.best_schedule = schedules[best_idx]
        
        self.iteration += 1
    
    def get_state(self) -> Dict[str, Any]:
        """Get current optimizer state for serialization/streaming."""
        return {
            'iteration': self.iteration,
            'bias_field': self.bias_field.tolist(),
            'best_cost': self.best_cost,
            'best_schedule': self.best_schedule.tolist() if self.best_schedule is not None else None,
            'coupling_strength': self.coupling_strength,
            'fuel_budget_fraction': self.fuel_budget_fraction
        }


# =============================================================================
# Module Info
# =============================================================================

def get_capabilities() -> Dict[str, bool]:
    """Return available capabilities of this module."""
    return {
        'thrml_available': THRML_AVAILABLE,
        'core_available': CORE_AVAILABLE,
        'physics_guided': CORE_AVAILABLE,
        'iterative_refinement': True
    }


__all__ = [
    'generate_thrust_schedules',
    'generate_thrust_schedules_thrml',
    'generate_physics_guided_schedules',
    'create_ising_chain_model',
    'IterativeThrustOptimizer',
    'get_capabilities',
    'THRML_AVAILABLE',
    'CORE_AVAILABLE'
]
