# Classical vs Hybrid Quantum-Classical Comparison Framework

## Overview

This document outlines the comparison framework between pure classical optimization and hybrid quantum-classical approaches for low-thrust trajectory optimization in the Earth-Moon system.

## Methods

### 1. Classical Method: Cross-Entropy Method (CEM)

**Type**: Pure classical iterative optimization

**Algorithm**:
1. Initialize probability distribution P(x) for thrust schedule (uniform 0.4)
2. Sample N candidate thrust schedules from P(x)
3. Propagate trajectories using CR3BP dynamics
4. Evaluate cost function (Moon proximity, fuel, velocity)
5. Select elite samples (top 10% lowest cost)
6. Update P(x) based on elite samples using learning rate α
7. Repeat until convergence or max iterations

**Key Characteristics**:
- Deterministic convergence (given same seed)
- Physics-aware bias field guides sampling
- No hardware requirements beyond CPU/GPU
- Scales with O(N × num_iterations) propagations

**Implementation**: `THRML-Sandbox/backend/server.py` lines 166-176

### 2. Hybrid Quantum-Classical Method

**Type**: Quantum annealing + classical refinement

**Algorithm**:
1. **Quantum Phase**: Map thrust schedule to Ising spin model
   - Variables: s_i ∈ {-1, +1} for each timestep
   - Energy: E = -Σ J_ij s_i s_j - Σ h_i s_i
   - J > 0 (coupling) encourages smooth thrust arcs
   - h_i (bias) incorporates physics knowledge
   - Solve using quantum annealer (simulated: D-Wave Neal)
2. **Classical Phase**: Refine samples using CEM bias update
3. Propagate and evaluate as in classical method

**Key Characteristics**:
- Quantum hardware can accelerate Ising sampling
- Hybrid approach combines quantum exploration + classical exploitation
- Currently simulated on classical hardware (D-Wave Neal library)
- Real quantum hardware would reduce sampling time

**Implementation**: `THRML-Sandbox/backend/server.py` lines 178-195

## Comparison Metrics

### Performance Metrics

1. **Success Rate**: Fraction of trajectories reaching Moon SOI (r < 66,100 km) with low velocity (v < 1 km/s)
2. **Solution Quality**:
   - Final distance to Moon
   - Fuel consumption (final mass)
   - Trajectory smoothness (jerk metric)
3. **Convergence Speed**:
   - Iterations to reach success
   - Wall-clock time to solution

### Quantum vs Classical Analysis

Since both methods currently run on classical hardware, the comparison focuses on:

**Classical Hardware Baseline**:
- Time per iteration measured directly
- Scales linearly with batch_size and num_steps

**Projected Quantum Hardware Benefits**:

1. **Ising Sampling Time**:
   - Classical simulation (Neal): O(N × annealing_steps) ~ 10-100ms for N=5000
   - D-Wave Advantage: Fixed annealing time ~20μs (1000x faster)
   - Speedup: ~500-5000x for sampling phase

2. **Total Speedup Estimate**:
   - Sampling phase: ~10-20% of total time (rest is propagation)
   - Expected total speedup: ~2-10x depending on problem size

3. **Scaling Analysis**:
   ```
   Classical: T_total = T_sample + T_propagate
              T_sample ∝ N × M (N=steps, M=annealing_iterations)
              T_propagate ∝ N × B (B=batch_size)

   Quantum:   T_total = T_quantum + T_propagate
              T_quantum ≈ 20μs (constant)
              T_propagate ∝ N × B (unchanged)

   Speedup factor = T_classical / T_quantum
                  ≈ (N × M) / 20μs  (for sampling only)
                  ≈ 2-10x (end-to-end)
   ```

## Success Criteria: Moon SOI Arrival

### Definition

A trajectory is considered **successful** if:
1. Distance to Moon < 66,100 km (Lunar SOI radius)
2. Relative velocity < 1.0 km/s (capture velocity)
3. Minimum timestep requirement met (avoid early termination)

### Implementation

```python
# In physics_core.py (to be implemented)
def check_lunar_soi_arrival(
    trajectory: jnp.ndarray,
    soi_radius: float = 0.172,  # 66,100 km normalized
    max_velocity_kms: float = 1.0
) -> Tuple[bool, int, float]:
    """
    Returns:
        (success, arrival_step, relative_velocity_kms)
    """
    moon_pos = jnp.array([1 - MU, 0])

    for i, state in enumerate(trajectory):
        pos = state[:2]
        vel = state[2:4]

        dist = jnp.linalg.norm(pos - moon_pos)
        vel_mag = jnp.linalg.norm(vel) * V_STAR_KMS

        if dist < soi_radius and vel_mag < max_velocity_kms:
            return (True, i, vel_mag)

    return (False, -1, 0.0)
```

## Cost Function

The optimization cost balances multiple objectives:

```python
cost = (
    w1 * min_distance_to_moon +      # Approach Moon
    w2 * final_distance_to_moon +    # End near Moon
    w3 * (-approach_progress) +      # Reward getting closer
    w4 * velocity_at_moon +          # Penalize high velocity (prevent flyby)
    w5 * fuel_consumption +          # Minimize fuel
    + penalties                       # Collision, escape, fuel limits
)
```

Weights: w1=0.5, w2=0.3, w3=0.2, w4=0.5, w5=0.1

## Benchmark Workflow

### Running Comparisons

```bash
# Start backend
cd THRML-Sandbox/backend
python launcher.py

# In another terminal, run comparison
curl -X POST http://localhost:8080/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "method": "classical",
    "num_steps": 5000,
    "batch_size": 100,
    "num_iterations": 30
  }'

# Then run hybrid
curl -X POST http://localhost:8080/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "method": "hybrid",
    "num_steps": 5000,
    "batch_size": 100,
    "num_iterations": 30
  }'
```

### Results Format

```json
{
  "iteration": 30,
  "method": "classical",
  "success_rate": 0.15,
  "best_cost": 0.045,
  "mean_cost": 0.23,
  "best_distance": 0.05,
  "best_thrust_fraction": 0.42,
  "wall_time_seconds": 12.3
}
```

## Future Work

### Near Term
1. Implement SOI detection with early termination
2. Add velocity constraint checking
3. Improve cost function for capture (not just approach)
4. Add convergence tracking plots

### Long Term
1. Test on real D-Wave quantum hardware
2. Hybrid warm-starting: quantum → CasADi refinement
3. Multi-objective optimization (fuel vs time Pareto front)
4. Adaptive annealing schedules based on physics

## References

- Cross-Entropy Method: Rubinstein & Kroese (2004)
- Quantum Annealing: D-Wave Systems documentation
- Lunar Transfers: Kluever (2010), Conway (2010)
- CR3BP Dynamics: Szebehely (1967)
