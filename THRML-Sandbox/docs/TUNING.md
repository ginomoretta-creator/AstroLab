# Hyperparameter Tuning Guide

This document explains the rationale behind key hyperparameters in the Cislunar Trajectory Sandbox and provides guidance for tuning them.

---

## Overview

The optimization process involves two main components:
1. **THRML Sampler**: Generates thrust schedules using probabilistic inference
2. **Iterative Refinement**: Updates bias field based on best trajectories

Both have parameters that significantly affect convergence behavior.

---

## Simulation Parameters

### `num_steps` (Time Resolution)
- **Default**: 5000
- **Range**: 500 - 20,000
- **What it controls**: Discretization of the trajectory (finer = more accurate, slower)
- **Typical values**:
  - Quick exploration: 1000
  - Standard runs: 5000
  - High-fidelity: 10,000+
- **Trade-offs**: 
  - ↑ Higher → Better temporal resolution, slower physics, larger THRML graph
  - ↓ Lower → Faster, but may miss critical thrust windows

### `dt` (Integration Time Step)
- **Default**: 0.01 (normalized time units)
- **Physical equivalent**: ~1 hour real-time
- **Range**: 0.005 - 0.05
- **What it controls**: RK4 integrator step size
- **Typical values**:
  - Stable orbits: 0.01 - 0.02
  - Close approaches: 0.005 (need fine resolution near Moon)
- **Validation**: Check energy conservation (Jacobi constant should be ~stable)

### `batch_size` (Exploration breadth)
- **Default**: 50
- **Range**: 10 - 200
- **What it controls**: Number of parallel trajectory candidates per iteration
- **Typical values**:
  - Fast testing: 20
  - Production: 50-100
  - GPU available: 200+
- **Trade-offs**:
  - ↑ Higher → Better exploration, slower per iteration
  - ↓ Lower → Faster, but may miss good solutions

### `num_iterations` (Optimization depth)
- **Default**: 50
- **Range**: 10 - 200
- **What it controls**: How many refinement cycles to run
- **Typical values**:
  - Quick check: 10
  - Standard: 50
  - Convergence study: 100+
- **Stopping criterion**: Currently fixed, but could check for cost plateau

---

## THRML Sampling Parameters

### `coupling_strength` (Smoothness constraint)
- **Default**: 0.5
- **Range**: 0.0 - 2.0
- **What it controls**: Ising model coupling J (favors neighboring steps having same thrust state)
- **Physics rationale**: Avoid rapid thrust switching (fuel-inefficient, unrealistic)
- **Typical values**:
  - No constraint: 0.0 (random-like)
  - Light smoothing: 0.3 - 0.5
  - Strong smoothing: 1.0 - 2.0
- **Effect on schedules**:
  ```
  J = 0.0:  1 0 1 1 0 1 0 0 1 1  (noisy)
  J = 0.5:  1 1 1 0 0 0 1 1 1 0  (some structure)
  J = 2.0:  1 1 1 1 1 0 0 0 0 0  (very smooth)
  ```

### `n_warmup` (Gibbs burn-in)
- **Default**: 50 (hardcoded in `generative.py:125`)
- **Range**: 10 - 200
- **What it controls**: Number of Gibbs sweeps before sampling
- **Why needed**: Let the Markov chain reach equilibrium distribution
- **Tuning guidance**:
  - Small graphs (num_steps < 1000): 20-30 sufficient
  - Large graphs (num_steps > 5000): 50-100 recommended
- **Validation**: Check autocorrelation of samples

### `steps_per_sample`
- **Default**: 1 (hardcoded)
- **What it controls**: Gibbs sweeps between samples (for multiple samples)
- **Current usage**: Only 1 sample per chain, so this doesn't affect results
- **Future use**: If collecting multiple samples, set to ~10 to reduce correlation

---

## Iterative Refinement Parameters

### `k_best` (Selection percentile)
- **Default**: 10% of batch_size (hardcoded in `server.py:146`)
- **Rationale**: Similar to Cross-Entropy Method (CEM) elite selection
- **What it controls**: How many top trajectories inform the next bias field
- **Tested alternatives**:
  - 5% (k=2 for batch=50): Too aggressive, can get stuck in local minima
  - 20% (k=10): Too conservative, slow convergence
  - **10% (k=5)**: Good balance between exploitation and exploration
- **Research reference**: CEM typically uses 5-10% (Rubinstein & Kroese, 2004)

### `bias_scale` (Bias field magnitude)
- **Default**: 4.0 (hardcoded in `server.py:152`)
- **Current formula**: `bias = (avg_schedule - 0.5) * 4.0`
- **What it does**: Maps [0,1] probability to [-2, +2] Ising field strength
- **Derivation**:
  - `avg_schedule ∈ [0, 1]` (average of best schedules)
  - Centered: `avg - 0.5 ∈ [-0.5, 0.5]`
  - Scaled: `4.0 * [-0.5, 0.5] = [-2, 2]`
- **Typical Ising field values**:
  - h = -2: Strong OFF preference
  - h = 0: Neutral
  - h = +2: Strong ON preference
- **Why 4.0?**:
  - Empirically tested: 2.0 too weak, 8.0 too strong (overfits early bias)
  - Should provide ~80% probability flip given coupling J~0.5
- **Adaptive alternative**: Could decay over iterations (like simulated annealing)

---

## Physical Parameters

### `thrust` (Newtons)
- **Default**: 10.0 N
- **Range**: 0.1 - 100 N (low-thrust regime)
- **Real missions**:
  - ARTEMIS (P1/P2): ~0.5 N (hydrazine thrusters)
  - Dawn: ~90 mN (ion drive)
  - Lunar IceCube: ~1 mN (micro-propulsion)
- **Effect**: Higher thrust → faster transfers, easier convergence

### `mass` (kg)
- **Default**: 1000.0 kg
- **What it affects**: Acceleration = thrust / mass
- **Range**: 100 - 5000 kg
- **Real missions**: Smallsats ~100-500 kg, medium ~1000-2000 kg

### `isp` (Specific impulse, seconds)
- **Default**: 300.0 s
- **Current status**: ⚠️ Not used in dynamics (TODO: add fuel consumption)
- **Typical values**:
  - Chemical: 200-450 s
  - Electric (ion): 1000-3000 s
- **Future use**: Constrain total Δv, penalize thrust-on time

### `initial_altitude` (km above Earth)
- **Default**: 400 km (LEO)
- **Range**: 200 - 36,000 km
- **What it controls**: Starting orbit energy
- **Effect on difficulty**:
  - LEO (400 km): Hardest (most Δv needed)
  - GTO (35,786 km): Easier
  - L1/L2 libration orbits: Easiest

---

## Constraint Parameters (Future)

### `eclipse_indices` (Power constraints)
- **Status**: Interface exists but not used
- **What it would do**: Set strong negative bias (h ~ -10) at time steps in Earth's shadow
- **How to compute**: Check if spacecraft-sun vector crosses Earth disk

### `perigee_indices` (Orbital mechanics heuristic)
- **Status**: Interface exists but not used
- **Heuristic**: Thrust near perigee is most Δv-efficient (Oberth effect)
- **Implementation**: Detect when distance to Earth is minimum

---

## Tuning Workflow

### Step 1: Baseline Run
```python
{
  "num_steps": 5000,
  "batch_size": 50,
  "coupling_strength": 0.5,
  "num_iterations": 50,
  "thrust": 10.0,
  "mass": 1000.0,
  "initial_altitude": 400
}
```
**Goal**: Does it reach the Moon at all? If no → increase thrust or iterations.

### Step 2: Tune Coupling
Try `coupling_strength` = [0.0, 0.3, 0.5, 1.0, 2.0]
- **Metric**: How quickly does best_cost decrease?
- **Optimal**: Usually 0.3-0.7 for this problem

### Step 3: Tune Selection Pressure
Modify `k_best` in `server.py` (requires code change):
- Try 5%, 10%, 20%
- **Metric**: Convergence rate vs final quality trade-off

### Step 4: Fine-tune Resolution
If reaching Moon consistently:
- ↑ Increase `num_steps` to 10,000
- ↓ Decrease `dt` to 0.005
- **Goal**: Smoother, more accurate trajectories

---

## Validation & Metrics

### Check Physics Correctness
- [ ] Energy conservation: Jacobi constant should vary < 1%
- [ ] Moon encounter: Final distance < 0.05 (normalized) ≈ 20,000 km
- [ ] No collisions: Min distance to Earth > 0.0166 (Earth radius)

### Check Optimizer Performance
- [ ] **Convergence**: Plot best_cost vs iteration (should decrease)
- [ ] **Diversity**: Early iterations should explore widely
- [ ] **Exploitation**: Later iterations should refine best solution

### Compare Methods
- [ ] THRML vs Random baseline
- [ ] THRML vs Heuristic (always thrust at apogee)
- [ ] With vs without coupling

---

## References

- **Cross-Entropy Method**: Rubinstein, R. Y., & Kroese, D. P. (2004). *The Cross-Entropy Method*
- **CR3BP**: Koon, W. S., et al. (2011). *Dynamical Systems, the Three-Body Problem and Space Mission Design*
- **Low-Thrust Optimization**: Taheri, E., & Abdelkhalik, O. (2016). "Fast initial trajectory design for low-thrust restricted-three-body problems"
- **THRML Library**: [Internal Extropic documentation]
