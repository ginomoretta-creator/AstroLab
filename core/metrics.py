"""
Enhanced Metrics Module for Trajectory Analysis
================================================

This module provides comprehensive metrics for comparing classical and hybrid
quantum-classical optimization methods in cislunar trajectory optimization.

Key Metrics:
- ΔV total: Total velocity change (Tsiolkovsky equation)
- Time of flight: Mission duration in days
- Fuel consumed: Propellant mass used
- Cost breakdown: Decomposition of multi-objective cost function
- Convergence rate: Speed of optimization convergence
- Success metrics: Capture statistics

Author: ASL-Sandbox Team
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional

from .constants import (
    MU, L_STAR_KM, T_STAR_S, V_STAR_KMS, G0_NORM, G0_MS2,
    MOON_POS, EARTH_POS, LUNAR_SOI_NORM
)


# =============================================================================
# Delta-V Calculations
# =============================================================================

def calculate_delta_v(
    schedule: np.ndarray,
    thrust_magnitude: float,
    mass_initial: float,
    dt: float,
    isp: float
) -> Dict[str, float]:
    """
    Calculate total ΔV from thrust schedule using Tsiolkovsky equation.

    Two methods:
    1. Analytical: ΔV = Isp * g0 * ln(m0/mf)
    2. Numerical: ΔV = sum(a_i * dt_i) with time-varying mass

    Args:
        schedule: (N,) binary thrust schedule [0 or 1]
        thrust_magnitude: Thrust magnitude in Newtons
        mass_initial: Initial mass in kg
        dt: Time step in normalized units
        isp: Specific impulse in seconds

    Returns:
        Dictionary with:
        - delta_v_analytical: ΔV from Tsiolkovsky (m/s)
        - delta_v_numerical: ΔV from numerical integration (m/s)
        - fuel_consumed: Mass of propellant used (kg)
        - final_mass: Final spacecraft mass (kg)
    """
    # Denormalize dt to seconds
    dt_seconds = dt * T_STAR_S

    # Calculate fuel consumed
    # Mass flow rate: mdot = T / (Isp * g0)
    mass_flow_rate = thrust_magnitude / (isp * G0_MS2)  # kg/s

    # Total burn time
    total_burn_steps = np.sum(schedule)
    total_burn_time = total_burn_steps * dt_seconds

    # Fuel consumed
    fuel_consumed = mass_flow_rate * total_burn_time
    fuel_consumed = min(fuel_consumed, mass_initial * 0.9)  # Cap at 90% of initial mass

    final_mass = mass_initial - fuel_consumed

    # Analytical ΔV (Tsiolkovsky)
    if final_mass > 0:
        delta_v_analytical = isp * G0_MS2 * np.log(mass_initial / final_mass)
    else:
        delta_v_analytical = 0.0

    # Numerical ΔV (time-varying mass)
    # Integrate acceleration over time with decreasing mass
    delta_v_numerical = 0.0
    current_mass = mass_initial

    for thrust_on in schedule:
        if thrust_on > 0.5:  # Thrusting
            # Acceleration at current mass
            accel = thrust_magnitude / current_mass  # m/s²
            delta_v_numerical += accel * dt_seconds

            # Update mass
            dm = mass_flow_rate * dt_seconds
            current_mass = max(current_mass - dm, mass_initial * 0.1)  # Don't deplete below 10%

    return {
        'delta_v_analytical': float(delta_v_analytical),
        'delta_v_numerical': float(delta_v_numerical),
        'fuel_consumed': float(fuel_consumed),
        'final_mass': float(final_mass),
        'mass_fraction': float(final_mass / mass_initial)
    }


# =============================================================================
# Time of Flight
# =============================================================================

def calculate_time_of_flight(
    num_steps: int,
    dt: float,
    capture_timestep: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate mission time of flight.

    Args:
        num_steps: Total number of simulation steps
        dt: Time step in normalized units
        capture_timestep: Step at which lunar capture occurred (if any)

    Returns:
        Dictionary with:
        - total_time_days: Total simulation time (days)
        - total_time_norm: Total time in normalized units
        - capture_time_days: Time to capture (days), or None
        - capture_time_norm: Time to capture (normalized), or None
    """
    # Normalized time
    total_time_norm = num_steps * dt

    # Convert to days
    # T_STAR_S = 375200 s ≈ 4.34 days
    total_time_days = total_time_norm * T_STAR_S / 86400.0  # Convert seconds to days

    # Capture time (if applicable)
    if capture_timestep is not None:
        capture_time_norm = capture_timestep * dt
        capture_time_days = capture_time_norm * T_STAR_S / 86400.0
    else:
        capture_time_norm = None
        capture_time_days = None

    return {
        'total_time_days': float(total_time_days),
        'total_time_norm': float(total_time_norm),
        'capture_time_days': float(capture_time_days) if capture_time_days is not None else None,
        'capture_time_norm': float(capture_time_norm) if capture_time_norm is not None else None
    }


# =============================================================================
# Cost Breakdown Analysis
# =============================================================================

def calculate_cost_breakdown(
    trajectory: np.ndarray,
    schedule: np.ndarray,
    moon_pos: np.ndarray = np.array([1 - MU, 0]),
    earth_pos: np.ndarray = np.array([-MU, 0]),
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Decompose the multi-objective cost function into components.

    Matches the cost calculation in server.py:
    - Minimum distance to Moon (50%)
    - Final distance to Moon (30%)
    - Approach progress (20%)
    - Velocity cost (50%)
    - Apoapsis reward (-20%)
    - Fuel penalty
    - Capture bonus (-1000)

    Args:
        trajectory: (N, 4) state history [x, y, vx, vy]
        schedule: (N,) thrust schedule
        moon_pos: Moon position (normalized)
        earth_pos: Earth position (normalized)
        weights: Optional custom weights for cost components

    Returns:
        Dictionary with individual cost components
    """
    if weights is None:
        weights = {
            'min_distance': 0.5,
            'final_distance': 0.3,
            'approach_progress': 0.2,
            'velocity': 0.5,
            'apoapsis': 0.2,
        }

    # Extract positions and velocities
    positions = trajectory[:, :2]
    velocities = trajectory[:, 2:4]

    # 1. Distance to Moon
    dists_to_moon = np.linalg.norm(positions - moon_pos, axis=1)
    min_dist_to_moon = np.min(dists_to_moon)
    final_dist_to_moon = dists_to_moon[-1]

    # 2. Approach progress
    initial_dist = dists_to_moon[0]
    approach_progress = initial_dist - final_dist_to_moon

    # 3. Apoapsis reward
    dists_from_earth = np.linalg.norm(positions - earth_pos, axis=1)
    max_apoapsis = np.max(dists_from_earth)
    apoapsis_reward = np.clip(max_apoapsis, 0.0, 1.0) * weights['apoapsis']

    # 4. Velocity cost (near Moon)
    vel_mags = np.linalg.norm(velocities, axis=1)
    is_close = dists_to_moon < 0.05  # Within 19,000 km
    velocity_cost = np.mean(np.where(is_close, vel_mags * 2.0, 0.0))

    # 5. Fuel penalty
    thrust_fraction = np.mean(schedule)
    max_allowed = 0.6
    fuel_penalty = max(0.0, (thrust_fraction - max_allowed) * 10.0 * 384400.0) if thrust_fraction > max_allowed else 0.0

    # 6. Radial penalty (escaping system)
    radii = np.linalg.norm(positions, axis=1)
    max_radius = np.max(radii)
    radial_penalty = max(0.0, (max_radius - 1.5) * 5e3) if max_radius > 1.5 else 0.0

    # 7. Collision penalty
    collision_radius = 0.005  # ~1922 km
    collision_penalty = 1e6 if min_dist_to_moon < collision_radius else 0.0

    # 8. Lunar capture detection
    CAPTURE_VELOCITY_NORM = 0.9756  # 1 km/s
    in_soi = dists_to_moon < LUNAR_SOI_NORM
    low_velocity = vel_mags < CAPTURE_VELOCITY_NORM
    capture_condition = in_soi & low_velocity
    has_captured = np.any(capture_condition)
    capture_bonus = -1000.0 if has_captured else 0.0

    # Combined cost (matching server.py)
    distance_cost = min_dist_to_moon * weights['min_distance'] + final_dist_to_moon * weights['final_distance']
    progress_cost = -approach_progress * weights['approach_progress']

    total_cost = (
        distance_cost +
        progress_cost +
        velocity_cost * weights['velocity'] +
        -apoapsis_reward +
        capture_bonus +
        fuel_penalty +
        radial_penalty +
        collision_penalty
    )

    return {
        'total_cost': float(total_cost),
        'distance_cost': float(distance_cost),
        'min_distance_component': float(min_dist_to_moon * weights['min_distance']),
        'final_distance_component': float(final_dist_to_moon * weights['final_distance']),
        'progress_cost': float(progress_cost),
        'velocity_cost': float(velocity_cost * weights['velocity']),
        'apoapsis_reward': float(apoapsis_reward),
        'fuel_penalty': float(fuel_penalty),
        'radial_penalty': float(radial_penalty),
        'collision_penalty': float(collision_penalty),
        'capture_bonus': float(capture_bonus),
        # Additional info
        'min_distance_km': float(min_dist_to_moon * L_STAR_KM),
        'final_distance_km': float(final_dist_to_moon * L_STAR_KM),
        'max_apoapsis_km': float(max_apoapsis * L_STAR_KM),
        'thrust_fraction': float(thrust_fraction),
        'captured': bool(has_captured)
    }


# =============================================================================
# Convergence Analysis
# =============================================================================

def calculate_convergence_rate(
    iteration_history: List[Dict]
) -> Dict[str, float]:
    """
    Analyze convergence behavior from iteration history.

    Metrics:
    - Convergence speed: How fast cost decreases
    - Convergence stability: Variance in recent iterations
    - Plateaued: Whether optimization has stagnated

    Args:
        iteration_history: List of iteration dicts with 'bestCost' field

    Returns:
        Dictionary with convergence metrics
    """
    if len(iteration_history) < 2:
        return {
            'convergence_rate': 0.0,
            'cost_improvement': 0.0,
            'iterations_to_best': 0,
            'is_converged': False,
            'plateau_detected': False
        }

    costs = np.array([it['bestCost'] for it in iteration_history])

    # Convergence rate: slope of cost vs iteration (negative = improving)
    iterations = np.arange(len(costs))
    if len(costs) > 1:
        # Linear regression
        A = np.vstack([iterations, np.ones(len(iterations))]).T
        slope, intercept = np.linalg.lstsq(A, costs, rcond=None)[0]
        convergence_rate = float(slope)
    else:
        convergence_rate = 0.0

    # Cost improvement
    initial_cost = costs[0]
    final_cost = costs[-1]
    cost_improvement = float(initial_cost - final_cost)
    relative_improvement = float((initial_cost - final_cost) / (abs(initial_cost) + 1e-10))

    # Best iteration
    best_idx = int(np.argmin(costs))
    iterations_to_best = best_idx + 1

    # Convergence detection (last 5 iterations have < 1% improvement)
    if len(costs) >= 5:
        recent_costs = costs[-5:]
        recent_improvement = (recent_costs[0] - recent_costs[-1]) / (abs(recent_costs[0]) + 1e-10)
        is_converged = abs(recent_improvement) < 0.01
    else:
        is_converged = False

    # Plateau detection (last 3 iterations have < 0.1% change)
    if len(costs) >= 3:
        last_3 = costs[-3:]
        max_change = np.max(np.abs(np.diff(last_3))) / (abs(last_3[0]) + 1e-10)
        plateau_detected = max_change < 0.001
    else:
        plateau_detected = False

    return {
        'convergence_rate': convergence_rate,
        'cost_improvement': cost_improvement,
        'relative_improvement': relative_improvement,
        'iterations_to_best': iterations_to_best,
        'is_converged': is_converged,
        'plateau_detected': plateau_detected,
        'final_cost': float(final_cost),
        'best_cost': float(costs[best_idx])
    }


# =============================================================================
# Success Metrics
# =============================================================================

def calculate_success_metrics(
    trajectories: np.ndarray,
    schedules: np.ndarray,
    moon_pos: np.ndarray = np.array([1 - MU, 0])
) -> Dict[str, float]:
    """
    Calculate success statistics for a batch of trajectories.

    Success criteria:
    - Reached Lunar SOI (< 66,100 km)
    - Low velocity at approach (< 1 km/s)

    Args:
        trajectories: (B, N, 4) batch of trajectories
        schedules: (B, N) batch of thrust schedules
        moon_pos: Moon position (normalized)

    Returns:
        Dictionary with success statistics
    """
    batch_size = trajectories.shape[0]
    num_steps = trajectories.shape[1]

    # Calculate distances and velocities
    positions = trajectories[:, :, :2]  # (B, N, 2)
    velocities = trajectories[:, :, 2:4]  # (B, N, 2)

    # Distances to Moon
    dists_to_moon = np.linalg.norm(positions - moon_pos, axis=2)  # (B, N)
    vel_mags = np.linalg.norm(velocities, axis=2)  # (B, N)

    # Capture condition
    CAPTURE_VELOCITY_NORM = 0.9756  # 1 km/s in normalized units
    in_soi = dists_to_moon < LUNAR_SOI_NORM
    low_velocity = vel_mags < CAPTURE_VELOCITY_NORM
    capture_condition = in_soi & low_velocity  # (B, N)

    # Has captured (any timestep)
    has_captured = np.any(capture_condition, axis=1)  # (B,)
    success_rate = float(np.mean(has_captured))

    # Minimum distances
    min_dists = np.min(dists_to_moon, axis=1)  # (B,)
    mean_min_dist = float(np.mean(min_dists))
    best_min_dist = float(np.min(min_dists))

    # For successful captures, find capture distance
    if np.any(has_captured):
        captured_indices = np.where(has_captured)[0]
        capture_dists = []

        for idx in captured_indices:
            # Find first capture timestep
            capture_mask = capture_condition[idx]
            if np.any(capture_mask):
                first_capture_step = int(np.argmax(capture_mask))
                capture_dist = dists_to_moon[idx, first_capture_step]
                capture_dists.append(capture_dist)

        mean_capture_dist = float(np.mean(capture_dists)) if capture_dists else None
        mean_capture_dist_km = mean_capture_dist * L_STAR_KM if mean_capture_dist else None
    else:
        mean_capture_dist = None
        mean_capture_dist_km = None

    # Thrust statistics
    thrust_fractions = np.mean(schedules, axis=1)
    mean_thrust_fraction = float(np.mean(thrust_fractions))

    return {
        'success_rate': success_rate,
        'num_successful': int(np.sum(has_captured)),
        'num_total': batch_size,
        'mean_min_distance_norm': mean_min_dist,
        'mean_min_distance_km': float(mean_min_dist * L_STAR_KM),
        'best_min_distance_norm': best_min_dist,
        'best_min_distance_km': float(best_min_dist * L_STAR_KM),
        'mean_capture_distance_norm': mean_capture_dist,
        'mean_capture_distance_km': mean_capture_dist_km,
        'mean_thrust_fraction': mean_thrust_fraction
    }


# =============================================================================
# Comprehensive Run Analysis
# =============================================================================

def analyze_run(
    iteration_history: List[Dict],
    best_trajectory: np.ndarray,
    best_schedule: np.ndarray,
    params: Dict
) -> Dict:
    """
    Comprehensive analysis of a simulation run.

    Combines all metric calculations into a single summary.

    Args:
        iteration_history: List of iteration results
        best_trajectory: (N, 4) best trajectory from run
        best_schedule: (N,) best thrust schedule
        params: Simulation parameters (mass, thrust, isp, dt, num_steps)

    Returns:
        Dictionary with comprehensive metrics
    """
    # Extract parameters
    mass = params.get('mass', 500.0)
    thrust = params.get('thrust', 0.5)
    isp = params.get('isp', 3000.0)
    dt = params.get('dt', 0.0005)
    num_steps = params.get('num_steps', 120000)

    # Calculate metrics
    delta_v_metrics = calculate_delta_v(best_schedule, thrust, mass, dt, isp)
    tof_metrics = calculate_time_of_flight(num_steps, dt)
    cost_breakdown = calculate_cost_breakdown(best_trajectory, best_schedule)
    convergence_metrics = calculate_convergence_rate(iteration_history)

    # Check if captured
    if cost_breakdown.get('captured', False):
        # Find capture timestep
        positions = best_trajectory[:, :2]
        velocities = best_trajectory[:, 2:4]
        moon_pos = np.array([1 - MU, 0])

        dists = np.linalg.norm(positions - moon_pos, axis=1)
        vel_mags = np.linalg.norm(velocities, axis=1)

        CAPTURE_VELOCITY_NORM = 0.9756
        in_soi = dists < LUNAR_SOI_NORM
        low_vel = vel_mags < CAPTURE_VELOCITY_NORM
        capture_condition = in_soi & low_vel

        if np.any(capture_condition):
            capture_step = int(np.argmax(capture_condition))
            tof_metrics_capture = calculate_time_of_flight(num_steps, dt, capture_step)
            tof_metrics.update({
                'capture_time_days': tof_metrics_capture['capture_time_days'],
                'capture_time_norm': tof_metrics_capture['capture_time_norm']
            })

    # Combine all metrics
    comprehensive_metrics = {
        **delta_v_metrics,
        **tof_metrics,
        **cost_breakdown,
        **convergence_metrics,
        'num_iterations': len(iteration_history),
        'method': params.get('method', 'unknown')
    }

    return comprehensive_metrics


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'calculate_delta_v',
    'calculate_time_of_flight',
    'calculate_cost_breakdown',
    'calculate_convergence_rate',
    'calculate_success_metrics',
    'analyze_run'
]
