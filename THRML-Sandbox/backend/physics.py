"""
CR3BP Physics Engine - THRML-Sandbox Wrapper
=============================================

This module wraps the unified core physics engine for THRML-Sandbox,
maintaining backwards compatibility while adding new physics features.
"""

import sys
import os

# Add core to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
core_path = os.path.join(project_root, 'core')
if core_path not in sys.path:
    sys.path.insert(0, project_root)

# Import from unified core
from core import (
    # Constants
    MU, EPSILON, L_STAR_KM, L_STAR_M, T_STAR_S, V_STAR_KMS, V_STAR_MS, A_STAR_MS2,
    R_EARTH_KM, R_MOON_KM, R_EARTH_NORM, R_MOON_NORM,
    EARTH_POS, MOON_POS, G0_NORM,
    
    # Physics
    equations_of_motion_4state as equations_of_motion,
    equations_of_motion_with_mass,
    rk4_step_4state as rk4_step,
    rk4_step_with_mass,
    propagate_trajectory_4state as propagate_trajectory,
    propagate_trajectory_with_mass,
    batch_propagate_4state as batch_propagate,
    batch_propagate_with_mass,
    
    # Utilities
    get_initial_state_4d,
    get_parking_orbit_state,
    dimensionalize_trajectory,
    compute_trajectory_cost,
    batch_compute_cost,
    compute_jacobi_constant,
    validate_trajectory,
    
    # Orbital mechanics
    detect_periapsis_apoapsis,
    compute_osculating_elements,
    
    # Constraints
    check_fuel_budget,
    compute_fuel_consumed,
)

# Re-export for backwards compatibility
__all__ = [
    'MU', 'EPSILON',
    'equations_of_motion',
    'rk4_step',
    'propagate_trajectory',
    'batch_propagate',
    'dimensionalize_trajectory',
    # New exports
    'equations_of_motion_with_mass',
    'rk4_step_with_mass',
    'propagate_trajectory_with_mass',
    'batch_propagate_with_mass',
    'get_parking_orbit_state',
    'compute_trajectory_cost',
    'batch_compute_cost',
    'validate_trajectory',
    'detect_periapsis_apoapsis',
    'check_fuel_budget',
]

# Constants for backwards compatibility (re-exported from core)
L_STAR = L_STAR_M  # meters
T_STAR = T_STAR_S  # seconds
V_STAR = V_STAR_MS  # m/s
