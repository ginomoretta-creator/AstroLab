import sys
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Literal, Generator
import json
from fastapi.responses import StreamingResponse, FileResponse
import jax
import jax.numpy as jnp
import numpy as np
import io
from scipy.io import savemat

# Ensure backend is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Add project root for core imports
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modules
from physics import batch_propagate, dimensionalize_trajectory

# Try to import core for physics-aware sampling and metrics
try:
    from core import compute_physics_bias_field, get_initial_state_4d, compute_reference_trajectory_for_bias
    from core.metrics import calculate_delta_v, calculate_time_of_flight, calculate_cost_breakdown, analyze_run
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    print("Core module not available, using basic sampling")

# Import run storage
try:
    from run_storage import RunStorage, compare_runs
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    print("Run storage not available")

# Import trajectory analysis
try:
    from trajectory_analysis import analyze_full_trajectory, export_trajectory_to_json
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False
    print("Trajectory analysis not available")

# Import quantum solver (Ising model with D-Wave Neal)
try:
    # Add QNTM-Sandbox to path
    qntm_backend = os.path.join(project_root, 'QNTM-Sandbox', 'backend')
    if qntm_backend not in sys.path:
        sys.path.insert(0, qntm_backend)

    from quantum_solver import SimulatedQuantumAnnealer, IterativeQuantumOptimizer
    QUANTUM_SOLVER_AVAILABLE = True
    print("[QUANTUM] D-Wave Neal Ising solver available")
except ImportError as e:
    QUANTUM_SOLVER_AVAILABLE = False
    print(f"[QUANTUM] Ising solver not available: {e}")
    print("[QUANTUM] Falling back to temperature-based hybrid method")

app = FastAPI(title="Cislunar Trajectory Sandbox API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize run storage
if STORAGE_AVAILABLE:
    run_storage = RunStorage(max_runs=50)
    print(f"[STORAGE] Initialized at {run_storage.storage_dir}")
else:
    run_storage = None

class SimulationRequest(BaseModel):
    # Schedule/trajectory parameters
    num_steps: int = 165000  # Long simulation for low-thrust missions (~716 days with dt=0.001, +10% margin)
    batch_size: int = 50  # Reduced for memory efficiency
    coupling_strength: float = 0.5

    # Physical Parameters
    mass: float = 400.0 # kg (SMART-1-like spacecraft)
    thrust: float = 0.07 # N (70 mN, low-thrust Hall thruster similar to SMART-1)
    isp: float = 1640.0 # s (Hall thruster Isp, SMART-1 value)
    apogee_altitude: float = 400.0 # km (Apogee altitude above Earth)
    perigee_altitude: float = 200.0 # km (Perigee altitude above Earth)

    # Method: classical uses CEM, hybrid uses quantum annealing + classical refinement
    method: Literal["classical", "hybrid"] = "classical"

    # Advanced
    dt: float = 0.001  # Balanced timestep for low-thrust (150000 * 0.001 = 150 norm time = ~650 days)
    num_iterations: int = 30  # More iterations for convergence

    # Demo mode: scales thrust up for faster visualization during development
    demo_mode: bool = False  # If True, thrust is multiplied by 50x

    # 3D Parameters (NEW)
    enable_3d: bool = False  # Enable 3D trajectories with inclination
    inclination_deg: float = 0.0  # Orbital inclination (degrees)
    raan_deg: float = 0.0  # Right Ascension of Ascending Node (degrees)
    thrust_mode: Literal["velocity_aligned", "orbital_plane"] = "orbital_plane"  # Thrust direction mode

# Constants for Normalization
L_STAR = 384400.0 * 1000 # meters (Earth-Moon Distance)
T_STAR = 375200.0 # seconds (approx 4.34 days)
M_STAR = 1000.0 # kg (Reference mass, arbitrary)
MU = 0.01215

@app.get("/")
def read_root():
    hybrid_description = "Ising Hamiltonian with D-Wave Neal simulated annealing (2D/3D)"
    if not QUANTUM_SOLVER_AVAILABLE:
        hybrid_description += " (fallback: temperature-based sampling)"

    return {
        "status": "online",
        "system": "ASL-Sandbox Backend",
        "core_available": CORE_AVAILABLE,
        "quantum_solver_available": QUANTUM_SOLVER_AVAILABLE,
        "methods": {
            "classical": "Cross-Entropy Method (CEM) - Pure classical optimization",
            "hybrid": hybrid_description
        },
        "quantum_solver": {
            "available": QUANTUM_SOLVER_AVAILABLE,
            "model": "1D Ising chain with ferromagnetic coupling",
            "sampler": "D-Wave Neal SimulatedAnnealingSampler",
            "energy": "E = -J * Σ s_i*s_{i+1} - Σ h_i*s_i",
            "dimensions": "2D and 3D supported"
        } if QUANTUM_SOLVER_AVAILABLE else None
    }

def get_initial_state(apogee_km, perigee_km, enable_3d=False, inclination_deg=0.0, raan_deg=0.0):
    """
    Calculate initial state for elliptical orbit at perigee.

    Args:
        apogee_km: Apogee altitude above Earth (km)
        perigee_km: Perigee altitude above Earth (km)
        enable_3d: If True, return 7-state 3D vector; otherwise 4-state 2D
        inclination_deg: Orbital inclination (degrees, only used if enable_3d=True)
        raan_deg: Right Ascension of Ascending Node (degrees, only used if enable_3d=True)

    Returns:
        [x, y, vx, vy] for 2D or [x, y, z, vx, vy, vz] for 3D (mass excluded) in normalized rotating frame
    """
    if enable_3d:
        # Calculate elliptical orbit in 3D
        # First, calculate in the orbital plane, then rotate to desired inclination/RAAN
        R_EARTH_KM = 6378.0
        L_MOON_KM = 384400.0

        # Convert altitudes to radii
        r_apogee = R_EARTH_KM + apogee_km
        r_perigee = R_EARTH_KM + perigee_km

        # Semi-major axis and eccentricity
        a = (r_apogee + r_perigee) / 2.0
        e = (r_apogee - r_perigee) / (r_apogee + r_perigee)

        print(f"[3D ORBIT DEBUG] Apogee: {apogee_km} km, Perigee: {perigee_km} km", flush=True)
        print(f"[3D ORBIT DEBUG] Semi-major axis: {a:.1f} km, Eccentricity: {e:.4f}", flush=True)
        if e < 0.001:
            print(f"[3D ORBIT WARNING] Orbit is essentially CIRCULAR (e={e:.6f})", flush=True)
        elif e < 0.05:
            print(f"[3D ORBIT INFO] Orbit is nearly circular (e={e:.6f}). Increase apogee-perigee difference for more elliptical orbit.", flush=True)

        # Start at perigee in orbital plane (x-axis)
        r_norm = r_perigee / L_MOON_KM
        a_norm = a / L_MOON_KM

        # Velocity at perigee using vis-viva equation
        v_perigee = np.sqrt((1 - MU) * (2.0 / r_norm - 1.0 / a_norm))

        print(f"[3D ORBIT DEBUG] Perigee velocity: {v_perigee:.6f} (normalized), {v_perigee * 1.025:.3f} km/s", flush=True)

        # Initial state in orbital plane (perigee on x-axis, velocity in y direction)
        # In inertial frame centered on Earth
        pos_orbital = np.array([r_norm, 0.0, 0.0])
        vel_orbital = np.array([0.0, v_perigee, 0.0])

        # Rotation matrices for inclination and RAAN
        inc_rad = np.deg2rad(inclination_deg)
        raan_rad = np.deg2rad(raan_deg)

        # Rotation matrix: R_z(RAAN) * R_x(inclination)
        cos_i, sin_i = np.cos(inc_rad), np.sin(inc_rad)
        cos_raan, sin_raan = np.cos(raan_rad), np.sin(raan_rad)

        R_x = np.array([
            [1, 0, 0],
            [0, cos_i, -sin_i],
            [0, sin_i, cos_i]
        ])

        R_z = np.array([
            [cos_raan, -sin_raan, 0],
            [sin_raan, cos_raan, 0],
            [0, 0, 1]
        ])

        R = R_z @ R_x

        # Apply rotation to get position and velocity in inertial frame
        pos_inertial = R @ pos_orbital
        vel_inertial = R @ vel_orbital

        # Transform to rotating frame (CR3BP)
        # Position: shift to rotating frame origin (Earth at -MU, 0, 0)
        x = -MU + pos_inertial[0]
        y = pos_inertial[1]
        z = pos_inertial[2]

        # Velocity: v_rotating = v_inertial - omega × r
        # omega = [0, 0, 1] (rotation about z-axis with omega = 1 in normalized units)
        # omega × r = [-y, x, 0]
        vx = vel_inertial[0] + y
        vy = vel_inertial[1] - x
        vz = vel_inertial[2]

        print(f"[3D ORBIT DEBUG] Initial state: pos=[{x:.4f}, {y:.4f}, {z:.4f}], vel=[{vx:.4f}, {vy:.4f}, {vz:.4f}]", flush=True)

        return [x, y, z, vx, vy, vz]

    # 2D Mode (existing logic)
    R_EARTH_KM = 6378.0
    L_MOON_KM = 384400.0

    # Convert altitudes to radii
    r_apogee = R_EARTH_KM + apogee_km
    r_perigee = R_EARTH_KM + perigee_km

    # Semi-major axis and eccentricity
    a = (r_apogee + r_perigee) / 2.0
    e = (r_apogee - r_perigee) / (r_apogee + r_perigee)

    print(f"[ORBIT DEBUG] Apogee: {apogee_km} km, Perigee: {perigee_km} km", flush=True)
    print(f"[ORBIT DEBUG] Semi-major axis: {a:.1f} km, Eccentricity: {e:.4f}", flush=True)
    if e < 0.001:
        print(f"[ORBIT WARNING] Orbit is essentially CIRCULAR (e={e:.6f})", flush=True)
    elif e < 0.05:
        print(f"[ORBIT INFO] Orbit is nearly circular (e={e:.6f}). Increase apogee-perigee difference for more elliptical orbit.", flush=True)

    # Start at perigee (closest approach)
    r_norm = r_perigee / L_MOON_KM

    # Position in rotating frame (x-axis, right of Earth at origin)
    x = -MU + r_norm
    y = 0.0

    # Velocity at perigee using vis-viva equation
    # v = sqrt(GM * (2/r - 1/a))
    # In normalized units: GM_Earth = 1 - MU, and a must be normalized too
    a_norm = a / L_MOON_KM
    v_perigee = np.sqrt((1 - MU) * (2.0 / r_norm - 1.0 / a_norm))

    print(f"[ORBIT DEBUG] Perigee velocity: {v_perigee:.6f} (normalized), {v_perigee * 1.025:.3f} km/s", flush=True)

    # Rotating frame correction: v_rotating = v_inertial - omega × r
    # omega = 1 in normalized units
    vx = 0.0
    vy = v_perigee - x

    return [x, y, vx, vy]

@app.post("/simulate")
def run_simulation(req: SimulationRequest):
    def simulation_generator() -> Generator[str, None, None]:
        try:
            key = jax.random.PRNGKey(int(np.random.randint(0, 100000)))
            
            # 1. Calculate Normalized Thrust (acceleration for 2D, force for 3D)
            accel_metric = req.thrust / req.mass
            accel_norm = accel_metric * (T_STAR**2 / L_STAR)

            # For 3D mode, also calculate normalized force and ISP
            if req.enable_3d:
                # Force normalization: F_star = M_star * L_star / T_star^2
                F_STAR = M_STAR * L_STAR / (T_STAR**2)
                thrust_norm = req.thrust / F_STAR
                # ISP normalization: isp_norm = isp * g0 * T_star / L_star
                isp_norm = req.isp * 9.80665 * T_STAR / L_STAR

            # Demo mode: scale up thrust by 50x for faster development iteration
            if req.demo_mode:
                accel_norm *= 50.0
                if req.enable_3d:
                    thrust_norm *= 50.0

            accel_norm = float(np.clip(accel_norm, 0.0, 0.5))  # Allow higher thrust in demo mode
            
            # 2. Determine Initial State
            # Support both 2D and 3D modes
            mode_str = f"3D (i={req.inclination_deg}deg, RAAN={req.raan_deg}deg)" if req.enable_3d else "2D"
            print(f"[DEBUG] Computing initial state ({mode_str}) for perigee={req.perigee_altitude} km", flush=True)
            init_state = get_initial_state(
                req.apogee_altitude,
                req.perigee_altitude,
                enable_3d=req.enable_3d,
                inclination_deg=req.inclination_deg,
                raan_deg=req.raan_deg
            )
            init_state_arr = jnp.array(init_state)
            print(f"[DEBUG] Initial state ({init_state_arr.shape[0]} components): {init_state_arr}", flush=True)

            if req.enable_3d:
                pos = init_state_arr[:3]
                vel = init_state_arr[3:6]
                print(f"[3D DEBUG] Position: {pos}, |r|={jnp.linalg.norm(pos):.6f}", flush=True)
                print(f"[3D DEBUG] Velocity: {vel}, |v|={jnp.linalg.norm(vel):.6f}", flush=True)
                print(f"[3D DEBUG] Thrust: {thrust_norm:.6e}, ISP: {isp_norm:.6f}", flush=True)
            
            # 3. Initialize physics-aware bias if available (only for 2D mode)
            # Note: Physics-aware bias is designed for 4-state 2D dynamics
            if CORE_AVAILABLE and not req.enable_3d:
                # Compute reference trajectory first!
                ref_traj = compute_reference_trajectory_for_bias(
                    req.num_steps,
                    req.dt,
                    accel_norm,
                    init_state_arr
                )

                initial_bias = compute_physics_bias_field(
                    req.num_steps,
                    ref_traj,  # Pass the reference!
                    0.4
                )
                current_bias = initial_bias
            else:
                current_bias = None
                if req.enable_3d:
                    print("[3D MODE] Physics-aware bias disabled (designed for 2D only)", flush=True)
            
            # 4. Moon position for cost calculation
            moon_pos = jnp.array([1 - MU, 0, 0]) if req.enable_3d else jnp.array([1 - MU, 0])

            # 5. Lunar Capture Criteria (SUCCESS CONDITION)
            # SOI radius: 66,100 km / 384,400 km = 0.172 normalized
            LUNAR_SOI_NORM = 0.172
            # Velocity limit: EXACTLY 1 km/s
            # V_STAR = L_STAR / T_STAR = 384,400 km / 375,068 s ≈ 1.025 km/s
            # So 1 km/s = 1.0 / 1.025 = 0.9756 normalized units
            CAPTURE_VELOCITY_NORM = 0.9756  # Exactly 1 km/s

            success_threshold = LUNAR_SOI_NORM  # Changed from 0.2 to SOI
            fuel_min = 0.05
            fuel_max = 0.75
            # Clamp timestep for stability (normalized units) - allow larger dt for long simulations
            dt = float(np.clip(req.dt, 0.0005, 0.05))

            print(f"[DEBUG] Starting iteration loop: {req.num_iterations} iterations", flush=True)
            for i in range(req.num_iterations):
                print(f"[DEBUG] ===== ITERATION {i+1}/{req.num_iterations} START =====", flush=True)
                # New randomness each iteration to avoid repeating schedules
                key, sample_key = jax.random.split(key)
                sample_key, sched_key = jax.random.split(sample_key)

                # Generate Schedules based on method
                if req.method == "classical":
                    # Classical Method: Cross-Entropy Method (CEM)
                    # Pure classical iterative optimization
                    # Uses physics-aware bias that gets refined each iteration
                    if current_bias is not None:
                        probs = jax.nn.sigmoid(current_bias)
                    else:
                        # Start with unbiased 40% thrust duty cycle
                        probs = jnp.ones(req.num_steps) * 0.4
                    schedules = jax.random.bernoulli(sample_key, probs, shape=(req.batch_size, req.num_steps)).astype(jnp.float32)

                elif req.method == "hybrid":
                    # Hybrid Quantum-Classical Method:
                    # Uses D-Wave Neal simulated annealing on Ising Hamiltonian
                    # E = -J * Σ s_i * s_{i+1} - Σ h_i * s_i

                    if QUANTUM_SOLVER_AVAILABLE:
                        # TRUE ISING SOLVER (D-Wave Neal) - NOW SUPPORTS 3D!
                        # Create annealer with temperature schedule based on iteration
                        beta_min = 0.1  # High temperature (exploration)
                        beta_max = 10.0 * (1 + i / req.num_iterations)  # Increasing inverse temperature

                        # TEMPORARY: Reduced sweeps for faster debugging (was 1000)
                        annealer = SimulatedQuantumAnnealer(
                            num_reads=req.batch_size,
                            num_sweeps=200,  # Reduced for speed during debug
                            beta_range=(beta_min, beta_max)
                        )
                        print(f"[DEBUG] Created Ising annealer with {req.batch_size} reads, 200 sweeps", flush=True)

                        # Convert current_bias to physics bias field
                        if current_bias is not None:
                            # Normalize bias to reasonable range for Ising model
                            physics_bias_field = np.array(current_bias * 2.0)  # Scale for Ising
                        else:
                            physics_bias_field = np.zeros(req.num_steps)

                        # Sample from Ising model
                        print(f"[DEBUG] Calling Ising sampler (num_steps={req.num_steps})...", flush=True)
                        result = annealer.generate_thrust_schedules(
                            num_steps=req.num_steps,
                            batch_size=req.batch_size,
                            coupling_strength=req.coupling_strength,
                            physics_bias_field=physics_bias_field
                        )

                        schedules = result['schedules']
                        ising_energies = result['energies']

                        mode_str = "3D" if req.enable_3d else "2D"
                        print(f"[HYBRID-ISING-{mode_str}] Iteration {i+1}: Beta={beta_max:.2f}, Mean energy={float(jnp.mean(ising_energies)):.2f}, Mean thrust={float(jnp.mean(schedules)):.3f}", flush=True)
                        print(f"[DEBUG] Ising sampling completed, got {len(schedules)} schedules", flush=True)

                    else:
                        # FALLBACK: Temperature-based sampling (only if solver unavailable)
                        print(f"[HYBRID-FALLBACK] Using temperature-based method (Ising solver not available)")

                        temperature = 1.0 - (i / req.num_iterations)

                        if current_bias is not None:
                            temp_factor = max(temperature, 0.1)
                            biased_probs = jax.nn.sigmoid(current_bias / temp_factor)
                        else:
                            biased_probs = jnp.ones(req.num_steps) * 0.4

                        noise_key, sample_key = jax.random.split(sample_key)
                        noise = (jax.random.uniform(noise_key, shape=(req.num_steps,)) - 0.5) * temperature * 0.3
                        final_probs = jnp.clip(biased_probs + noise, 0.0, 1.0)

                        schedules = jax.random.bernoulli(sample_key, final_probs, shape=(req.batch_size, req.num_steps)).astype(jnp.float32)

                        print(f"[HYBRID-TEMP] Iteration {i+1}: Temperature={temperature:.3f}, Mean prob={float(jnp.mean(final_probs)):.3f}")

                # Enforce reasonable fuel fractions by replacing outliers
                thrust_fractions = jnp.mean(schedules, axis=1)
                valid_mask = (thrust_fractions >= fuel_min) & (thrust_fractions <= fuel_max)
                replacement_schedules = jax.random.bernoulli(sched_key, p=0.4, shape=schedules.shape).astype(jnp.float32)
                schedules = jnp.where(valid_mask[:, None], schedules, replacement_schedules)
                thrust_fractions = jnp.mean(schedules, axis=1)
                    
                # Propagate Physics (2D or 3D based on enable_3d flag)
                if req.enable_3d:
                    # 3D Mode: Use batch_propagate_with_mass_3d from core
                    try:
                        from core.physics_core import batch_propagate_with_mass_3d
                        # init_state_arr is [x, y, z, vx, vy, vz] (6 elements)
                        # Add mass component to get [x, y, z, vx, vy, vz, m] (7 elements)
                        init_state_with_mass = jnp.append(init_state_arr, 1.0)
                        # Use pre-calculated normalized thrust and ISP
                        print(f"[DEBUG] Starting 3D propagation with {len(schedules)} trajectories...", flush=True)
                        trajectories = batch_propagate_with_mass_3d(
                            init_state_with_mass,
                            schedules,
                            thrust_norm,
                            isp_norm,
                            dt,
                            req.num_steps,
                            req.thrust_mode
                        )
                        print(f"[DEBUG] 3D propagation completed!", flush=True)
                        print(f"[3D DEBUG] Initial state shape: {init_state_with_mass.shape}, Trajectories shape: {trajectories.shape}", flush=True)
                    except ImportError:
                        print("[ERROR] 3D mode requires core module with batch_propagate_with_mass_3d", flush=True)
                        raise
                else:
                    # 2D Mode: Use existing batch_propagate (4-state)
                    thrust_schedules_mag = schedules * accel_norm
                    trajectories = batch_propagate(init_state_arr, thrust_schedules_mag, dt, req.num_steps)

                # === IMPROVED MOON-SEEKING COST FUNCTION ===

                # Extract position and velocity indices based on dimension
                pos_slice = slice(0, 3) if req.enable_3d else slice(0, 2)
                vel_slice = slice(3, 6) if req.enable_3d else slice(2, 4)

                # 1. Final distance to Moon
                final_positions = trajectories[:, -1, pos_slice]
                final_dist = jnp.linalg.norm(final_positions - moon_pos, axis=1)
                radii = jnp.linalg.norm(final_positions, axis=1)

                # 2. MINIMUM distance to Moon during entire trajectory (key improvement!)
                all_dists_to_moon = jnp.linalg.norm(trajectories[:, :, pos_slice] - moon_pos, axis=2)
                min_dist_to_moon = jnp.min(all_dists_to_moon, axis=1)

                # 3. Approach progress: are we getting closer over time?
                initial_dist = jnp.linalg.norm(trajectories[:, 0, pos_slice] - moon_pos, axis=1)
                approach_progress = initial_dist - final_dist  # positive = getting closer

                # 4. Apoapsis reward (keep this for orbit-raising behavior)
                earth_pos = jnp.array([-MU, 0, 0]) if req.enable_3d else jnp.array([-MU, 0])
                all_dists_from_earth = jnp.linalg.norm(trajectories[:, :, pos_slice] - earth_pos, axis=2)
                max_apoapsis = jnp.max(all_dists_from_earth, axis=1)
                apoapsis_reward = jnp.clip(max_apoapsis, 0.0, 1.0) * 0.2

                # 5. Velocity Guidance (Prevent flybys)
                # Calculate velocity relative to Moon frame (already in rotating frame)
                velocities = trajectories[:, :, vel_slice]
                vel_mags = jnp.linalg.norm(velocities, axis=2)
                
                # Identify where we are close to Moon (< 0.05 normalized ~ 19,000 km)
                # If close, penalize high velocity to encourage capture
                is_close = all_dists_to_moon < 0.05
                # Only penalize if close AND fast
                velocity_cost = jnp.mean(jnp.where(is_close, vel_mags * 2.0, 0.0), axis=1)

                # Fuel Penalty (Soft Constraint)
                max_allowed = 0.6
                fuel_penalty = jnp.where(
                    thrust_fractions > max_allowed, 
                    10.0 * (thrust_fractions - max_allowed) * 384400.0,
                    0.0
                )
                budget_violation_penalty = jnp.where(
                    (thrust_fractions < fuel_min) | (thrust_fractions > fuel_max),
                    1e6,
                    0.0
                )
                # Radial penalty for escaping system
                radial_penalty = jnp.where(radii > 1.5, (radii - 1.5) * 5e3, 0.0)

                # Minimum distance to Moon (Collision Check)
                # R_MOON_NORM = 1737.4 / 384400.0 ~= 0.0045
                # Using 0.005 (~1922 km) as a safe collision buffer
                collision_radius = 0.005
                collision_penalty = jnp.where(min_dist_to_moon < collision_radius, 1e6, 0.0)

                # === LUNAR CAPTURE DETECTION ===
                # A trajectory is "captured" if at ANY point it reaches:
                # 1. Within Lunar SOI (< 0.172 normalized = 66,100 km)
                # 2. With velocity relative to Moon < 1 km/s (~1.0 normalized)

                # Check capture condition: inside SOI AND low velocity
                in_soi = all_dists_to_moon < LUNAR_SOI_NORM
                low_velocity = vel_mags < CAPTURE_VELOCITY_NORM
                capture_condition = in_soi & low_velocity

                # A trajectory is successful if capture occurs at ANY timestep
                has_captured = jnp.any(capture_condition, axis=1)

                # Huge bonus for successful capture
                capture_bonus = jnp.where(has_captured, -1000.0, 0.0)

                # === COMBINED COST ===
                # - 50% Min distance (approach)
                # - 30% Final distance
                # - 20% Approach progress
                # - Velocity control (prevent flybys)
                # - HUGE BONUS: Lunar capture (-1000 for successful SOI + low velocity)
                # - Bonus: Apoapsis reward
                total_cost = (
                    min_dist_to_moon * 0.5 +
                    final_dist * 0.3 +
                    - approach_progress * 0.2 +
                    velocity_cost * 0.5 +
                    - apoapsis_reward +
                    capture_bonus +  # NEW: Massive reward for lunar capture
                    fuel_penalty + budget_violation_penalty + radial_penalty + collision_penalty
                )

                # Calculate success rate from capture detection
                success_rate = float(jnp.mean(has_captured))

                # For captured trajectories, find the first capture point
                capture_timesteps = jnp.argmax(capture_condition, axis=1)
                capture_distance = jnp.where(
                    has_captured,
                    all_dists_to_moon[jnp.arange(req.batch_size), capture_timesteps],
                    min_dist_to_moon
                )
                
                # Select Best (Top 10%) based on Total Cost
                k_best = max(1, int(req.batch_size * 0.1))
                best_indices = jnp.argsort(total_cost)[:k_best]
                best_schedules = schedules[best_indices]
                
                # Update Bias for next iteration (Cross-Entropy Method style)
                avg_schedule = jnp.mean(best_schedules, axis=0)
                learning_rate = 0.3
                if current_bias is not None:
                    current_bias = current_bias * (1 - learning_rate) + (avg_schedule - 0.5) * 4.0 * learning_rate
                else:
                    current_bias = (avg_schedule - 0.5) * 4.0
                
                # Update Key
                # Prepare response data
                traj_np = np.array(trajectories)
                min_dists_np = np.array(min_dist_to_moon)  # Use min distance to Moon for sorting
                schedules_np = np.array(schedules)
                total_cost_np = np.array(total_cost)

                # True best based on total cost (respects penalties)
                best_idx = int(np.argmin(total_cost_np))
                best_schedule = schedules_np[best_idx]
                best_thrust_fraction = float(np.mean(best_schedule))
                best_cost = float(total_cost_np[best_idx])
                best_distance = float(np.linalg.norm(traj_np[best_idx, -1, :2] - np.array([1 - MU, 0])))

                # Check if best trajectory achieved capture
                has_captured_np = np.array(has_captured)
                best_captured = bool(has_captured_np[best_idx])
                capture_distance_np = np.array(capture_distance)
                best_capture_dist = float(capture_distance_np[best_idx]) if best_captured else None

                # Get capture timestep for best trajectory
                capture_timesteps_np = np.array(capture_timesteps)
                best_capture_timestep = int(capture_timesteps_np[best_idx]) if best_captured else None

                # Get indices to send: Top 5 by min distance + Random 5
                sorted_indices = np.argsort(min_dists_np)
                top_indices = sorted_indices[:5]
                random_indices = np.random.choice(req.batch_size, 5, replace=False)
                display_indices = np.unique(np.concatenate([top_indices, random_indices]))

                chunk_trajectories = traj_np[display_indices]
                best_trajectory = traj_np[best_idx]

                # Calculate enhanced metrics for this iteration
                enhanced_metrics = {}
                orbital_snapshots = {}

                if CORE_AVAILABLE:
                    try:
                        # Delta-V metrics
                        dv_metrics = calculate_delta_v(
                            best_schedule,
                            req.thrust,
                            req.mass,
                            dt,
                            req.isp
                        )

                        # Time of flight metrics
                        tof_metrics = calculate_time_of_flight(req.num_steps, dt)

                        # Cost breakdown
                        cost_breakdown = calculate_cost_breakdown(
                            best_trajectory,
                            best_schedule
                        )

                        enhanced_metrics = {
                            'deltaV': dv_metrics.get('delta_v_analytical', 0.0),
                            'deltaV_numerical': dv_metrics.get('delta_v_numerical', 0.0),
                            'fuelConsumed': dv_metrics.get('fuel_consumed', 0.0),
                            'timeOfFlight': tof_metrics.get('total_time_days', 0.0),
                            'costBreakdown': {
                                'distance': cost_breakdown.get('distance_cost', 0.0),
                                'velocity': cost_breakdown.get('velocity_cost', 0.0),
                                'fuel': cost_breakdown.get('fuel_penalty', 0.0),
                                'capture_bonus': cost_breakdown.get('capture_bonus', 0.0)
                            }
                        }
                    except Exception as e:
                        print(f"[METRICS WARNING] Failed to calculate enhanced metrics: {e}")

                # Add orbital parameter snapshots at key points
                if ANALYSIS_AVAILABLE:
                    try:
                        from trajectory_analysis import compute_orbital_elements_at_timestep

                        # Sample orbital parameters at key timesteps
                        # Start, 25%, 50%, 75%, End
                        sample_indices = [
                            0,
                            len(best_trajectory) // 4,
                            len(best_trajectory) // 2,
                            3 * len(best_trajectory) // 4,
                            len(best_trajectory) - 1
                        ]

                        snapshots = {}
                        for idx in sample_indices:
                            state = best_trajectory[idx]
                            elements = compute_orbital_elements_at_timestep(state)

                            # Calculate time in days
                            time_days = idx * dt * T_STAR / 86400.0

                            # Store key parameters
                            progress = idx / (len(best_trajectory) - 1)
                            snapshots[f"{int(progress*100)}%"] = {
                                'timestep': int(idx),
                                'time_days': float(time_days),
                                'altitude_km': elements['altitude_km'],
                                'dist_moon_km': elements['dist_moon_km'],
                                'velocity_kms': elements['velocity_mag_kms'],
                                'eccentricity': elements['eccentricity'],
                                'perigee_km': elements['perigee_altitude_km'],
                                'apogee_km': elements['apogee_altitude_km'],
                                'orbit_type': elements['orbit_type'],
                            }

                        orbital_snapshots = snapshots

                    except Exception as e:
                        print(f"[ORBITAL SNAPSHOTS WARNING] Failed to compute: {e}")
                        import traceback
                        print(traceback.format_exc())

                chunk_data = {
                    "iteration": i + 1,
                    "total_iterations": req.num_iterations,
                    "dimension": 3 if req.enable_3d else 2,  # NEW: Dimension field
                    "trajectories": chunk_trajectories.tolist(),
                    "best_cost": best_cost,
                    "mean_cost": float(np.mean(total_cost_np)),
                    "best_trajectory": best_trajectory.tolist(),
                    "success_rate": success_rate,
                    "method": req.method,
                    "best_schedule": best_schedule.tolist(),
                    "best_thrust_fraction": best_thrust_fraction,
                    "best_distance": best_distance,
                    "radial_penalty": float(np.mean(radial_penalty)),
                    "captured": best_captured,  # Lunar capture flag
                    "capture_distance": best_capture_dist,  # Distance at capture
                    "capture_timestep": best_capture_timestep,  # Timestep when capture occurred
                    "metrics": enhanced_metrics,  # Enhanced metrics
                    "orbital_snapshots": orbital_snapshots  # NEW: Orbital parameters at key points
                }

                # Enhanced logging with capture status
                capture_msg = f" >> CAPTURED at {best_capture_dist*384.4:.0f}km" if best_captured else ""
                print(f"[{req.method.upper()}] Iteration {i+1}/{req.num_iterations}: Best dist={best_distance:.4f}, Success={success_rate:.0%}{capture_msg}", flush=True)

                print(f"[DEBUG] Yielding iteration {i+1} data to frontend...", flush=True)
                yield json.dumps(chunk_data) + "\n"
                print(f"[DEBUG] ===== ITERATION {i+1} COMPLETE =====", flush=True)
                
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"[ERROR] Simulation failed:", flush=True)
            print(error_msg, flush=True)
            yield json.dumps({
                "error": str(e),
                "traceback": error_msg
            }) + "\n"

    return StreamingResponse(simulation_generator(), media_type="application/x-ndjson")

@app.post("/benchmark")
def run_benchmark(methods: List[str], num_samples: int = 100, num_steps: int = 200):
    """Run comparative benchmark across methods."""
    results = {}

    for method in methods:
        if method not in ["classical", "hybrid"]:
            continue

        # Create request
        req = SimulationRequest(
            method=method,
            num_steps=num_steps,
            batch_size=num_samples,
            num_iterations=1,
        )

        # Run single iteration for benchmark
        key = jax.random.PRNGKey(42)
        accel_metric = req.thrust / req.mass
        accel_norm = accel_metric * (T_STAR**2 / L_STAR)
        init_state_arr = jnp.array(get_initial_state(req.apogee_altitude, req.perigee_altitude))

        # Both methods use same sampling for benchmark (single iteration)
        probs = jnp.ones(num_steps) * 0.4
        schedules = jax.random.bernoulli(key, probs, shape=(num_samples, num_steps)).astype(jnp.float32)

        thrust_schedules_mag = schedules * accel_norm
        trajectories = batch_propagate(init_state_arr, thrust_schedules_mag, req.dt, num_steps)

        moon_pos = jnp.array([1 - MU, 0])
        final_positions = trajectories[:, -1, :2]
        dists = jnp.linalg.norm(final_positions - moon_pos, axis=1)

        results[method] = {
            "mean_distance": float(jnp.mean(dists)),
            "min_distance": float(jnp.min(dists)),
            "std_distance": float(jnp.std(dists)),
            "success_rate": float(jnp.mean(dists < 0.13)),
        }

    return results


# =============================================================================
# Run Management Endpoints
# =============================================================================

class SaveRunRequest(BaseModel):
    """Request to save a simulation run."""
    method: str
    params: dict
    iterations: List[dict]
    finalMetrics: dict
    best_trajectory: List[List[float]]
    best_schedule: List[float]

@app.post("/runs/save")
def save_run(req: SaveRunRequest):
    """Save a completed simulation run."""
    if not STORAGE_AVAILABLE or run_storage is None:
        raise HTTPException(status_code=503, detail="Storage not available")

    run_data = {
        'method': req.method,
        'params': req.params,
        'iterations': req.iterations,
        'finalMetrics': req.finalMetrics,
        'best_trajectory': req.best_trajectory,
        'best_schedule': req.best_schedule
    }

    run_id = run_storage.save_run(run_data)

    return {
        'success': True,
        'run_id': run_id,
        'message': f'Run saved successfully with ID {run_id}'
    }

@app.get("/runs/list")
def list_runs(
    method: Optional[str] = None,
    limit: Optional[int] = 50,
    sort_by: str = 'timestamp'
):
    """List all stored runs with metadata."""
    if not STORAGE_AVAILABLE or run_storage is None:
        raise HTTPException(status_code=503, detail="Storage not available")

    runs = run_storage.list_runs(method=method, limit=limit, sort_by=sort_by)

    return {
        'runs': runs,
        'count': len(runs)
    }

@app.get("/runs/{run_id}")
def get_run(run_id: str):
    """Retrieve full data for a specific run."""
    if not STORAGE_AVAILABLE or run_storage is None:
        raise HTTPException(status_code=503, detail="Storage not available")

    run_data = run_storage.load_run(run_id)

    if run_data is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return run_data

@app.delete("/runs/{run_id}")
def delete_run(run_id: str):
    """Delete a specific run."""
    if not STORAGE_AVAILABLE or run_storage is None:
        raise HTTPException(status_code=503, detail="Storage not available")

    success = run_storage.delete_run(run_id)

    if not success:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return {
        'success': True,
        'message': f'Run {run_id} deleted successfully'
    }

@app.get("/runs/storage/stats")
def get_storage_stats():
    """Get storage statistics."""
    if not STORAGE_AVAILABLE or run_storage is None:
        raise HTTPException(status_code=503, detail="Storage not available")

    return run_storage.get_storage_stats()


# =============================================================================
# Comparison Endpoints
# =============================================================================

class CompareRunsRequest(BaseModel):
    """Request to compare multiple runs."""
    run_ids: List[str]

@app.post("/runs/compare")
def compare_multiple_runs(req: CompareRunsRequest):
    """Generate comparison statistics for multiple runs."""
    if not STORAGE_AVAILABLE or run_storage is None:
        raise HTTPException(status_code=503, detail="Storage not available")

    comparison = compare_runs(req.run_ids, run_storage)

    if 'error' in comparison:
        raise HTTPException(status_code=400, detail=comparison['error'])

    return comparison


# =============================================================================
# Export Endpoints
# =============================================================================

@app.post("/export/csv")
def export_csv(req: CompareRunsRequest):
    """Export runs as CSV table (metrics only)."""
    if not STORAGE_AVAILABLE or run_storage is None:
        raise HTTPException(status_code=503, detail="Storage not available")

    # Load all requested runs
    runs_data = []
    for run_id in req.run_ids:
        run_data = run_storage.load_run(run_id)
        if run_data:
            runs_data.append(run_data)

    if not runs_data:
        raise HTTPException(status_code=404, detail="No runs found")

    # Create CSV content
    csv_lines = []

    # Header
    headers = [
        'run_id', 'timestamp', 'method',
        'delta_v_ms', 'time_of_flight_days', 'fuel_consumed_kg',
        'final_cost', 'captured', 'num_iterations',
        'convergence_rate', 'cost_improvement'
    ]
    csv_lines.append(','.join(headers))

    # Data rows
    for run in runs_data:
        metrics = run.get('finalMetrics', {})

        row = [
            run.get('run_id', ''),
            run.get('timestamp', ''),
            run.get('method', ''),
            str(metrics.get('delta_v_analytical', '')),
            str(metrics.get('total_time_days', '')),
            str(metrics.get('fuel_consumed', '')),
            str(metrics.get('final_cost', '')),
            str(metrics.get('captured', False)),
            str(len(run.get('iterations', []))),
            str(metrics.get('convergence_rate', '')),
            str(metrics.get('cost_improvement', ''))
        ]
        csv_lines.append(','.join(row))

    csv_content = '\n'.join(csv_lines)

    # Return as downloadable file
    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=runs_export.csv"}
    )

@app.post("/export/json")
def export_json(req: CompareRunsRequest):
    """Export runs as JSON (full data)."""
    if not STORAGE_AVAILABLE or run_storage is None:
        raise HTTPException(status_code=503, detail="Storage not available")

    # Load all requested runs
    runs_data = []
    for run_id in req.run_ids:
        run_data = run_storage.load_run(run_id)
        if run_data:
            runs_data.append(run_data)

    if not runs_data:
        raise HTTPException(status_code=404, detail="No runs found")

    # Return as JSON
    json_content = json.dumps(runs_data, indent=2)

    return StreamingResponse(
        io.StringIO(json_content),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=runs_export.json"}
    )

@app.post("/export/matlab")
def export_matlab(req: CompareRunsRequest):
    """Export runs as MATLAB .mat format."""
    if not STORAGE_AVAILABLE or run_storage is None:
        raise HTTPException(status_code=503, detail="Storage not available")

    # Load all requested runs
    runs_data = []
    for run_id in req.run_ids:
        run_data = run_storage.load_run(run_id)
        if run_data:
            runs_data.append(run_data)

    if not runs_data:
        raise HTTPException(status_code=404, detail="No runs found")

    # Convert to MATLAB-friendly format
    # Structure: runs(i).field = value
    matlab_data = {
        'num_runs': len(runs_data),
        'runs': []
    }

    for i, run in enumerate(runs_data):
        metrics = run.get('finalMetrics', {})

        run_struct = {
            'run_id': run.get('run_id', ''),
            'method': run.get('method', ''),
            'timestamp': run.get('timestamp', ''),
            'delta_v': metrics.get('delta_v_analytical', 0.0),
            'time_of_flight': metrics.get('total_time_days', 0.0),
            'fuel_consumed': metrics.get('fuel_consumed', 0.0),
            'final_cost': metrics.get('final_cost', 0.0),
            'captured': int(metrics.get('captured', False)),
            'num_iterations': len(run.get('iterations', [])),
            # Include trajectory if available
            'best_trajectory': np.array(run.get('best_trajectory', [])),
            'best_schedule': np.array(run.get('best_schedule', []))
        }

        matlab_data['runs'].append(run_struct)

    # Save to BytesIO buffer
    buffer = io.BytesIO()
    savemat(buffer, matlab_data)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=runs_export.mat"}
    )


# =============================================================================
# Detailed Trajectory Analysis Endpoint
# =============================================================================

class DetailedAnalysisRequest(BaseModel):
    """Request for detailed trajectory analysis."""
    trajectory: List[List[float]]  # Full trajectory data
    schedule: List[float]  # Thrust schedule
    params: SimulationRequest  # Simulation parameters

@app.post("/analyze/detailed")
def get_detailed_analysis(req: DetailedAnalysisRequest):
    """
    Analyze a trajectory and return detailed orbital parameters for each timestep.

    This endpoint computes:
    - Orbital elements (a, e, i, Ω, ω, ν) for each timestep
    - Distance to Earth and Moon
    - Velocity magnitude
    - Apogee and perigee altitudes
    - Periapsis/apoapsis event detection
    - Delta-V and fuel consumption

    Returns a comprehensive JSON with per-timestep data.
    """
    if not ANALYSIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Trajectory analysis module not available")

    try:
        # Convert inputs to numpy
        trajectory_np = np.array(req.trajectory)
        schedule_np = np.array(req.schedule)

        # Run analysis
        analysis = analyze_full_trajectory(
            trajectory_np,
            schedule_np,
            req.params.dt,
            req.params.thrust,
            req.params.mass,
            req.params.isp
        )

        return analysis

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"[ERROR] Analysis failed:", flush=True)
        print(error_msg, flush=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/export/trajectory_json")
def export_trajectory_json_endpoint(req: DetailedAnalysisRequest):
    """
    Export detailed trajectory analysis as a downloadable JSON file.

    This provides the same data as /analyze/detailed but as a downloadable file.
    Ideal for offline analysis, plotting, or importing into other tools.
    """
    if not ANALYSIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Trajectory analysis module not available")

    try:
        # Convert inputs to numpy
        trajectory_np = np.array(req.trajectory)
        schedule_np = np.array(req.schedule)

        # Run analysis
        analysis = analyze_full_trajectory(
            trajectory_np,
            schedule_np,
            req.params.dt,
            req.params.thrust,
            req.params.mass,
            req.params.isp
        )

        # Add metadata
        analysis['metadata'] = {
            'method': req.params.method,
            'num_steps': req.params.num_steps,
            'batch_size': req.params.batch_size,
            'mass_kg': req.params.mass,
            'thrust_N': req.params.thrust,
            'isp_s': req.params.isp,
            'apogee_altitude_km': req.params.apogee_altitude,
            'perigee_altitude_km': req.params.perigee_altitude,
            'dt_normalized': req.params.dt,
            'enable_3d': req.params.enable_3d,
            'inclination_deg': req.params.inclination_deg if req.params.enable_3d else 0.0,
            'raan_deg': req.params.raan_deg if req.params.enable_3d else 0.0,
        }

        # Convert to JSON
        json_content = json.dumps(analysis, indent=2)

        return StreamingResponse(
            io.StringIO(json_content),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=trajectory_detailed_analysis.json"}
        )

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"[ERROR] Export failed:", flush=True)
        print(error_msg, flush=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
