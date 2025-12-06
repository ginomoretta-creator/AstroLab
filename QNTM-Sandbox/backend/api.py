import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import jax.numpy as jnp
import numpy as np
import json
from fastapi.responses import StreamingResponse

from .quantum_solver import SimulatedQuantumAnnealer
from .physics import batch_propagate, dimensionalize_trajectory, MU

app = FastAPI(title="QNTM-Sandbox API")

# Initialize Annealer
annealer = SimulatedQuantumAnnealer()

class SimulationRequest(BaseModel):
    num_steps: int
    batch_size: int
    coupling_strength: float
    bias: float
    mass: float
    thrust: float
    isp: float
    initial_altitude: float
    method: str
    dt: float = 0.01
    num_iterations: int = 1

@app.post("/simulate")
async def simulate(request: SimulationRequest):
    """
    Runs the quantum-assisted trajectory simulation.
    Returns a stream of JSON objects (ndjson style) or a single JSON response.
    For now, we'll return a single JSON object in a stream to satisfy the frontend's iter_lines().
    """
    
    # 1. Setup Initial State
    # Earth radius ~6378 km. Altitude is above surface.
    # Convert to dimensionless units.
    L_STAR = 384400.0 # km
    R_EARTH_KM = 6378.0
    
    r0_km = R_EARTH_KM + request.initial_altitude
    r0_dim = r0_km / L_STAR
    
    # Start at Earth (-mu, 0) + radius
    # We'll start on the x-axis for simplicity, or a circular parking orbit state.
    # Let's assume a simple circular parking orbit for now.
    # v_circ = sqrt(mu_earth / r)
    # In CR3BP, velocity is in rotating frame.
    
    # For simplicity in this sandbox, let's just start at a fixed point with some velocity.
    # Better: Use the physics engine to define a stable starting orbit if possible, 
    # but for now, we'll just pick a point.
    
    x0 = -MU + r0_dim
    y0 = 0.0
    vx0 = 0.0
    vy0 = 1.6 # Approximate orbital velocity in rotating frame? 
              # v_inertial ~ 7.8 km/s (LEO) / 1.024 km/s (V*) ~ 7.6
              # v_rot = v_in - cross(omega, r)
              # This is complex to get right instantly, let's stick to a reasonable guess 
              # or allow the user to tweak it later. 
              # For the "Sandbox" nature, we'll use a fixed guess.
    
    initial_state = jnp.array([x0, y0, vx0, vy0])
    
    # 2. Generate Thrust Schedules
    # We need to normalize thrust.
    # F = ma => a = F/m
    # a_dim = (F / m) * (T_star^2 / L_star) ... 
    # Let's just use the raw value passed for now as a "magnitude" parameter 
    # relative to the system, or doing a rough conversion.
    # 1 N / 1000 kg = 0.001 m/s^2.
    # Earth gravity at LEO ~ 9 m/s^2.
    # Low thrust is very small.
    # Let's treat the 'thrust' input as a multiplier for the 'thrust_mag' in the physics engine.
    # In the physics engine, thrust_mag is added to acceleration.
    # We'll assume the user inputs are somewhat calibrated or we just use them as is for the "Sandbox".
    
    # Map method to annealer parameters
    if request.method == "quantum-annealing":
        # Use the requested coupling and bias
        pass
    else:
        # Classical Random: coupling=0, bias=0 (or just random noise)
        # The annealer with J=0 is effectively random coin flips if T is high enough 
        # or we can just ignore J.
        # But the annealer code handles J=0 fine.
        pass

    # Generate schedules
    # (batch_size, num_steps)
    # Generate schedules
    # Returns dict with schedules, energies, h, J
    solver_output = annealer.generate_thrust_schedules(
        num_steps=request.num_steps,
        batch_size=request.batch_size,
        coupling_strength=request.coupling_strength if request.method == "quantum-annealing" else 0.0,
        bias=request.bias if request.method == "quantum-annealing" else 0.0
    )
    
    schedules = solver_output["schedules"]
    
    # Scale schedules by thrust magnitude
    # F = ma => a = F/m (m/s^2)
    # Characteristic Acceleration A* = V*^2 / L* (approx) or V* * omega
    # L* = 384400 km = 3.844e8 m
    # V* = 1024 m/s
    # A* = 1024^2 / 3.844e8 ~= 0.002727 m/s^2
    
    L_STAR_M = 384400.0 * 1000.0
    V_STAR_MS = 1024.0
    A_STAR = (V_STAR_MS ** 2) / L_STAR_M
    
    accel_m_s2 = request.thrust / request.mass
    accel_dim = accel_m_s2 / A_STAR
    
    with open("debug.log", "a") as f:
        f.write(f"DEBUG: Mass={request.mass}, Thrust={request.thrust}\n")
        f.write(f"DEBUG: Accel (m/s^2)={accel_m_s2}, Accel (dim)={accel_dim}\n")
        f.write(f"DEBUG: Schedule Mean={jnp.mean(schedules)}\n")
    
    thrust_schedules = schedules * accel_dim
    
    with open("debug.log", "a") as f:
        f.write(f"DEBUG: Thrust Schedule Max={jnp.max(thrust_schedules)}\n")
    
    # 3. Propagate
    # (batch_size, num_steps+1, 4)
    trajectories = batch_propagate(initial_state, thrust_schedules, request.dt, request.num_steps)
    
    # 4. Evaluate Costs (Distance to Moon)
    # Moon is at (1-mu, 0)
    moon_pos = jnp.array([1.0 - MU, 0.0])
    
    # Get final positions
    final_positions = trajectories[:, -1, :2]
    distances = jnp.linalg.norm(final_positions - moon_pos, axis=1)
    
    best_idx = jnp.argmin(distances)
    best_cost = float(distances[best_idx])
    best_traj = trajectories[best_idx]
    
    # 5. Format Response
    # Convert JAX arrays to standard lists for JSON serialization
    response_data = {
        "trajectories": np.array(trajectories).tolist(),
        "best_trajectory": np.array(best_traj).tolist(),
        "best_cost": best_cost,
        "ising_params": {
            "h": solver_output["h"],
            "J": solver_output["J"]
        },
        "sample_energies": np.array(solver_output["energies"]).tolist()
    }
    
    # Generator for streaming response
    async def response_generator():
        yield json.dumps(response_data) + "\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
