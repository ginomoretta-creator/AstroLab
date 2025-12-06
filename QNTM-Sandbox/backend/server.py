import sys
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Literal, Generator
import json
from fastapi.responses import StreamingResponse
import jax
import jax.numpy as jnp
import numpy as np

# Ensure backend is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import modules
from physics import batch_propagate, dimensionalize_trajectory
from quantum_solver import SimulatedQuantumAnnealer

app = FastAPI(title="QNTM-Sandbox API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Quantum Annealer
annealer = SimulatedQuantumAnnealer()

class SimulationRequest(BaseModel):
    num_steps: int = 100
    batch_size: int = 50
    coupling_strength: float = 1.0 # J
    bias: float = 0.0 # h
    
    # Physical Parameters
    mass: float = 1000.0 # kg
    thrust: float = 1.0 # N
    isp: float = 300.0 # s
    initial_altitude: float = 200.0 # km
    
    # Method
    method: Literal["quantum-annealing", "classical-random"] = "quantum-annealing"
    
    dt: float = 0.01
    num_iterations: int = 1 # For now, just one shot of annealing + propagation

# Constants for Normalization
L_STAR = 384400.0 * 1000 # meters
T_STAR = 375200.0 # seconds

@app.get("/")
def read_root():
    return {"status": "online", "system": "QNTM-Sandbox Backend"}

def get_initial_state(altitude_km):
    # Earth Radius ~ 6378 km
    R_EARTH_KM = 6378.0
    L_MOON_KM = 384400.0
    
    r_norm = (R_EARTH_KM + altitude_km) / L_MOON_KM
    
    # Earth is at (-MU, 0)
    MU = 0.01215
    x = -MU + r_norm
    y = 0.0
    
    # Velocity: Vis-viva or simple circular approximation
    v_mag = np.sqrt((1 - MU) / r_norm)
    
    vx = 0.0
    vy = v_mag - r_norm # Prograde
    
    return [x, y, vx, vy]

@app.post("/simulate")
def run_simulation(req: SimulationRequest):
    def simulation_generator() -> Generator[str, None, None]:
        try:
            # 1. Calculate Normalized Thrust Acceleration
            accel_metric = req.thrust / req.mass
            accel_norm = accel_metric * (T_STAR**2 / L_STAR)
            
            # 2. Determine Initial State
            init_state_arr = jnp.array(get_initial_state(req.initial_altitude))
            
            # 3. Generate Schedules (Quantum or Classical)
            if req.method == "quantum-annealing":
                schedules = annealer.generate_thrust_schedules(
                    num_steps=req.num_steps,
                    batch_size=req.batch_size,
                    coupling_strength=req.coupling_strength,
                    bias=req.bias
                )
            else:
                # Classical Random (Bernoulli p=0.5)
                key = jax.random.PRNGKey(np.random.randint(0, 100000))
                schedules = jax.random.bernoulli(key, p=0.5, shape=(req.batch_size, req.num_steps)).astype(jnp.float32)
            
            # 4. Propagate Physics
            thrust_schedules_mag = schedules * accel_norm
            trajectories = batch_propagate(init_state_arr, thrust_schedules_mag, req.dt, req.num_steps)
            
            # 5. Calculate Costs (Distance to Moon)
            MU = 0.01215
            moon_pos = jnp.array([1 - MU, 0])
            final_positions = trajectories[:, -1, :2]
            dists = jnp.linalg.norm(final_positions - moon_pos, axis=1)
            
            # 6. Prepare Response
            # We stream just one chunk for now as it's a "One Shot" generation
            # But we structure it to allow iteration later if we add refinement loops
            
            traj_np = np.array(trajectories)
            
            # Get indices to send: Top 5 + Random 5
            sorted_indices = np.argsort(dists)
            top_indices = sorted_indices[:5]
            random_indices = np.random.choice(req.batch_size, 5, replace=False)
            display_indices = np.unique(np.concatenate([top_indices, random_indices]))
            
            chunk_trajectories = traj_np[display_indices]
            
            chunk_data = {
                "iteration": 1,
                "total_iterations": 1,
                "trajectories": chunk_trajectories.tolist(),
                "best_cost": float(np.min(dists)),
                "best_trajectory": traj_np[top_indices[0]].tolist()
            }
            
            yield json.dumps(chunk_data) + "\n"
                
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(simulation_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) # Port 8001 to avoid conflict with THRML
