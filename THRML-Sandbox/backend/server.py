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

# Add project root for core imports
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modules
from physics import batch_propagate, dimensionalize_trajectory
from generative import generate_thrust_schedules

# Try to import core for physics-aware sampling
try:
    from core import compute_physics_bias_field, get_initial_state_4d
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    print("Core module not available, using basic sampling")

app = FastAPI(title="Cislunar Trajectory Sandbox API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationRequest(BaseModel):
    num_steps: int = 100
    batch_size: int = 50
    coupling_strength: float = 0.5
    
    # Physical Parameters
    mass: float = 1000.0 # kg
    thrust: float = 1.0 # N
    isp: float = 3000.0 # s
    initial_altitude: float = 200.0 # km (Altitude above Earth)
    
    # Method
    method: Literal["thrml", "quantum", "random"] = "thrml"
    
    # Advanced
    dt: float = 0.01
    num_iterations: int = 10

# Constants for Normalization
L_STAR = 384400.0 * 1000 # meters (Earth-Moon Distance)
T_STAR = 375200.0 # seconds (approx 4.34 days)
M_STAR = 1000.0 # kg (Reference mass, arbitrary)
MU = 0.01215

@app.get("/")
def read_root():
    return {
        "status": "online", 
        "system": "ASL-Sandbox Backend",
        "core_available": CORE_AVAILABLE,
        "methods": ["thrml", "quantum", "random"]
    }

def get_initial_state(altitude_km):
    R_EARTH_KM = 6378.0
    L_MOON_KM = 384400.0
    
    r_norm = (R_EARTH_KM + altitude_km) / L_MOON_KM
    
    x = -MU + r_norm
    y = 0.0
    
    v_mag = np.sqrt((1 - MU) / r_norm)
    
    vx = 0.0
    vy = v_mag - r_norm
    
    return [x, y, vx, vy]

@app.post("/simulate")
def run_simulation(req: SimulationRequest):
    def simulation_generator() -> Generator[str, None, None]:
        try:
            key = jax.random.PRNGKey(int(np.random.randint(0, 100000)))
            
            # 1. Calculate Normalized Thrust Acceleration
            accel_metric = req.thrust / req.mass
            accel_norm = accel_metric * (T_STAR**2 / L_STAR)
            
            # 2. Determine Initial State
            init_state_arr = jnp.array(get_initial_state(req.initial_altitude))
            
            # 3. Initialize physics-aware bias if available
            if CORE_AVAILABLE:
                initial_bias = compute_physics_bias_field(req.num_steps, None, 0.4)
                current_bias = initial_bias
            else:
                current_bias = None
            
            # 4. Moon position for cost calculation
            moon_pos = jnp.array([1 - MU, 0])
            
            # 5. Success threshold (approximately 50,000 km)
            success_threshold = 50000 / 384400.0
            
            for i in range(req.num_iterations):
                # Generate Schedules based on method
                if req.method == "thrml":
                    schedules = generate_thrust_schedules(
                        key, 
                        req.num_steps, 
                        req.batch_size, 
                        req.coupling_strength, 
                        [], [],
                        bias_field=current_bias
                    )
                elif req.method == "quantum":
                    # Try to use quantum solver if available
                    try:
                        from qntm_backend import SimulatedQuantumAnnealer
                        annealer = SimulatedQuantumAnnealer()
                        schedules = annealer.generate_schedules(
                            req.num_steps,
                            req.batch_size,
                            req.coupling_strength,
                            float(jnp.mean(current_bias)) if current_bias is not None else 0.0
                        )
                        schedules = jnp.array(schedules)
                    except ImportError:
                        # Fallback: Use biased random with annealing-like structure
                        if current_bias is not None:
                            probs = jax.nn.sigmoid(current_bias)
                        else:
                            probs = jnp.ones(req.num_steps) * 0.4
                        schedules = jax.random.bernoulli(key, probs, shape=(req.batch_size, req.num_steps)).astype(jnp.float32)
                else:  # random
                    schedules = jax.random.bernoulli(key, p=0.4, shape=(req.batch_size, req.num_steps)).astype(jnp.float32)
                    
                # Propagate Physics
                thrust_schedules_mag = schedules * accel_norm
                trajectories = batch_propagate(init_state_arr, thrust_schedules_mag, req.dt, req.num_steps)
                
                # Calculate Costs (Distance to Moon)
                final_positions = trajectories[:, -1, :2]
                dists = jnp.linalg.norm(final_positions - moon_pos, axis=1)
                
                # Calculate success rate
                successes = dists < success_threshold
                success_rate = float(jnp.mean(successes))
                
                # Select Best (Top 10%)
                k_best = max(1, int(req.batch_size * 0.1))
                best_indices = jnp.argsort(dists)[:k_best]
                best_schedules = schedules[best_indices]
                
                # Update Bias for next iteration (Cross-Entropy Method style)
                avg_schedule = jnp.mean(best_schedules, axis=0)
                learning_rate = 0.3
                if current_bias is not None:
                    current_bias = current_bias * (1 - learning_rate) + (avg_schedule - 0.5) * 4.0 * learning_rate
                else:
                    current_bias = (avg_schedule - 0.5) * 4.0
                
                # Update Key
                key, _ = jax.random.split(key)
                
                # Prepare response data
                traj_np = np.array(trajectories)
                dists_np = np.array(dists)
                
                # Get indices to send: Top 5 + Random 5
                sorted_indices = np.argsort(dists_np)
                top_indices = sorted_indices[:5]
                random_indices = np.random.choice(req.batch_size, 5, replace=False)
                display_indices = np.unique(np.concatenate([top_indices, random_indices]))
                
                chunk_trajectories = traj_np[display_indices]
                best_trajectory = traj_np[sorted_indices[0]]
                
                chunk_data = {
                    "iteration": i + 1,
                    "total_iterations": req.num_iterations,
                    "trajectories": chunk_trajectories.tolist(),
                    "best_cost": float(np.min(dists_np)),
                    "mean_cost": float(np.mean(dists_np)),
                    "best_trajectory": best_trajectory.tolist(),
                    "success_rate": success_rate,
                    "method": req.method,
                }
                
                yield json.dumps(chunk_data) + "\n"
                
        except Exception as e:
            import traceback
            yield json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc()
            }) + "\n"

    return StreamingResponse(simulation_generator(), media_type="application/x-ndjson")

@app.post("/benchmark")
def run_benchmark(methods: List[str], num_samples: int = 100, num_steps: int = 200):
    """Run comparative benchmark across methods."""
    results = {}
    
    for method in methods:
        if method not in ["thrml", "quantum", "random"]:
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
        init_state_arr = jnp.array(get_initial_state(req.initial_altitude))
        
        if method == "thrml":
            schedules = generate_thrust_schedules(
                key, num_steps, num_samples, req.coupling_strength, [], []
            )
        elif method == "quantum":
            probs = jnp.ones(num_steps) * 0.4
            schedules = jax.random.bernoulli(key, probs, shape=(num_samples, num_steps)).astype(jnp.float32)
        else:
            schedules = jax.random.bernoulli(key, p=0.4, shape=(num_samples, num_steps)).astype(jnp.float32)
        
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
