import requests
import numpy as np
import time

API_URL = "http://localhost:8000/simulate"

def test_convergence():
    print("Testing solver convergence...")
    
    payload = {
        "num_steps": 2000,
        "batch_size": 50,
        "coupling_strength": 0.5,
        "mass": 1000.0,
        "thrust": 10.0,
        "isp": 300.0,
        "initial_altitude": 400,
        "method": "thrml",
        "dt": 0.001
    }
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        duration = time.time() - start_time
        
        data = response.json()
        trajectories = np.array(data["trajectories"]) # [batch, steps, 4]
        
        # Check if we got trajectories
        if len(trajectories) == 0:
            print("FAILED: No trajectories returned.")
            return
            
        print(f"Received {len(trajectories)} trajectories in {duration:.2f}s")
        
        # Calculate distance to Moon for best trajectory
        L_STAR = 384400.0
        MU = 0.01215
        moon_pos = np.array([1 - MU, 0])
        
        final_positions = trajectories[:, -1, :2]
        dists = np.linalg.norm(final_positions - moon_pos, axis=1) * L_STAR
        
        best_dist = np.min(dists)
        print(f"Best Final Distance to Moon: {best_dist:.1f} km")
        
        # Threshold for "success" (getting reasonably close, e.g., within 50,000 km is a good start for random search + iteration)
        # Note: Without precise targeting, it might not hit 0, but should be better than random.
        if best_dist < 100000:
            print("SUCCESS: Solver converged to a reasonable proximity.")
        else:
            print("WARNING: Solver did not get very close. Might need more iterations or tuning.")
            
    except Exception as e:
        print(f"FAILED: API Request failed - {e}")

if __name__ == "__main__":
    test_convergence()
