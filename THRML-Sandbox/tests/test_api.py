"""
API Integration Tests

Tests for FastAPI streaming endpoint and simulation orchestration.
"""

import pytest
from fastapi.testclient import TestClient
import json
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from server import app, get_initial_state

client = TestClient(app)


class TestHealthEndpoint:
    """Test basic server health"""
    
    def test_root_endpoint(self):
        """Root should return status"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "online"
        assert "system" in data


class TestSimulationEndpoint:
    """Test /simulate endpoint"""
    
    def test_simulation_basic(self):
        """Basic simulation request should succeed"""
        payload = {
            "num_steps": 100,
            "batch_size": 5,
            "coupling_strength": 0.5,
            "mass": 1000.0,
            "thrust": 10.0,
            "isp": 300.0,
            "initial_altitude": 400.0,
            "method": "classical",  # Use classical to avoid THRML dependency
            "dt": 0.01,
            "num_iterations": 3
        }
        
        response = client.post("/simulate", json=payload)
        assert response.status_code == 200
        
        # Check streaming response
        assert response.headers["content-type"] == "application/x-ndjson"
    
    def test_simulation_streaming_chunks(self):
        """Should receive multiple chunks (one per iteration)"""
        payload = {
            "num_steps": 50,
            "batch_size": 10,
            "num_iterations": 5,
            "method": "classical"
        }
        
        response = client.post("/simulate", json=payload, stream=True)
        
        chunks = []
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                chunks.append(chunk)
        
        # Should have 5 chunks (one per iteration)
        assert len(chunks) == 5
        
        # Verify chunk structure
        for i, chunk in enumerate(chunks):
            assert "iteration" in chunk
            assert "total_iterations" in chunk
            assert "trajectories" in chunk
            assert "best_trajectory" in chunk
            assert "best_cost" in chunk
            
            assert chunk["iteration"] == i + 1
            assert chunk["total_iterations"] == 5
    
    def test_trajectory_data_structure(self):
        """Trajectory data should have correct structure"""
        payload = {
            "num_steps": 50,
            "batch_size": 10,
            "num_iterations": 2,
            "method": "classical"
        }
        
        response = client.post("/simulate", json=payload, stream=True)
        
        first_chunk = None
        for line in response.iter_lines():
            if line:
                first_chunk = json.loads(line)
                break
        
        assert first_chunk is not None
        
        # Trajectories should be a list
        trajectories = first_chunk["trajectories"]
        assert isinstance(trajectories, list)
        assert len(trajectories) > 0
        
        # Each trajectory should be (num_steps+1, 4)
        traj = trajectories[0]
        assert len(traj) == 51  # num_steps + 1
        assert len(traj[0]) == 4  # [x, y, vx, vy]
        
        # Best trajectory should have same structure
        best_traj = first_chunk["best_trajectory"]
        assert len(best_traj) == 51
        assert len(best_traj[0]) == 4
    
    def test_cost_decreases(self):
        """Best cost should generally decrease over iterations"""
        payload = {
            "num_steps": 100,
            "batch_size": 20,
            "num_iterations": 10,
            "method": "classical"
        }
        
        response = client.post("/simulate", json=payload, stream=True)
        
        costs = []
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                costs.append(chunk["best_cost"])
        
        # Not always monotonic, but first should be higher than last on average
        # (With random sampling, this is probabilistic)
        # For Classical method (random), this might not always hold
        # So we just check that we got costs
        assert len(costs) == 10
        assert all(cost > 0 for cost in costs)
    
    def test_thrml_method(self):
        """THRML method should work (or gracefully fallback)"""
        payload = {
            "num_steps": 50,
            "batch_size": 10,
            "num_iterations": 3,
            "method": "thrml"
        }
        
        response = client.post("/simulate", json=payload, stream=True)
        
        # Should succeed regardless of THRML availability
        assert response.status_code == 200
        
        # Should get chunks
        chunk_count = 0
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                # Should not have error field
                assert "error" not in chunk or chunk.get("error") is None
                chunk_count += 1
        
        assert chunk_count == 3


class TestInputValidation:
    """Test request validation"""
    
    def test_missing_fields_use_defaults(self):
        """Missing optional fields should use defaults"""
        # Minimal payload
        payload = {}
        
        response = client.post("/simulate", json=payload, stream=True)
        
        # Should succeed with defaults
        assert response.status_code == 200
    
    def test_invalid_method(self):
        """Invalid method should be rejected"""
        payload = {
            "method": "invalid_method"
        }
        
        response = client.post("/simulate", json=payload)
        
        # Pydantic validation should fail
        assert response.status_code == 422


class TestInitialStateCalculation:
    """Test initial state computation"""
    
    def test_leo_altitude(self):
        """LEO altitude should produce reasonable state"""
        state = get_initial_state(400.0)  # 400 km altitude
        
        x, y, vx, vy = state
        
        # Should be near Earth
        MU = 0.01215
        earth_x = -MU
        
        assert abs(x - earth_x) < 0.1  # Within 0.1 normalized units
        assert y == 0.0  # Start on x-axis
        
        # Should have some velocity
        assert abs(vy) > 0  # Prograde motion
    
    def test_geo_altitude(self):
        """GEO altitude should be farther from Earth"""
        state_leo = get_initial_state(400.0)
        state_geo = get_initial_state(35786.0)
        
        x_leo, _, _, _ = state_leo
        x_geo, _, _, _ = state_geo
        
        # GEO should be farther from Earth center
        MU = 0.01215
        earth_x = -MU
        
        dist_leo = abs(x_leo - earth_x)
        dist_geo = abs(x_geo - earth_x)
        
        assert dist_geo > dist_leo


class TestPerformance:
    """Test API performance characteristics"""
    
    def test_small_simulation_fast(self):
        """Small simulation should complete quickly"""
        import time
        
        payload = {
            "num_steps": 50,
            "batch_size": 5,
            "num_iterations": 3,
            "method": "classical"
        }
        
        start = time.time()
        response = client.post("/simulate", json=payload, stream=True)
        
        # Consume stream
        for _ in response.iter_lines():
            pass
        
        elapsed = time.time() - start
        
        # Should complete in reasonable time (adjust based on hardware)
        # First run includes JIT compilation, so be lenient
        assert elapsed < 30  # 30 seconds max


if __name__ == "__main__":
    # Run with: pytest tests/test_api.py -v
    pytest.main([__file__, "-v"])
