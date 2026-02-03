"""
Test script for CasADi classical solver.
"""
import sys
import os
import numpy as np

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
asl_root = os.path.dirname(current_dir)
if asl_root not in sys.path:
    sys.path.insert(0, asl_root)

from core.classical_solver import (
    create_collocation_problem,
    smooth_schedule,
    CASADI_AVAILABLE
)
from core.constants import MU

def test_casadi_solver():
    """Test the CasADi solver with a simple lunar transfer."""
    print("=" * 60)
    print("Testing CasADi Classical Solver")
    print("=" * 60)

    if not CASADI_AVAILABLE:
        print("\nCASADI NOT AVAILABLE")
        print("Install with: pip install casadi")
        return False

    print("\nCasADi is available")

    # Initial state: Low Earth orbit (200 km altitude)
    # x = -MU + r, where r = (6378 + 200) / 384400 ~= 0.0171
    r_earth = (6378.0 + 200.0) / 384400.0
    x0 = -MU + r_earth
    y0 = 0.0

    # Circular velocity
    v_circ = np.sqrt((1 - MU) / r_earth)
    vx0 = 0.0
    vy0 = v_circ - x0

    # Initial mass (normalized by 500 kg)
    m0 = 1.0

    initial_state = np.array([x0, y0, vx0, vy0, m0])

    print(f"\nInitial State:")
    print(f"  Position: ({x0:.4f}, {y0:.4f})")
    print(f"  Velocity: ({vx0:.4f}, {vy0:.4f})")
    print(f"  Mass: {m0:.4f}")

    # Create simple initial thrust guess (50% duty cycle)
    num_nodes = 200
    thrust_guess = np.ones(num_nodes) * 0.5

    # Solver parameters
    T_total = 10.0  # Transfer time (normalized)
    thrust_max = 0.01  # Max thrust (normalized)
    isp_normalized = 3000.0 / 375200.0  # ~0.008

    print(f"\nSolver Parameters:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Transfer Time: {T_total:.2f} (normalized)")
    print(f"  Max Thrust: {thrust_max:.4f}")
    print(f"  ISP: {isp_normalized:.6f}")

    print("\n" + "-" * 60)
    print("Running CasADi/IPOPT Solver...")
    print("-" * 60)

    try:
        result = create_collocation_problem(
            x0=initial_state,
            thrust_initial_guess=thrust_guess,
            num_nodes=num_nodes,
            T_total=T_total,
            thrust_max=thrust_max,
            isp_normalized=isp_normalized,
            lunar_capture_radius=0.172,  # Lunar SOI
            min_final_mass_fraction=0.3
        )

        print("\n" + "=" * 60)
        print("SOLVER RESULT")
        print("=" * 60)

        if result.success:
            print("\nSolver SUCCEEDED")
            print(f"  Final Mass: {result.final_mass:.4f}")
            print(f"  Delta-V: {result.delta_v:.2f} m/s" if result.delta_v else "  Delta-V: N/A")
            print(f"  Iterations: {result.iterations}")
            print(f"  Solve Time: {result.solve_time_seconds:.2f} seconds")

            if result.trajectory is not None:
                final_state = result.trajectory[-1]
                moon_pos = np.array([1 - MU, 0])
                final_pos = final_state[:2]
                dist_to_moon = np.linalg.norm(final_pos - moon_pos)
                print(f"\n  Final Position: ({final_pos[0]:.4f}, {final_pos[1]:.4f})")
                print(f"  Distance to Moon: {dist_to_moon:.4f} ({dist_to_moon * 384400:.0f} km)")

                if dist_to_moon < 0.172:
                    print("  Within Lunar SOI!")
                else:
                    print(f"  Outside Lunar SOI (threshold: 0.172)")

            return True
        else:
            print(f"\nSolver FAILED")
            print(f"  Message: {result.message}")
            print(f"  Iterations: {result.iterations}")
            print(f"  Solve Time: {result.solve_time_seconds:.2f} seconds")
            return False

    except Exception as e:
        print(f"\nException during solve: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_casadi_solver()
    sys.exit(0 if success else 1)
