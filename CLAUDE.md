# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AstroLab (ASL-Sandbox) is a research project comparing classical and hybrid quantum-classical approaches for low-thrust trajectory optimization in cislunar space. The project consists of:

1. **Classical Optimization**: Cross-Entropy Method (CEM) - pure classical iterative optimization
2. **Hybrid Quantum-Classical**: Quantum annealing (simulated) + classical refinement
3. **Desktop Application**: Electron-based standalone app that bundles the frontend and Python backend

**Key Goal**: Compare classical optimization performance against hybrid quantum-classical approaches to quantify potential quantum hardware benefits for trajectory design.

## Repository Structure

```
ASL-Sandbox/
├── desktop-app/              # Electron desktop application wrapper
│   ├── electron/            # Main and preload scripts
│   ├── src/                # React/TypeScript UI
│   └── resources/          # Bundled backend (in production builds)
├── THRML-Sandbox/
│   ├── backend/            # FastAPI server for trajectory simulation
│   │   ├── server.py       # Main API endpoints (/simulate, /benchmark)
│   │   ├── launcher.py     # PyInstaller entry point
│   │   └── physics.py      # CR3BP propagation wrapper
│   └── frontend/           # React web interface (Three.js visualization)
├── QNTM-Sandbox/           # Quantum annealing implementation (simulated)
└── core/                   # Shared physics and optimization modules
    ├── physics_core.py     # CR3BP dynamics, propagation
    ├── energy_model.py     # Physics-aware bias fields
    ├── classical_solver.py # CasADi/IPOPT direct collocation
    └── constants.py        # Physical constants, normalization
```

## Commands

### Desktop App (Recommended for Development)

The desktop app combines the frontend UI with an automatically-managed Python backend.

```bash
cd desktop-app
npm install
npm run electron:dev    # Run in development mode (hot reload, dev tools)
npm run build          # Build frontend
npm run electron:build # Build packaged executable
```

**Note**: The desktop app looks for the backend in two places:
1. `resources/backend/asl-sandbox-backend.exe` (bundled production build)
2. `../THRML-Sandbox/backend/server.py` (development fallback)

### THRML-Sandbox / Frontend (Web-based Development)

**Frontend**:
```bash
cd THRML-Sandbox/frontend
npm install
npm run dev     # Start Vite dev server on http://localhost:5173
npm run build   # Build for production
```

**Backend**:
```bash
cd THRML-Sandbox/backend
python -m venv .venv
.\.venv\Scripts\Activate   # Windows
source .venv/bin/activate  # Unix
pip install -r requirements.txt
python launcher.py         # Start FastAPI server on http://127.0.0.1:8080
# OR
uvicorn server:app --host 127.0.0.1 --port 8080
```

### Core Module Development

The `core` module contains shared physics and optimization code:

```bash
cd core
python test_casadi_solver.py       # Test CasADi solver
pytest                              # Run all core tests (if pytest configured)
```

## Architecture

### Backend Architecture

**Backend Flow**:
1. **API Layer** (`server.py`): FastAPI endpoints accept simulation parameters
2. **Method Selection**:
   - **Classical**: Cross-Entropy Method (CEM) with physics-aware bias
   - **Hybrid**: Quantum annealing (simulated) + classical refinement
3. **Physics Layer** (`physics.py` → `core/physics_core.py`): CR3BP propagation
4. **Streaming Response**: Results stream back to frontend as NDJSON for real-time visualization

**Frontend Flow**:
1. **Control Panel** (`ControlPanel.jsx`): User selects method and parameters
2. **Simulation Hook** (`useSimulation.js`): Manages API calls and state
3. **Canvas** (`SimulationCanvas.jsx`): Three.js rendering of Earth, Moon, and trajectories
4. **Stats Overlay** (`StatsOverlay.jsx`): Real-time metrics display

**Key Concepts**:
- **Cross-Entropy Method (CEM)**: Iteratively refines probability distribution over thrust schedules
- **Bias Fields**: Physics-aware biasing guides sampling toward physically realistic trajectories
- **Batch Propagation**: JAX-accelerated parallel trajectory propagation
- **Cost Function**: Multi-objective cost balancing Moon SOI arrival, fuel usage, and velocity constraints
- **Success Criterion**: Reach Moon SOI (< 66,100 km) with low velocity (< 1 km/s)

### Optimization Methods

**Classical Method: Cross-Entropy Method (CEM)**
- Pure classical iterative optimization
- Maintains probability distribution over thrust schedules
- Iteratively refines distribution based on elite samples (top 10%)
- Learning rate α = 0.3 balances exploration vs exploitation
- No external dependencies beyond JAX

**Hybrid Quantum-Classical Method**
- Phase 1: Quantum annealing samples from Ising spin model
  - Maps thrust schedule to spins s_i ∈ {-1, +1}
  - Energy: E = -Σ J_ij s_i s_j - Σ h_i s_i
  - Currently simulated using D-Wave Neal library
- Phase 2: Classical refinement using CEM bias update
- Expected quantum hardware speedup: 2-10x end-to-end (see docs/CLASSICAL_VS_HYBRID_COMPARISON.md)

### Desktop App Architecture

The Electron app manages the Python backend lifecycle:

1. **Backend Discovery** (`electron/main.ts`):
   - Production: Looks for bundled `.exe` in `resources/backend/`
   - Development: Falls back to running `server.py` with Python
2. **Process Management**: Spawns backend on app start, terminates on quit
3. **IPC Communication**: Exposes backend status/control to renderer via `window.api`
4. **Frontend Loading**:
   - Dev: Loads from `http://localhost:5173` (Vite dev server)
   - Production: Loads from bundled `dist/index.html`

## Important Technical Details

### Physics (CR3BP Dynamics)

The backend simulates trajectories in the **Circular Restricted Three-Body Problem** (Earth-Moon system):
- **Normalized Units**: Distances normalized by Earth-Moon distance (384,400 km), time by ~4.34 days
- **Rotating Frame**: Coordinates rotate with Earth-Moon system (Earth near origin, Moon at (1-μ, 0))
- **Mass Parameter**: μ = 0.01215 (Moon/Earth mass ratio)
- **Propagation**: 4th-order Runge-Kutta integration in `physics.py:batch_propagate()`

### Thrust Schedule Generation

Both methods generate binary thrust profiles (on/off at each timestep):
- Binary schedules are easier to optimize than continuous controls
- Coupling strength (if used) encourages smooth thrust arcs
- Physics-aware bias fields guide sampling toward feasible trajectories
- Bias is updated iteratively based on best performers (CEM-style)

### Backend API Endpoints

**`POST /simulate`**: Main simulation endpoint
- **Streaming Response**: Returns NDJSON stream of iteration results
- **Iterative Refinement**: Runs multiple iterations, updating bias based on best performers
- **Cost Function**: Balances Moon SOI arrival, approach progress, fuel usage, and velocity constraints
- **Methods**: `"classical"` (CEM), `"hybrid"` (quantum annealing + classical)
- **Success Criterion**: Distance to Moon < 66,100 km AND velocity < 1 km/s

**`POST /benchmark`**: Comparative method evaluation
- Single-iteration comparison across methods
- Returns aggregate statistics (mean/min distance, success rate)

**`GET /`**: Health check
- Returns backend status and available methods

### Git Workflow Notes

This repository excludes build artifacts (`.exe`, `dist/`, `build/`, `.venv/`) but tracks source code. When building the desktop app, the backend executable is generated at build time via PyInstaller.

## Development Tips

1. **Desktop App Testing**: Use `npm run electron:dev` to test with hot reload - the app will automatically find your local Python backend if no bundled executable exists
2. **Backend Changes**: Restart the backend via the desktop app's debug panel or manually restart `launcher.py`
3. **JAX Debugging**: Set `JAX_DISABLE_JIT=1` to disable JIT compilation for easier debugging
4. **Backend Logs**: Check Electron console for backend stdout/stderr in development mode
5. **Comparison Testing**: See `docs/CLASSICAL_VS_HYBRID_COMPARISON.md` for benchmark procedures

## Important Files

**Backend**:
- `THRML-Sandbox/backend/server.py` - Main API, method selection, cost function
- `core/physics_core.py` - CR3BP dynamics, trajectory propagation
- `core/energy_model.py` - Physics-aware bias field computation
- `core/classical_solver.py` - CasADi/IPOPT direct collocation (experimental)

**Frontend**:
- `THRML-Sandbox/frontend/src/components/ControlPanel.jsx` - Method selection UI
- `THRML-Sandbox/frontend/src/hooks/useSimulation.js` - API interaction
- `THRML-Sandbox/frontend/src/components/SimulationCanvas.jsx` - 3D visualization

## Testing

**Core Module**:
```bash
cd core
python test_casadi_solver.py       # Test CasADi direct collocation
```

**Backend**:
```bash
cd THRML-Sandbox/backend
python launcher.py                  # Start server and test manually
curl -X POST http://localhost:8080/simulate \
  -H "Content-Type: application/json" \
  -d '{"method": "classical", "num_steps": 1000}'
```

## Common Issues

- **Backend won't start**: Check Python dependencies are installed (`pip install -r requirements.txt`)
- **JAX errors**: Ensure JAX and jaxlib versions are compatible with your platform
- **Electron app crashes**: Check backend logs in Electron console (F12 in dev mode)
- **Trajectory visualization issues**: Verify Three.js camera controls and coordinate system consistency
