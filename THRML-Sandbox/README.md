# Cislunar Trajectory Sandbox

A hybrid probabilistic–deterministic research tool for Earth-Moon low-thrust trajectory optimization, combining THRML (probabilistic inference) with classical astrodynamics.

![Status](https://img.shields.io/badge/status-active%20development-yellow)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![JAX](https://img.shields.io/badge/JAX-GPU%20accelerated-green)

---

## 🎯 Project Overview

This sandbox investigates whether **probabilistic inference can warm-start deterministic optimal control methods** for low-thrust Earth-to-Moon transfers.

**Core Innovation**: Using [THRML](https://github.com/extropic-ai/thrml) (a JAX-based probabilistic graphical model library) to sample physically-meaningful thrust schedules that help classical solvers converge.

### Research Question

> Can structured probabilistic sampling (via THRML) improve trajectory optimization convergence compared to uninformed random search?

---

## 🏗️ Architecture

```
Frontend (React/Streamlit)
    ↓ HTTP Streaming
Backend (FastAPI)
    ↓ Iterative Optimization
THRML Sampler ←→ CR3BP Physics (JAX)
    ↓
Trajectory Candidates
```

**Key Components**:
- **Physics Engine** (`backend/physics.py`): CR3BP dynamics with RK4 integration
- **Generative Model** (`backend/generative.py`): THRML-based thrust schedule sampling
- **API Server** (`backend/server.py`): FastAPI with streaming responses
- **Frontend** (Streamlit `app.py` or React `frontend/`): Real-time visualization

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+ (for React frontend)
- Optional: CUDA-capable GPU for acceleration

### 1. Clone & Setup

```bash
git clone <repository-url>
cd THRML-Sandbox

# Install Python dependencies
pip install -r backend/requirements.txt

# Optional: Install THRML properly
pip install -e thrml-main/
```

### 2. Start Backend

```bash
python backend/server.py
```

Backend runs at `http://localhost:8000`

### 3. Start Frontend

**Option A: Streamlit (Simple, working)**
```bash
streamlit run app.py
```

**Option B: React (Advanced, under development)**
```bash
cd frontend
npm install
npm run dev
```

### 4. Run Simulation

1. Adjust parameters (mass, thrust, altitude, coupling strength)
2. Click "Simulate" or let auto-run trigger
3. Watch real-time trajectory optimization!

---

## 📚 Documentation

- **[DESIGN.md](DESIGN.md)**: System architecture, data flow, state management
- **[STATUS.md](STATUS.md)**: Component status, known issues, roadmap
- **[docs/TUNING.md](docs/TUNING.md)**: Hyperparameter explanations and tuning guide
- **[docs/BENCHMARKING.md](docs/BENCHMARKING.md)**: Success metrics and benchmark protocols
- **[Abstract.md](Abstract.md)**: Research abstract

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_physics.py -v
pytest tests/test_thrml.py -v
pytest tests/test_api.py -v
```

**Test Coverage**:
- ✅ Physics validation (CR3BP dynamics, energy conservation)
- ✅ THRML integration (sampling, coupling, bias fields)
- ✅ API streaming endpoint
- ⏳ Benchmarking (planned)

---

## 🎮 Usage Examples

### Python API

```python
import requests
import json

payload = {
    "num_steps": 5000,
    "batch_size": 50,
    "coupling_strength": 0.5,
    "mass": 1000.0,
    "thrust": 10.0,
    "initial_altitude": 400.0,
    "method": "thrml",
    "num_iterations": 50
}

response = requests.post(
    "http://localhost:8000/simulate",
    json=payload,
    stream=True
)

for line in response.iter_lines():
    if line:
        chunk = json.loads(line)
        print(f"Iteration {chunk['iteration']}: "
              f"Best distance = {chunk['best_cost']:.4f}")
```

### Command Line (via workflow)

```bash
# Start dev servers
# (Defined in .agent/workflows/start-dev-server.md)
```

---

## 🔬 How It Works

### 1. Thrust Schedule Generation (THRML)

Creates a 1D Ising model where:
- **Nodes**: Time steps (thrust on/off decision)
- **Edges**: Sequential coupling (smoothness constraint)
- **Bias field**: Learned from best previous trajectories

```python
# Ising Energy
E = -Σ J·s_i·s_{i+1}  (coupling)
    -Σ h_i·s_i        (bias field)
```

### 2. Physics Propagation (CR3BP)

Propagates each thrust schedule through 3-body dynamics:

```python
# Equations of Motion (rotating frame)
ẍ - 2ẏ = ∂Ω/∂x + thrust_x
ÿ + 2ẋ = ∂Ω/∂y + thrust_y
```

### 3. Iterative Refinement

1. Sample thrust schedules from THRML
2. Propagate physics → get trajectories
3. Evaluate cost (distance to Moon)
4. Select top 10% → update bias field
5. Repeat

**Result**: Bias field guides future sampling toward successful regions.

---

## 📊 Performance

**Typical Run** (50 iterations, batch_size=50):
- First iteration: ~10-15s (JIT compilation)
- Subsequent: ~3-5s total
- With GPU: ~5-10x faster

**Convergence** (preliminary):
- THRML: ~60-80% reach Moon
- Random: ~20-30% reach Moon

---

## 🛠️ Technology Stack

### Backend
- **JAX**: GPU-accelerated physics simulation
- **THRML**: Probabilistic graphical models (Gibbs sampling)
- **FastAPI**: Async streaming API
- **NumPy**: Data processing

### Frontend
- **Streamlit**: Rapid prototyping UI
- **React** (in progress): Production-ready interface
- **Plotly/Three.js**: 2D/3D visualization
- **Framer Motion**: UI animations

---

## 🗺️ Roadmap

### ✅ Completed
- CR3BP physics engine with JAX
- THRML integration for thrust scheduling
- Streaming API with real-time updates
- Streamlit frontend with live visualization
- Comprehensive test suite
- Documentation (design, tuning, benchmarking)

### 🚧 In Progress
- React + Three.js frontend
- Component-based UI architecture

### 📋 Planned
- CasADi + IPOPT classical solver integration
- Constraint handling (eclipse, phase angle)
- Fuel consumption tracking (ISP integration)
- Published trajectory validation (ARTEMIS comparison)
- Automated benchmarking CI/CD
- Educational tutorials

---

## 🎓 Educational Use

This sandbox is designed for:
- **Students**: Learn about CR3BP, low-thrust optimization, probabilistic methods
- **Researchers**: Prototype trajectory optimization algorithms
- **Mission Designers**: Explore feasibility of low-thrust lunar transfers

**Key Learning Outcomes**:
1. Understanding CR3BP dynamics
2. Probabilistic inference for combinatorial optimization
3. Hybrid AI + physics approaches
4. Real-time visualization of iterative algorithms

---

## 📖 References

### Theory
- **CR3BP**: Koon et al., *Dynamical Systems, the Three-Body Problem and Space Mission Design* (2011)
- **Low-Thrust Opt**: Taheri & Abdelkhalik, "Fast Initial Trajectory Design for Low-Thrust Restricted-Three-Body Problems" (2016)
- **Probabilistic Methods**: Rubinstein & Kroese, *The Cross-Entropy Method* (2004)

### Missions
- **ARTEMIS**: Folta et al., "Earth-Moon Libration Point Orbit Stationkeeping" (2012)
- **GRAIL**: Roncoli & Fujii, "Mission Design Overview for the Gravity Recovery and Interior Laboratory" (2010)

### Software
- **JAX**: [github.com/google/jax](https://github.com/google/jax)
- **THRML**: [github.com/extropic-ai/thrml](https://github.com/extropic-ai/thrml)

---

## 🤝 Contributing

This is a research/educational project. Contributions welcome!

**Areas needing help**:
- Classical solver (CasADi) integration
- Additional physics models (J2, perturbations)
- Frontend development (React components)
- Benchmarking against published methods

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **THRML Library**: Extropic AI
- **CR3BP Methods**: NASA JPL, ESA Advanced Concepts Team
- **Inspiration**: ARTEMIS mission design team

---

**Status**: Active Development | Last Updated: November 2025

For questions or collaboration: [Contact Information]
