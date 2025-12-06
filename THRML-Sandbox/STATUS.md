# Project Status

**Last Updated**: 2025-11-21

## Current State Summary

The Cislunar Trajectory Sandbox is in **active development**. The backend (Python/JAX + THRML) and Streamlit frontend are functional with real-time streaming visualization. Migration to a React + Three.js frontend is in progress.

---

## Component Status

### ✅ Backend (Fully Functional)

#### Physics Engine (`backend/physics.py`)
- **Status**: ✅ Working
- **Features**:
  - CR3BP (Circular Restricted Three-Body Problem) equations of motion
  - RK4 numerical integration
  - Tangential thrust model
  - Softening to avoid singularities
  - JIT-compiled for performance
- **Known Issues**: None critical
- **Testing**: Manual verification only (needs automated tests)

#### Generative Model (`backend/generative.py`)
- **Status**: ✅ Working
- **Features**:
  - THRML integration for Gibbs sampling
  - 1D Ising chain model for thrust schedules
  - Bias field updates for iterative refinement
  - Fallback to random sampling if THRML unavailable
- **Known Issues**: 
  - Warm-up iterations hardcoded (50)
  - No constraint handling for eclipse/phase angle
- **Testing**: Imports verified, needs unit tests

#### API Server (`backend/server.py`)
- **Status**: ✅ Working
- **Features**:
  - FastAPI with CORS enabled
  - Streaming response (NDJSON format)
  - Iterative solver loop with bias field updates
  - Pydantic request validation
- **Known Issues**: None
- **Testing**: Manual testing via Streamlit, needs integration tests

### ✅ Frontend - Streamlit (`app.py`)

- **Status**: ✅ Working
- **Features**:
  - Real-time streaming visualization
  - Interactive controls (mass, thrust, altitude, etc.)
  - Plotly 2D trajectory plot with Earth/Moon
  - "Ghost" trajectories + best trajectory display
  - Auto-run on parameter change
- **Known Issues**: 
  - Variable `MU` defined after use (line 106 vs 116) - non-critical due to execution order
- **Testing**: Manual verification

### 🚧 Frontend - React + Three.js (In Progress)

- **Status**: 🚧 Scaffolding
- **Progress**:
  - Vite initialization started (`npx create-vite`)
  - Plan created for component hierarchy
  - Aesthetic design defined ("Deep Void" theme, Michroma/Barlow fonts)
- **Blocking**: Vite setup needs to complete
- **Next Steps**:
  1. Complete `create-vite` command
  2. Install dependencies
  3. Create 2D canvas prototype (before Three.js)
  4. Implement streaming data consumption

### ❌ Testing Infrastructure (Not Started)

- **Unit Tests**: ❌ None
  - Need `tests/test_physics.py` for CR3BP validation
  - Need `tests/test_thrml.py` for sampling verification
  - Need `tests/test_api.py` for streaming endpoint

- **Integration Tests**: ❌ None
  - Need end-to-end simulation test

- **Physics Validation**: ❌ No benchmark comparisons
  - Should compare against published trajectories (e.g., ARTEMIS)

### ❌ Classical Solver Integration (Not Started)

- **CasADi + IPOPT**: ❌ Not implemented
- **Abstract mentions**: Using THRML to warm-start deterministic solvers
- **Current Reality**: Only THRML sampling, no refinement step
- **Impact**: This is critical for the research contribution claim

### ⚠️ Documentation (Partial)

- **Completed**:
  - ✅ `Abstract.md`: Clear research goal
  - ✅ `DESIGN.md`: System architecture (just created)
  - ✅ Implementation plans for past work
  
- **Missing**:
  - ❌ README.md (at project root)
  - ❌ API documentation
  - ❌ Hyperparameter tuning guide
  - ❌ Development setup instructions

---

## Dependency Status

### Python (Backend)
```
✅ jax / jaxlib
✅ fastapi
✅ uvicorn
✅ pydantic
✅ numpy
⚠️ thrml (vendored in thrml-main/, not pip-installed)
❌ casadi (not installed, needed for classical solver)
```

### Python (Frontend - Streamlit)
```
✅ streamlit
✅ requests
✅ plotly
```

### JavaScript (Frontend - React)
```
🚧 Vite (setup in progress)
❌ react, react-dom
❌ three, @react-three/fiber
❌ framer-motion
❌ tailwindcss
```

---

## Performance Metrics

### Backend
- **First Simulation**: ~10-15s (JIT compilation overhead)
- **Subsequent Simulations**: ~2-5s (50 iterations, batch_size=50)
- **GPU Acceleration**: Not tested (would significantly improve if available)

### Frontend (Streamlit)
- **Streaming Update Frequency**: ~20 updates/sec
- **Plot Render Time**: <100ms per update
- **Browser Performance**: Good for batch_size ≤ 100

---

## Known Issues & Technical Debt

### Critical
1. **No Classical Solver**: Abstract promises THRML + deterministic refinement, only first part exists
2. **No Automated Tests**: Zero test coverage
3. **Variable Scoping Bug in app.py**: `MU` used before definition (lines 106 vs 116)

### Important
4. **Magic Numbers**: Hyperparameters (k_best=10%, bias_scale=4.0) lack documentation
5. **No Benchmarking**: Can't measure if THRML actually helps vs random
6. **THRML Import Handling**: Silent fallback to random if import fails

### Minor
7. **No Logging**: Difficult to debug issues
8. **Hardcoded Constants**: L_STAR, T_STAR duplicated across files
9. **No Rate Limiting**: API could be DoS'd

---

## Success Criteria (Proposed)

To complete the research goal, the project should demonstrate:

1. **Convergence Improvement**: THRML finds lunar trajectories X% more often than random
2. **Solver Warm-Starting**: THRML initialization reduces IPOPT iterations by Y%
3. **Educational Value**: Clear documentation + working demo for students

**Current Achievement**: ~40% (backend works, frontend partial, no solver, no benchmarks)

---

## Next Milestones

### Short-term (1 week)
- [ ] Complete React frontend setup
- [ ] Create basic 2D visualization
- [ ] Add physics validation tests
- [ ] Document hyperparameters

### Medium-term (1 month)
- [ ] Integrate CasADi + IPOPT solver
- [ ] Implement benchmarking system
- [ ] Complete Three.js visualization
- [ ] Add constraint handling (eclipse, phase)

### Long-term (research paper)
- [ ] Compare against published methods
- [ ] Quantify improvement metrics
- [ ] Create educational tutorials
- [ ] Deploy public demo
