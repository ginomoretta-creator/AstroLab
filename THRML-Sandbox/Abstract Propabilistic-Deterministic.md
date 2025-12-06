**Paper title**:

Hybrid Probabilistic–Deterministic Sandbox for Earth–Moon Low-Thrust Transfers

**Abstract**:

Low-thrust Earth-to-Moon transfers are sensitive to thrust scheduling and often fail to converge when initialized from uninformed control profiles. Classical direct-collocation solvers can refine trajectories to high accuracy, but they depend strongly on a physically meaningful initial guess. This work introduces an open cislunar trajectory sandbox that investigates probabilistic inference as a strategy for generating structured burn/coast patterns to warm-start deterministic optimal-control methods.



The transfer horizon is discretized and thrust activation is represented as a binary schedule. A factor-based energy model encodes mission-relevant structure, including apoapsis-raising behavior, phasing toward lunar encounter, low arrival relative velocity, thrust duty-cycle, smoothness of thrust windows, and eclipse-based power constraints. Using THRML, a GPU-accelerated probabilistic sampling library, we draw low-energy thrust schedules via blocked Gibbs updates. These schedules are smoothed into continuous thrust profiles and refined with a direct-collocation solver (CasADi + IPOPT) to obtain dynamically feasible trajectories.



The sandbox evaluates whether probabilistic sampling improves solver robustness and reduces reliance on handcrafted initialization for low-thrust missions. Designed for research and education, it enables students and mission designers to explore structured trajectory generation, solver convergence behavior, and physics-guided probabilistic methods for cislunar mission design.





THRML:

THRML is a JAX library for building and sampling probabilistic graphical models, with a focus on efficient block Gibbs sampling and energy-based models. Extropic is developing hardware to make sampling from certain classes of discrete PGMs massively more energy efficient; THRML provides GPU‑accelerated tools for block sampling on sparse, heterogeneous graphs, making it a natural place to prototype today and experiment with future Extropic hardware.

