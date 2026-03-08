# Cislunar Trajectory Sandbox

A hybrid probabilistic–deterministic research tool for Earth-Moon low-thrust trajectory optimization, combining THRML (probabilistic inference) with classical astrodynamics.

This sandbox investigates whether **probabilistic inference can warm-start deterministic optimal control methods** for low-thrust Earth-to-Moon transfers.

**Core Innovation**: Using [THRML](https://github.com/extropic-ai/thrml) (a JAX-based probabilistic graphical model library) to sample physically-meaningful thrust schedules that help classical solvers converge.

### Research Question

> Can structured probabilistic sampling (via THRML) improve trajectory optimization convergence compared to uninformed random search?
