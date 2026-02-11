# 📝 Guía para el Paper SSEA26

## ✅ Cambios Implementados - RESUMEN EJECUTIVO

**Problema Original**: El código en `server.py` usaba Bernoulli con temperatura, NO un Hamiltoniano de Ising real.

**Solución**: Ahora el método "hybrid" usa **D-Wave Neal** con Hamiltoniano de Ising correctamente implementado.

**Status**: ✅ **LISTO PARA REPORTAR EN EL PAPER**

---

## 🎓 Descripción Correcta del Método Hybrid

### Para el Abstract

```
"We propose a hybrid quantum-classical approach for low-thrust trajectory
optimization using a 1D Ising Hamiltonian solved via D-Wave's Neal simulated
annealing sampler. The Ising model encodes binary thrust decisions as spins
with ferromagnetic coupling to enforce trajectory smoothness, while
physics-aware external fields guide the search toward feasible solutions.
We compare this approach against a pure classical Cross-Entropy Method (CEM)
on cislunar transfer missions."
```

### Para la Sección de Metodología

#### 2.1 Classical Baseline: Cross-Entropy Method (CEM)

```
The classical method uses a standard Cross-Entropy approach:

1. Initialize: Sample thrust schedules from Bernoulli(p=0.4)
2. Evaluate: Propagate trajectories and compute costs
3. Select: Choose elite top-10% based on minimum cost
4. Update: Refine probability distribution toward elite mean
5. Repeat: Iterate until convergence

Probability update rule:
    p^(t+1) = (1-α) * p^(t) + α * mean(elite schedules)

where α = 0.3 is the learning rate.
```

#### 2.2 Hybrid Quantum-Classical: Ising + CEM

```
The hybrid method combines quantum-inspired sampling with classical refinement:

**Ising Hamiltonian:**

    E = -J * Σ_{i=1}^{N-1} s_i * s_{i+1} - Σ_{i=1}^{N} h_i * s_i

where:
  • s_i ∈ {-1, +1}: spin representing thrust on/off at timestep i
  • J > 0: ferromagnetic coupling strength (promotes smoothness)
  • h_i: external field at timestep i (physics-aware bias)
  • N: number of timesteps

**Sampling Procedure:**

1. Initialize: Set h_i = 0 for all i
2. For each iteration t:
   a. Configure annealing schedule:
      β_max = 10.0 * (1 + t/T)
      where β = 1/kT (inverse temperature)

   b. Construct Binary Quadratic Model:
      BQM(h, J, vartype=SPIN)

   c. Sample using D-Wave Neal:
      samples = SimulatedAnnealingSampler.sample(
          BQM,
          num_reads=batch_size,
          num_sweeps=1000,
          beta_range=(0.1, β_max)
      )

   d. Map spins to binary thrust: s_i → (s_i + 1)/2

   e. Evaluate trajectories and select elite top-10%

   f. Update external fields (CEM-style):
      h_i^(t+1) = (1-α) * h_i^(t) + α * target_bias(elite)

**Key Differences from Classical:**
  • Sampling: Ising annealing vs independent Bernoulli
  • Structure: Coupling J enforces smoothness
  • Exploration: Temperature schedule β vs fixed probabilities
```

### Para la Sección de Experimentos

```
**Experimental Setup:**

Platform: Intel Core i7, 16GB RAM
Software: Python 3.11, JAX 0.4.23, D-Wave Neal 0.6.0

**Parameters:**
  • Initial orbit: 200 km × 7,500 km (perigee × apogee)
  • Spacecraft mass: 400 kg
  • Thrust: 0.07 N (Hall thruster)
  • Specific impulse: 1,640 s
  • Timesteps: N = 1,000 (dt = 0.001 normalized units ≈ 4.3 days)
  • Batch size: 50 samples per iteration
  • Iterations: 30
  • Coupling strength: J = 1.0 (Ising model)
  • Learning rate: α = 0.3 (both methods)

**Metrics:**
  1. Delta-V consumption (m/s) - lower is better
  2. Minimum distance to Moon (km) - lower is better
  3. Trajectory smoothness: 1/(number of thrust changes) - higher is better
  4. Time of flight (days)
  5. Fuel consumption (kg)

**Runs:** 5 independent runs per method (different random seeds)
```

---

## 📊 Cómo Ejecutar los Experimentos

### Paso 1: Reiniciar el Backend

```bash
# Detener cualquier instancia anterior
# Luego iniciar el backend
cd THRML-Sandbox/backend
python launcher.py
```

**Verificar en consola:**
```
[QUANTUM] D-Wave Neal Ising solver available  ← DEBE aparecer
```

### Paso 2: Ejecutar 5 Runs del Método Classical

```bash
cd desktop-app
npm run electron:dev
```

Para cada run (i=1 a 5):
1. Seleccionar método: **"Classical (CEM)"**
2. Configurar parámetros:
   - num_steps: 1000
   - num_iterations: 30
   - batch_size: 50
   - coupling_strength: 1.0 (no se usa en classical, pero consistencia)
   - enable_3d: **FALSE** (importante)
3. Clic en "Start"
4. Esperar a que termine (30 iteraciones)
5. En "Iteration History", exportar la **MEJOR** iteración:
   - Ordena por "Dist" (distancia a la Luna)
   - Haz clic en ⬇️ de la mejor
6. Renombrar el archivo descargado a: `Classical_run{i}.json`
7. Mover a `JSON_RESULTS/`

### Paso 3: Ejecutar 5 Runs del Método Hybrid-Ising

Repetir el proceso anterior, pero:
1. Seleccionar método: **"Hybrid Quantum-Classical"**
2. Mismos parámetros que Classical
3. **VERIFICAR EN LA CONSOLA DEL BACKEND:**
   ```
   [HYBRID-ISING] Iteration 1: Beta=10.03, Mean energy=-156.23
   [HYBRID-ISING] Iteration 2: Beta=10.07, Mean energy=-168.45
   ```
   **Si ves `[HYBRID-FALLBACK]` o `[HYBRID-TEMP]`** → ALGO ESTÁ MAL, no uses esos datos

4. Exportar y renombrar a: `Hybrid_Ising_run{i}.json`

### Paso 4: Analizar Resultados

```bash
python compare_classical_vs_ising.py
```

Esto genera:
- `comparacion_metricas.png` - Gráficos comparativos
- `tabla_comparacion.tex` - Tabla LaTeX lista para copiar
- `comparacion_datos.csv` - Datos para procesamiento adicional

---

## 📈 Análisis Estadístico

Con 5 runs de cada método, calcular:

```python
import numpy as np
import json
from pathlib import Path

# Cargar datos
classical_runs = [json.load(open(f)) for f in Path("JSON_RESULTS").glob("Classical_run*.json")]
hybrid_runs = [json.load(open(f)) for f in Path("JSON_RESULTS").glob("Hybrid_Ising_run*.json")]

# Extraer métricas
def get_delta_v(data):
    return data['summary']['delta_v_ms']

def get_min_dist(data):
    return data['summary']['closest_moon_approach']['distance_km']

# Delta-V
classical_dv = [get_delta_v(r) for r in classical_runs]
hybrid_dv = [get_delta_v(r) for r in hybrid_runs]

print(f"Classical Delta-V: {np.mean(classical_dv):.1f} ± {np.std(classical_dv):.1f} m/s")
print(f"Hybrid Delta-V:    {np.mean(hybrid_dv):.1f} ± {np.std(hybrid_dv):.1f} m/s")

# Test t de Student
from scipy import stats
t_stat, p_value = stats.ttest_ind(classical_dv, hybrid_dv)
print(f"t-test: t={t_stat:.3f}, p={p_value:.4f}")

if p_value < 0.05:
    print("Diferencia estadísticamente significativa (p < 0.05)")
```

---

## 📝 Tabla de Resultados para el Paper

### Tabla 1: Comparación de Métricas (Media ± SD)

```latex
\begin{table}[h]
\centering
\caption{Performance Comparison: Classical vs Hybrid-Ising (5 runs each)}
\label{tab:results}
\begin{tabular}{lrrr}
\hline
\textbf{Metric} & \textbf{Classical} & \textbf{Hybrid-Ising} & \textbf{p-value} \\
\hline
Delta-V (m/s) & 12,129 ± 156 & 11,987 ± 143 & 0.042* \\
Min. dist to Moon (km) & 19,584 ± 1,234 & 18,921 ± 987 & 0.038* \\
Smoothness (1/changes) & 0.0234 ± 0.003 & 0.0412 ± 0.005 & 0.001** \\
Fuel consumed (kg) & 158.9 ± 2.1 & 157.2 ± 1.8 & 0.051 \\
\hline
\multicolumn{4}{l}{* p < 0.05, ** p < 0.01 (two-tailed t-test)}
\end{tabular}
\end{table}
```

*(Nota: Estos son valores de ejemplo - usa tus datos reales)*

---

## 🎯 Afirmaciones que PUEDES Hacer en el Paper

✅ **CORRECTO:**
- "The hybrid method employs a 1D Ising Hamiltonian solved via D-Wave Neal simulated annealing"
- "Ferromagnetic coupling promotes trajectory smoothness"
- "Physics-aware external fields guide the quantum-inspired search"
- "Our implementation uses the Binary Quadratic Model (BQM) formulation"

❌ **INCORRECTO (NO DIGAS ESTO):**
- ~~"We use a real quantum computer"~~ (es simulado)
- ~~"Quantum tunneling provides speedup"~~ (es simulado, no hay speedup cuántico real)
- ~~"The method scales to 3D"~~ (actualmente solo 2D)

⚠️ **SÉ HONESTO:**
- "We simulate quantum annealing behavior using D-Wave's Neal classical sampler"
- "This approach mimics quantum exploration without requiring quantum hardware"
- "Future work will extend to 3D trajectories and test on actual quantum annealers"

---

## 📚 Referencias Clave para el Paper

```bibtex
@article{dwave_neal,
  title={D-Wave Neal: Simulated Annealing Sampler},
  author={D-Wave Systems},
  journal={GitHub repository},
  year={2023},
  url={https://github.com/dwavesystems/dwave-neal}
}

@article{rubinstein1999cross,
  title={The cross-entropy method for combinatorial and continuous optimization},
  author={Rubinstein, Reuven Y},
  journal={Methodology and computing in applied probability},
  volume={1},
  number={2},
  pages={127--190},
  year={1999}
}

@article{izzo2007pygmo,
  title={PyGMO and PyKEP: Open source tools for massively parallel optimization in astrodynamics},
  author={Izzo, Dario},
  journal={Proceed. of ICATT},
  year={2012}
}
```

---

## ✅ Checklist Final para el Paper

Antes de enviar a SSEA26:

- [ ] ✅ Código usa D-Wave Neal **realmente** (verificado en logs)
- [ ] ✅ Descripción del Hamiltoniano es **matemáticamente correcta**
- [ ] ✅ Ecuación del Ising está en la metodología
- [ ] ✅ Parámetros (J, β, α, N) están **reportados**
- [ ] ✅ 5+ runs de cada método ejecutados
- [ ] ✅ Estadísticas (media ± SD) calculadas
- [ ] ✅ Test t de Student para significancia
- [ ] ✅ Gráficos generados con datos **reales del Ising**
- [ ] ✅ Tabla LaTeX con resultados incluida
- [ ] ✅ Limitaciones (2D only, simulado) **documentadas**
- [ ] ✅ Referencias a D-Wave Neal incluidas
- [ ] ✅ Código está disponible (GitHub link en el paper)

---

## 🚀 Siguiente Paso INMEDIATO

```bash
# 1. Verificar que el solver funciona
cd THRML-Sandbox/backend
python launcher.py
# Busca: [QUANTUM] D-Wave Neal Ising solver available

# 2. Ejecutar primera comparación
cd ../..
cd desktop-app
npm run electron:dev
# Ejecuta 1 run classical + 1 run hybrid
# Exporta ambos a JSON_RESULTS/

# 3. Comparar
python compare_classical_vs_ising.py
```

---

**¡El solver de Ising está conectado y listo!** 🎉

Ahora puedes reportar honestamente en el paper que usas un Hamiltoniano de Ising con D-Wave Neal.

**Buena suerte con el SSEA26!** 📝🚀
