#!/usr/bin/env python3
"""
Test Ising Solver in 3D Mode
=============================

Verifica que el solver de Ising funciona correctamente en modo 3D.
"""

import requests
import json
import time

BACKEND_URL = "http://127.0.0.1:8080"

print("="*80)
print("TEST: ISING SOLVER EN MODO 3D")
print("="*80)

# Test 1: Verificar que el backend soporta 3D
print("\n[TEST 1] Verificando soporte 3D del backend...")
try:
    response = requests.get(f"{BACKEND_URL}/")
    data = response.json()

    if data.get('quantum_solver_available'):
        print("[OK] Quantum solver disponible")
        if 'dimensions' in data.get('quantum_solver', {}):
            print(f"[OK] Dimensiones soportadas: {data['quantum_solver']['dimensions']}")
        else:
            print("[WARNING] No se reportan dimensiones soportadas")
    else:
        print("[ERROR] Quantum solver NO disponible")
        exit(1)

except Exception as e:
    print(f"[ERROR] No se pudo conectar al backend: {e}")
    print("Ejecuta: python THRML-Sandbox/backend/launcher.py")
    exit(1)

# Test 2: Ejecutar simulación 3D con método hybrid
print("\n" + "="*80)
print("[TEST 2] Ejecutando simulacion HYBRID en modo 3D...")
print("="*80)

simulation_params = {
    "method": "hybrid",
    "num_steps": 500,  # Corto para prueba
    "batch_size": 10,
    "coupling_strength": 1.0,
    "mass": 400.0,
    "thrust": 0.07,
    "isp": 1640.0,
    "apogee_altitude": 40400.0,
    "perigee_altitude": 9600.0,
    "dt": 0.001,
    "num_iterations": 3,  # Solo 3 para prueba rápida
    "demo_mode": False,
    "enable_3d": True,  # MODO 3D ACTIVADO
    "inclination_deg": 8.0,
    "raan_deg": 0.0,
    "thrust_mode": "orbital_plane"
}

print(f"\n[*] Parametros:")
print(f"    - enable_3d: {simulation_params['enable_3d']}")
print(f"    - inclination: {simulation_params['inclination_deg']}°")
print(f"    - num_iterations: {simulation_params['num_iterations']}")

try:
    response = requests.post(
        f"{BACKEND_URL}/simulate",
        json=simulation_params,
        stream=True,
        timeout=120
    )

    print("\n[*] Simulacion iniciada...")
    print("    Buscando logs del backend...\n")

    ising_3d_detected = False
    fallback_detected = False
    iteration_count = 0

    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line)

                if 'error' in data:
                    print(f"[ERROR] {data['error']}")
                    if 'traceback' in data:
                        print(data['traceback'])
                    break

                iteration = data.get('iteration', '?')
                total = data.get('total_iterations', '?')
                best_cost = data.get('best_cost', 0)
                dimension = data.get('dimension', '?')

                print(f"  Iteracion {iteration}/{total}: Cost={best_cost:.4f}, Dim={dimension}")
                iteration_count += 1

            except json.JSONDecodeError:
                pass

    if iteration_count > 0:
        print(f"\n[OK] Simulacion completada: {iteration_count} iteraciones")
    else:
        print(f"\n[WARNING] No se recibieron datos de iteraciones")

except Exception as e:
    print(f"[ERROR] Simulacion fallo: {e}")
    import traceback
    traceback.print_exc()

# Instrucciones finales
print("\n" + "="*80)
print("[VERIFICACION MANUAL]")
print("="*80)
print("\nRevisa la consola del backend y busca:")
print("\n  [HYBRID-ISING-3D] Iteration 1: Beta=10.03, Mean energy=-45.23")
print("                     ^^^^^^^^^^")
print("\nSi ves esto, el solver de Ising esta funcionando en modo 3D!")
print("\nSi ves:")
print("  [HYBRID-FALLBACK] ... <- MAL, no esta usando Ising")
print("  [HYBRID-TEMP] ...     <- MAL, no esta usando Ising")

print("\n" + "="*80)
print("SIGUIENTE PASO:")
print("="*80)
print("\n1. Si viste [HYBRID-ISING-3D] -> Perfecto! Ya puedes usar 3D con Ising")
print("2. Ejecuta simulaciones completas en la app desktop")
print("3. Verifica los logs del backend para confirmar")
print("4. Exporta resultados y analiza con compare_classical_vs_ising.py")
