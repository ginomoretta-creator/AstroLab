import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Constants
API_URL = "http://localhost:8001/simulate"
L_STAR = 384400.0  # km (Earth-Moon Distance)
MU = 0.01215  # Earth-Moon mass parameter
R_EARTH = 6378.0 / L_STAR
R_MOON = 1737.0 / L_STAR

st.set_page_config(page_title="QNTM-Sandbox", layout="wide", page_icon="⚛️")

# --- Sidebar Controls ---
st.sidebar.header("⚛️ Quantum Parameters")

method = st.sidebar.radio("Initialization Method", ["quantum-annealing", "classical-random"], format_func=lambda x: "Quantum Annealing (Simulated)" if x == "quantum-annealing" else "Classical Random")

if method == "quantum-annealing":
    st.sidebar.info("Simulating a 1D Ising Chain to generate structured thrust schedules.")
    coupling = st.sidebar.slider("Coupling Strength (J)", 0.0, 5.0, 1.0, 0.1, help="Higher J = Smoother, clumpier schedules (Ferromagnetic).")
    bias = st.sidebar.slider("Bias Field (h)", -1.0, 1.0, 0.0, 0.1, help="Positive = Prefer Thrust, Negative = Prefer Coast.")
else:
    coupling = 0.0
    bias = 0.0

st.sidebar.subheader("Spacecraft")
mass = st.sidebar.number_input("Mass (kg)", value=1000.0, step=100.0)
thrust = st.sidebar.number_input("Thrust (N)", value=10.0, step=1.0)
isp = st.sidebar.number_input("ISP (s)", value=300.0, step=10.0)

st.sidebar.subheader("Initial State")
initial_altitude = st.sidebar.slider("Altitude (km)", 200, 50000, 36000, step=1000)

st.sidebar.subheader("Simulation Config")
steps = st.sidebar.slider("Time Steps", 100, 10000, 2000, 100)
dt = st.sidebar.slider("Time Step (dt)", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
batch_size = st.sidebar.slider("Batch Size (Reads)", 10, 200, 50, 10)

# --- Main UI ---
st.title("Quantum-Assisted Trajectory Initialization")

# 1. Collect Simulation Parameters
current_params = {
    "num_steps": steps,
    "batch_size": batch_size,
    "coupling_strength": coupling,
    "bias": bias,
    "mass": mass,
    "thrust": thrust,
    "isp": isp,
    "initial_altitude": initial_altitude,
    "method": method,
    "dt": dt,
    "num_iterations": 1
}

# 2. Run Simulation (Only if params changed)
if "last_params" not in st.session_state:
    st.session_state.last_params = {}

# Check equality (ignoring float precision issues for simplicity, or just rely on exact match)
params_changed = st.session_state.last_params != current_params

if params_changed:
    with st.spinner("Annealing & Propagating..."):
        try:
            response = requests.post(API_URL, json=current_params, stream=True)
            response.raise_for_status()
            
            # Consume the stream (we only care about the final result for now for the state)
            # But to keep the "live" feel, we could update a placeholder. 
            # For now, let's just get the last chunk.
            final_chunk = None
            for line in response.iter_lines():
                if line:
                    final_chunk = json.loads(line)
            
            if final_chunk and "error" not in final_chunk:
                st.session_state.sim_results = final_chunk
                st.session_state.last_params = current_params
            elif final_chunk and "error" in final_chunk:
                st.error(f"Simulation error: {final_chunk['error']}")
                
        except Exception as e:
            st.error(f"Connection failed. Is the backend running on port 8001? Error: {e}")

# 3. Visualization
if "sim_results" in st.session_state:
    results = st.session_state.sim_results
    trajectories = np.array(results["trajectories"])
    best_traj = np.array(results["best_trajectory"])
    best_cost = results["best_cost"]
    ising_params = results.get("ising_params", {})
    energies = results.get("sample_energies", [])

    # Create Tabs
    tab_sim, tab_quantum = st.tabs(["🚀 Trajectory Simulation", "⚛️ Quantum Inspection"])

    with tab_sim:
        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Dist to Moon", f"{best_cost * L_STAR:.1f} km")
        c2.metric("Coupling (J)", f"{coupling}")
        c3.metric("Bias (h)", f"{bias}")

        # Animation Slider
        total_steps = len(best_traj)
        anim_step = st.slider("Animation Step", 0, total_steps, total_steps, key="anim_slider")
        
        # Plot
        fig = go.Figure()

        # Ghost Trajectories (Full)
        for i in range(len(trajectories)):
            traj = trajectories[i]
            fig.add_trace(go.Scatter(
                x=traj[:, 0] * L_STAR,
                y=traj[:, 1] * L_STAR,
                mode='lines',
                line=dict(color='rgba(147, 51, 234, 0.1)', width=1),
                hoverinfo='skip',
                showlegend=False
            ))

        # Best Trajectory (Animated)
        # Show path up to anim_step
        fig.add_trace(go.Scatter(
            x=best_traj[:anim_step, 0] * L_STAR,
            y=best_traj[:anim_step, 1] * L_STAR,
            mode='lines',
            name='Best Candidate',
            line=dict(color='#d946ef', width=3),
        ))
        
        # Current Position Marker
        if anim_step > 0:
            current_pos = best_traj[anim_step-1]
            fig.add_trace(go.Scatter(
                x=[current_pos[0] * L_STAR],
                y=[current_pos[1] * L_STAR],
                mode='markers',
                marker=dict(color='#d946ef', size=10, symbol='diamond'),
                showlegend=False
            ))

        # Earth & Moon
        earth_x = -MU * L_STAR
        fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=earth_x - R_EARTH * L_STAR, y0=-R_EARTH * L_STAR,
            x1=earth_x + R_EARTH * L_STAR, y1=R_EARTH * L_STAR,
            fillcolor="#2563eb", line_color="#1e40af", opacity=0.8
        )
        
        moon_x = (1 - MU) * L_STAR
        fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=moon_x - R_MOON * L_STAR, y0=-R_MOON * L_STAR,
            x1=moon_x + R_MOON * L_STAR, y1=R_MOON * L_STAR,
            fillcolor="#94a3b8", line_color="#64748b", opacity=0.8
        )

        fig.update_layout(
            title="Earth-Moon Trajectory",
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            template="plotly_dark",
            width=800,
            height=800,
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_quantum:
        st.subheader("Ising Model Inspection")
        
        if ising_params:
            col_q1, col_q2 = st.columns(2)
            
            with col_q1:
                st.markdown("### Bias Field ($h$)")
                h_vals = ising_params["h"]
                # h is a dict {index: value}, convert to list sorted by index
                h_list = [h_vals[str(i)] for i in range(len(h_vals))]
                fig_h = go.Figure(go.Bar(y=h_list, marker_color='#3b82f6'))
                fig_h.update_layout(title="Qubit Biases (Thrust Probability)", xaxis_title="Time Step", yaxis_title="Bias (h)", template="plotly_dark")
                st.plotly_chart(fig_h, use_container_width=True)
                
            with col_q2:
                st.markdown("### Coupling Matrix ($J$)")
                # J is dict {"(i, j)": val}
                # Construct matrix
                num_qubits = len(h_list)
                J_matrix = np.zeros((num_qubits, num_qubits))
                J_vals = ising_params["J"]
                for k, v in J_vals.items():
                    # k is string "(i, j)"
                    # Parse it
                    try:
                        # Remove parens and split
                        parts = k.replace('(', '').replace(')', '').split(',')
                        i, j = int(parts[0]), int(parts[1])
                        J_matrix[i, j] = v
                        J_matrix[j, i] = v # Symmetric
                    except:
                        pass
                
                fig_j = go.Figure(go.Heatmap(z=J_matrix, colorscale="Viridis"))
                fig_j.update_layout(title="Coupling Strength (Smoothness)", xaxis_title="Qubit i", yaxis_title="Qubit j", template="plotly_dark")
                st.plotly_chart(fig_j, use_container_width=True)
        
        if energies:
            st.markdown("### Energy Landscape")
            fig_e = go.Figure(go.Histogram(x=energies, nbinsx=20, marker_color='#ef4444'))
            fig_e.update_layout(title="Sample Energies (Lower is Better)", xaxis_title="Energy", yaxis_title="Count", template="plotly_dark")
            st.plotly_chart(fig_e, use_container_width=True)
