import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Constants
API_URL = "http://localhost:8000/simulate"
L_STAR = 384400.0  # km (Earth-Moon Distance)
MU = 0.01215  # Earth-Moon mass parameter
R_EARTH = 6378.0 / L_STAR
R_MOON = 1737.0 / L_STAR

st.set_page_config(page_title="Cislunar AI Sandbox", layout="wide", page_icon="🚀")

# --- Sidebar Controls ---
st.sidebar.header("🚀 Mission Parameters")

method = st.sidebar.radio("Guidance Method", ["thrml", "classical"], format_func=lambda x: "THRML (AI)" if x == "thrml" else "Classical")

st.sidebar.subheader("Spacecraft")
mass = st.sidebar.number_input("Mass (kg)", value=1000.0, step=100.0)
thrust = st.sidebar.number_input("Thrust (N)", value=10.0, step=1.0)
isp = st.sidebar.number_input("ISP (s)", value=300.0, step=10.0)

st.sidebar.subheader("Initial State")
initial_altitude = st.sidebar.slider("Altitude (km)", 200, 36000, 400, step=100)

st.sidebar.subheader("Simulation Config")
coupling = st.sidebar.slider("Coupling (Smoothness)", 0.0, 2.0, 0.5, 0.1, help="Higher values enforce smoother trajectories.")
steps = st.sidebar.slider("Time Steps", 500, 20000, 5000, 500)
batch_size = st.sidebar.slider("Batch Size", 10, 200, 50, 10)
iterations = st.sidebar.slider("Solver Iterations", 10, 200, 50, 10)

# --- Main UI ---
# --- Main UI ---
st.title("Cislunar Trajectory Design")
st.markdown(f"""
**Method:** `{method.upper()}` | **Altitude:** `{initial_altitude} km` | **Thrust:** `{thrust} N`
""")

# Auto-run simulation on any input change
with st.spinner("Calculating trajectories..."):
    try:
        payload = {
            "num_steps": steps,
            "batch_size": batch_size,
            "coupling_strength": coupling,
            "mass": mass,
            "thrust": thrust,
            "isp": isp,
            "initial_altitude": initial_altitude,
            "method": method,
            "dt": 0.01,
            "num_iterations": iterations
        }
        
        # Streaming Request
        response = requests.post(API_URL, json=payload, stream=True)
        response.raise_for_status()
        
        # Placeholder for the plot
        plot_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Iterate over the stream
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "error" in chunk:
                    st.error(f"Simulation error: {chunk['error']}")
                    break
                
                iteration = chunk["iteration"]
                total_iterations = chunk["total_iterations"]
                trajectories = np.array(chunk["trajectories"]) # [batch_subset, steps, 4]
                best_traj = np.array(chunk["best_trajectory"])
                best_cost = chunk["best_cost"]
                
                # --- Visualization ---
                fig = go.Figure()

                # 1. Plot Ghost Trajectories (Faint)
                for i in range(len(trajectories)):
                    traj = trajectories[i]
                    fig.add_trace(go.Scatter(
                        x=traj[:, 0] * L_STAR,
                        y=traj[:, 1] * L_STAR,
                        mode='lines',
                        line=dict(color='rgba(100, 116, 139, 0.1)', width=1),
                        hoverinfo='skip',
                        showlegend=False
                    ))

                # 2. Plot Best Trajectory (Bright)
                fig.add_trace(go.Scatter(
                    x=best_traj[:, 0] * L_STAR,
                    y=best_traj[:, 1] * L_STAR,
                    mode='lines',
                    name=f'Best (Iter {iteration})',
                    line=dict(color='#3b82f6', width=3),
                ))

                # 3. Earth
                # Earth is at (-MU, 0) in rotating frame
                earth_x = -MU * L_STAR
                fig.add_shape(type="circle",
                    xref="x", yref="y",
                    x0=earth_x - R_EARTH * L_STAR, y0=-R_EARTH * L_STAR,
                    x1=earth_x + R_EARTH * L_STAR, y1=R_EARTH * L_STAR,
                    fillcolor="#2563eb", line_color="#1e40af", opacity=0.8
                )
                fig.add_trace(go.Scatter(x=[earth_x], y=[0], mode='text', text=['Earth'], textposition="bottom center", showlegend=False))

                # 4. Moon
                moon_x = (1 - MU) * L_STAR
                
                fig.add_shape(type="circle",
                    xref="x", yref="y",
                    x0=moon_x - R_MOON * L_STAR, y0=-R_MOON * L_STAR,
                    x1=moon_x + R_MOON * L_STAR, y1=R_MOON * L_STAR,
                    fillcolor="#94a3b8", line_color="#64748b", opacity=0.8
                )
                fig.add_trace(go.Scatter(x=[moon_x], y=[0], mode='text', text=['Moon'], textposition="bottom center", showlegend=False))

                # Layout
                fig.update_layout(
                    title=f"Trajectory Optimization - Iteration {iteration}/{total_iterations}",
                    xaxis_title="X (km)",
                    yaxis_title="Y (km)",
                    template="plotly_dark",
                    width=800,
                    height=800,
                    showlegend=True,
                    margin=dict(l=20, r=20, t=40, b=20),
                    yaxis=dict(scaleanchor="x", scaleratio=1)
                )
                
                # Update Plot
                plot_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Update Metrics
                with metrics_placeholder.container():
                    c1, c2 = st.columns(2)
                    c1.metric("Iteration", f"{iteration}/{total_iterations}")
                    c2.metric("Best Dist to Moon", f"{best_cost * L_STAR:.1f} km")

    except Exception as e:
        st.error(f"Simulation failed: {e}")
