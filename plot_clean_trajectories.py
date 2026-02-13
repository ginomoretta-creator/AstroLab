#!/usr/bin/env python3
"""
Clean Trajectory Visualization - Simple and High Quality
=========================================================

Generates clean, publication-quality trajectory plots focusing solely
on the trajectory paths without thrust markers or clutter.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# High-quality figure settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['lines.linewidth'] = 1.0

# File paths
CLASSICAL_FILE = Path("JSON_RESULTS/CLASSICAL.json")
HYBRID_FILE = Path("JSON_RESULTS/HYBRID.json")
OUTPUT_DIR = Path("clean_trajectory_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("CLEAN TRAJECTORY VISUALIZATION")
print("="*80)

# Load data
print("\n[*] Loading trajectory data...")
with open(CLASSICAL_FILE, 'r') as f:
    classical_data = json.load(f)

with open(HYBRID_FILE, 'r') as f:
    hybrid_data = json.load(f)

print(f"[OK] Classical: {classical_data['metadata']['method']}")
print(f"[OK] Hybrid: {hybrid_data['metadata']['method']}")

# Extract trajectories
classical_timesteps = classical_data['timestep_data']
hybrid_timesteps = hybrid_data['timestep_data']

classical_traj = np.array([[t['x'], t['y'], t['z']] for t in classical_timesteps])
hybrid_traj = np.array([[t['x'], t['y'], t['z']] for t in hybrid_timesteps])

print(f"\n[*] Classical: {len(classical_traj)} points")
print(f"[*] Hybrid: {len(hybrid_traj)} points")

# Physical constants
L_STAR = 384400.0  # km (Earth-Moon distance)
MU = 0.01215  # Moon/Earth mass ratio

# Earth and Moon positions (Earth at origin)
earth_pos = np.array([0, 0, 0])
moon_pos = np.array([1, 0, 0])

# ============================================================================
# PLOT 1: Classical 3D Trajectory (Clean)
# ============================================================================
print("\n[*] Generating Classical 3D trajectory...")

fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

x = classical_traj[:, 0]
y = classical_traj[:, 1]
z = classical_traj[:, 2]

# Plot trajectory with gradient color (start to end)
n_points = len(x)
colors = plt.cm.viridis(np.linspace(0, 1, n_points))

# Plot as a continuous colored line
for i in range(n_points - 1):
    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2],
            color=colors[i], linewidth=1.2, alpha=0.8)

# Add Earth and Moon spheres
earth_radius = 0.006
moon_radius = 0.005

u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
x_sphere = earth_radius * np.outer(np.cos(u), np.sin(v))
y_sphere = earth_radius * np.outer(np.sin(u), np.sin(v))
z_sphere = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x_sphere + earth_pos[0], y_sphere + earth_pos[1],
                z_sphere + earth_pos[2], color='dodgerblue', alpha=0.8, label='Earth')
ax.plot_surface(x_sphere * (moon_radius/earth_radius) + moon_pos[0],
                y_sphere * (moon_radius/earth_radius) + moon_pos[1],
                z_sphere * (moon_radius/earth_radius) + moon_pos[2],
                color='lightgray', alpha=0.8, label='Moon')

# No end marker - trajectory speaks for itself

# Clean labels
ax.set_xlabel('X', fontweight='bold', labelpad=15)
ax.set_ylabel('Y', fontweight='bold', labelpad=15)
ax.set_zlabel('Z', fontweight='bold', labelpad=15)
ax.set_title('Classical Method - Complete 3D Trajectory',
             fontweight='bold', fontsize=26, pad=25)

# Legend
ax.legend(loc='upper right', framealpha=0.95, fontsize=19)

# Clean grid
ax.grid(True, alpha=0.2, linestyle='--')

# Set viewing angle for best perspective
ax.view_init(elev=25, azim=45)

# Remove background panels for cleaner look
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Light gray pane edges
ax.xaxis.pane.set_edgecolor('lightgray')
ax.yaxis.pane.set_edgecolor('lightgray')
ax.zaxis.pane.set_edgecolor('lightgray')

plt.tight_layout()
output_file = OUTPUT_DIR / "classical_3d_clean.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Saved: {output_file}")
plt.close()

# ============================================================================
# PLOT 2: Hybrid 3D Trajectory (Clean)
# ============================================================================
print("[*] Generating Hybrid 3D trajectory...")

fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

x = hybrid_traj[:, 0]
y = hybrid_traj[:, 1]
z = hybrid_traj[:, 2]

# Plot trajectory with gradient color
n_points = len(x)
colors = plt.cm.plasma(np.linspace(0, 1, n_points))

for i in range(n_points - 1):
    ax.plot(x[i:i+2], y[i:i+2], z[i:i+2],
            color=colors[i], linewidth=1.2, alpha=0.8)

# Add Earth and Moon spheres
earth_radius = 0.006
moon_radius = 0.005

u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
x_sphere = earth_radius * np.outer(np.cos(u), np.sin(v))
y_sphere = earth_radius * np.outer(np.sin(u), np.sin(v))
z_sphere = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x_sphere + earth_pos[0], y_sphere + earth_pos[1],
                z_sphere + earth_pos[2], color='dodgerblue', alpha=0.8, label='Earth')
ax.plot_surface(x_sphere * (moon_radius/earth_radius) + moon_pos[0],
                y_sphere * (moon_radius/earth_radius) + moon_pos[1],
                z_sphere * (moon_radius/earth_radius) + moon_pos[2],
                color='lightgray', alpha=0.8, label='Moon')

# No end marker - trajectory speaks for itself

# Clean labels
ax.set_xlabel('X', fontweight='bold', labelpad=15)
ax.set_ylabel('Y', fontweight='bold', labelpad=15)
ax.set_zlabel('Z', fontweight='bold', labelpad=15)
ax.set_title('Hybrid Quantum-Classical Method - Complete 3D Trajectory',
             fontweight='bold', fontsize=26, pad=25)

# Legend
ax.legend(loc='upper right', framealpha=0.95, fontsize=19)

# Clean grid
ax.grid(True, alpha=0.2, linestyle='--')

# Set viewing angle
ax.view_init(elev=25, azim=45)

# Remove background panels
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor('lightgray')
ax.yaxis.pane.set_edgecolor('lightgray')
ax.zaxis.pane.set_edgecolor('lightgray')

plt.tight_layout()
output_file = OUTPUT_DIR / "hybrid_3d_clean.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Saved: {output_file}")
plt.close()

# ============================================================================
# PLOT 3: Classical Projections (XY and XZ Combined)
# ============================================================================
print("[*] Generating Classical projections (XY + XZ)...")

fig, (ax_xy, ax_xz) = plt.subplots(2, 1, figsize=(12, 16))

# === XY Projection (Top) ===
x = classical_traj[:, 0]
y = classical_traj[:, 1]
z = classical_traj[:, 2]

# Plot trajectory with gradient
n_points = len(x)
colors = plt.cm.viridis(np.linspace(0, 1, n_points))

for i in range(n_points - 1):
    ax_xy.plot(x[i:i+2], y[i:i+2],
               color=colors[i], linewidth=1.0, alpha=0.8)

# No end marker - trajectory speaks for itself

# Add Earth and Moon
ax_xy.scatter([earth_pos[0]], [earth_pos[1]],
              s=200, c='dodgerblue', marker='o',
              edgecolors='navy', linewidth=1.5,
              label='Earth', zorder=90)
ax_xy.scatter([moon_pos[0]], [moon_pos[1]],
              s=100, c='lightgray', marker='o',
              edgecolors='gray', linewidth=1.5,
              label='Moon', zorder=90)

ax_xy.set_xlabel('X', fontweight='bold', fontsize=20)
ax_xy.set_ylabel('Y', fontweight='bold', fontsize=20)
ax_xy.set_title('a) XY Projection (Top View)',
                fontweight='bold', fontsize=24, pad=15)
ax_xy.legend(framealpha=0.95, fontsize=18, loc='best')
ax_xy.grid(True, alpha=0.2, linestyle='--')
ax_xy.set_aspect('equal')

# === XZ Projection (Bottom) ===
for i in range(n_points - 1):
    ax_xz.plot(x[i:i+2], z[i:i+2],
               color=colors[i], linewidth=1.0, alpha=0.8)

# No end marker - trajectory speaks for itself

# Add Earth and Moon
ax_xz.scatter([earth_pos[0]], [earth_pos[2]],
              s=200, c='dodgerblue', marker='o',
              edgecolors='navy', linewidth=1.5,
              label='Earth', zorder=90)
ax_xz.scatter([moon_pos[0]], [moon_pos[2]],
              s=100, c='lightgray', marker='o',
              edgecolors='gray', linewidth=1.5,
              label='Moon', zorder=90)

ax_xz.set_xlabel('X', fontweight='bold', fontsize=20)
ax_xz.set_ylabel('Z', fontweight='bold', fontsize=20)
ax_xz.set_title('b) XZ Projection (Side View)',
                fontweight='bold', fontsize=24, pad=15)
ax_xz.legend(framealpha=0.95, fontsize=18, loc='upper left')
ax_xz.grid(True, alpha=0.2, linestyle='--')
ax_xz.set_aspect('equal')

# Ensure both subplots have the same X-axis range for consistent width
x_min, x_max = min(x.min(), x.min()), max(x.max(), x.max())
x_range = x_max - x_min
x_center = (x_max + x_min) / 2
# Add 10% padding
x_padding = x_range * 0.1
ax_xy.set_xlim(x_center - (x_range/2 + x_padding), x_center + (x_range/2 + x_padding))
ax_xz.set_xlim(x_center - (x_range/2 + x_padding), x_center + (x_range/2 + x_padding))

# Overall title
fig.suptitle('Classical Method - Trajectory Projections',
             fontweight='bold', fontsize=26, y=0.995)

plt.tight_layout()
output_file = OUTPUT_DIR / "classical_projections_clean.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Saved: {output_file}")
plt.close()

# ============================================================================
# PLOT 4: Hybrid Projections (XY and XZ Combined)
# ============================================================================
print("[*] Generating Hybrid projections (XY + XZ)...")

fig, (ax_xy, ax_xz) = plt.subplots(2, 1, figsize=(12, 16))

# === XY Projection (Top) ===
x = hybrid_traj[:, 0]
y = hybrid_traj[:, 1]
z = hybrid_traj[:, 2]

# Plot trajectory with gradient
n_points = len(x)
colors = plt.cm.plasma(np.linspace(0, 1, n_points))

for i in range(n_points - 1):
    ax_xy.plot(x[i:i+2], y[i:i+2],
               color=colors[i], linewidth=1.0, alpha=0.8)

# No end marker - trajectory speaks for itself

# Add Earth and Moon
ax_xy.scatter([earth_pos[0]], [earth_pos[1]],
              s=200, c='dodgerblue', marker='o',
              edgecolors='navy', linewidth=1.5,
              label='Earth', zorder=90)
ax_xy.scatter([moon_pos[0]], [moon_pos[1]],
              s=100, c='lightgray', marker='o',
              edgecolors='gray', linewidth=1.5,
              label='Moon', zorder=90)

ax_xy.set_xlabel('X', fontweight='bold', fontsize=20)
ax_xy.set_ylabel('Y', fontweight='bold', fontsize=20)
ax_xy.set_title('a) XY Projection (Top View)',
                fontweight='bold', fontsize=24, pad=15)
ax_xy.legend(framealpha=0.95, fontsize=18, loc='best')
ax_xy.grid(True, alpha=0.2, linestyle='--')
ax_xy.set_aspect('equal')

# === XZ Projection (Bottom) ===
for i in range(n_points - 1):
    ax_xz.plot(x[i:i+2], z[i:i+2],
               color=colors[i], linewidth=1.0, alpha=0.8)

# No end marker - trajectory speaks for itself

# Add Earth and Moon
ax_xz.scatter([earth_pos[0]], [earth_pos[2]],
              s=200, c='dodgerblue', marker='o',
              edgecolors='navy', linewidth=1.5,
              label='Earth', zorder=90)
ax_xz.scatter([moon_pos[0]], [moon_pos[2]],
              s=100, c='lightgray', marker='o',
              edgecolors='gray', linewidth=1.5,
              label='Moon', zorder=90)

ax_xz.set_xlabel('X', fontweight='bold', fontsize=20)
ax_xz.set_ylabel('Z', fontweight='bold', fontsize=20)
ax_xz.set_title('b) XZ Projection (Side View)',
                fontweight='bold', fontsize=24, pad=15)
ax_xz.legend(framealpha=0.95, fontsize=18, loc='upper left')
ax_xz.grid(True, alpha=0.2, linestyle='--')
ax_xz.set_aspect('equal')

# Ensure both subplots have the same X-axis range for consistent width
x_min, x_max = min(x.min(), x.min()), max(x.max(), x.max())
x_range = x_max - x_min
x_center = (x_max + x_min) / 2
# Add 10% padding
x_padding = x_range * 0.1
ax_xy.set_xlim(x_center - (x_range/2 + x_padding), x_center + (x_range/2 + x_padding))
ax_xz.set_xlim(x_center - (x_range/2 + x_padding), x_center + (x_range/2 + x_padding))

# Overall title
fig.suptitle('Hybrid Quantum-Classical Method - Trajectory Projections',
             fontweight='bold', fontsize=26, y=0.995)

plt.tight_layout()
output_file = OUTPUT_DIR / "hybrid_projections_clean.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Saved: {output_file}")
plt.close()

# ============================================================================
# PLOT 5: Side-by-side 3D Comparison
# ============================================================================
print("[*] Generating side-by-side 3D comparison...")

fig = plt.figure(figsize=(20, 10))

# Classical
ax1 = fig.add_subplot(121, projection='3d')

x_c = classical_traj[:, 0]
y_c = classical_traj[:, 1]
z_c = classical_traj[:, 2]

n_points = len(x_c)
colors = plt.cm.viridis(np.linspace(0, 1, n_points))

for i in range(n_points - 1):
    ax1.plot(x_c[i:i+2], y_c[i:i+2], z_c[i:i+2],
             color=colors[i], linewidth=1.0, alpha=0.8)

# Add Earth and Moon
earth_radius = 0.009
moon_radius = 0.008

u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
x_sphere = earth_radius * np.outer(np.cos(u), np.sin(v))
y_sphere = earth_radius * np.outer(np.sin(u), np.sin(v))
z_sphere = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

ax1.plot_surface(x_sphere + earth_pos[0], y_sphere + earth_pos[1],
                 z_sphere + earth_pos[2], color='dodgerblue', alpha=0.8)
ax1.plot_surface(x_sphere * (moon_radius/earth_radius) + moon_pos[0],
                 y_sphere * (moon_radius/earth_radius) + moon_pos[1],
                 z_sphere * (moon_radius/earth_radius) + moon_pos[2],
                 color='lightgray', alpha=0.8)

# No end marker - trajectory speaks for itself

ax1.set_xlabel('X', fontweight='bold', labelpad=12)
ax1.set_ylabel('Y', fontweight='bold', labelpad=12)
ax1.set_zlabel('Z', fontweight='bold', labelpad=12)
ax1.set_title('Classical Method', fontweight='bold', fontsize=24, pad=20)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.view_init(elev=25, azim=45)

ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False

# Hybrid
ax2 = fig.add_subplot(122, projection='3d')

x_h = hybrid_traj[:, 0]
y_h = hybrid_traj[:, 1]
z_h = hybrid_traj[:, 2]

n_points = len(x_h)
colors = plt.cm.plasma(np.linspace(0, 1, n_points))

for i in range(n_points - 1):
    ax2.plot(x_h[i:i+2], y_h[i:i+2], z_h[i:i+2],
             color=colors[i], linewidth=1.0, alpha=0.8)

# Add Earth and Moon
ax2.plot_surface(x_sphere + earth_pos[0], y_sphere + earth_pos[1],
                 z_sphere + earth_pos[2], color='dodgerblue', alpha=0.8)
ax2.plot_surface(x_sphere * (moon_radius/earth_radius) + moon_pos[0],
                 y_sphere * (moon_radius/earth_radius) + moon_pos[1],
                 z_sphere * (moon_radius/earth_radius) + moon_pos[2],
                 color='lightgray', alpha=0.8)

# No end marker - trajectory speaks for itself

ax2.set_xlabel('X', fontweight='bold', labelpad=12)
ax2.set_ylabel('Y', fontweight='bold', labelpad=12)
ax2.set_zlabel('Z', fontweight='bold', labelpad=12)
ax2.set_title('Hybrid Quantum-Classical Method', fontweight='bold', fontsize=24, pad=20)
ax2.grid(True, alpha=0.2, linestyle='--')
ax2.view_init(elev=25, azim=45)

ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False

plt.tight_layout()
output_file = OUTPUT_DIR / "comparison_3d_clean.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Saved: {output_file}")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nGenerated 5 clean trajectory plots:")
print("  - classical_3d_clean.png (3D with Earth & Moon)")
print("  - hybrid_3d_clean.png (3D with Earth & Moon)")
print("  - classical_projections_clean.png (XY + XZ combined)")
print("  - hybrid_projections_clean.png (XY + XZ combined)")
print("  - comparison_3d_clean.png (side-by-side 3D)")
print(f"\nAll saved to: {OUTPUT_DIR.absolute()}")
print("="*80)
