# React Frontend Component Plan

## Overview

This document outlines the component hierarchy and architecture for the React + Three.js frontend.

**Philosophy**: Start with 2D Canvas visualization (simpler), then upgrade to Three.js once data pipeline is proven.

---

## Phase 1: Basic 2D Visualization (MVP)

### Component Hierarchy

```
<App>
├── <SimulationProvider>          # Context for simulation state
│   ├── <ControlPanel>             # Left sidebar
│   │   ├── <MethodSelector>       # THRML vs Classical
│   │   ├── <SpacecraftControls>   # Mass, Thrust, ISP
│   │   ├── <OrbitControls>        # Initial altitude
│   │   └── <SolverControls>       # Coupling, iterations, batch
│   ├── <VisualizationCanvas>      # Main 2D canvas
│   │   ├── <EarthMoon>            # Static bodies
│   │   ├── <TrajectoryLines>      # Ghost + best trajectories
│   │   └── <Legend>               # Color coding explanation
│   └── <StatusBar>                # Bottom metrics
│       ├── <IterationCounter>     # Current iteration
│       ├── <CostMetric>           # Best distance
│       └── <ProgressBar>          # Overall progress
```

### File Structure

```
frontend/src/
├── App.jsx                    # Main app shell
├── main.jsx                   # Entry point
├── index.css                  # Global styles
│
├── contexts/
│   └── SimulationContext.jsx  # State management
│
├── hooks/
│   ├── useSimulation.js       # API streaming hook
│   └── useAnimation.js        # requestAnimationFrame helper
│
├── components/
│   ├── ControlPanel/
│   │   ├── ControlPanel.jsx
│   │   ├── MethodSelector.jsx
│   │   ├── SpacecraftControls.jsx
│   │   ├── OrbitControls.jsx
│   │   └── SolverControls.jsx
│   │
│   ├── Visualization/
│   │   ├── VisualizationCanvas.jsx
│   │   ├── EarthMoon.jsx
│   │   ├── TrajectoryLines.jsx
│   │   └── Legend.jsx
│   │
│   └── StatusBar/
│       ├── StatusBar.jsx
│       ├── IterationCounter.jsx
│       ├── CostMetric.jsx
│       └── ProgressBar.jsx
│
└── utils/
    ├── api.js                 # fetch helpers
    ├── physics.js             # Unit conversions
    └── colors.js              # Theme palette
```

---

## Phase 2: Three.js Upgrade (Production)

### Additional Three.js Components

```
<VisualizationCanvas>
├── <Canvas>                   # react-three-fiber
│   ├── <Camera>               # Perspective camera
│   ├── <Lights>               # Ambient + point lights
│   ├── <Earth>                # Sphere with shader
│   ├── <Moon>                 # Glowing sphere
│   ├── <TrajectoryMeshes>     # Lines with bloom
│   └── <OrbitControls>        # Camera rotation
```

### Key Libraries
```json
{
  "three": "^0.160.0",
  "@react-three/fiber": "^8.15.0",
  "@react-three/drei": "^9.92.0",
  "@react-three/postprocessing": "^2.15.0"
}
```

---

## State Management

### SimulationContext

```javascript
const SimulationContext = createContext({
  // State
  status: 'idle' | 'simulating' | 'complete' | 'error',
  currentIteration: 0,
  totalIterations: 0,
  trajectories: [],
  bestTrajectory: null,
  bestCost: Infinity,
  
  // Parameters
  parameters: {
    numSteps: 5000,
    batchSize: 50,
    couplingStrength: 0.5,
    mass: 1000,
    thrust: 10,
    isp: 300,
    initialAltitude: 400,
    method: 'thrml',
    dt: 0.01,
    numIterations: 50
  },
  
  // Actions
  updateParameter: (key, value) => {},
  startSimulation: () => {},
  stopSimulation: () => {},
  reset: () => {}
});
```

### useSimulation Hook

```javascript
function useSimulation() {
  const [state, setState] = useState({
    status: 'idle',
    data: null
  });
  
  const startSimulation = async (parameters) => {
    setState({ status: 'simulating', data: null });
    
    const response = await fetch('http://localhost:8000/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(parameters)
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(Boolean);
      
      for (const line of lines) {
        const data = JSON.parse(line);
        setState(prev => ({
          ...prev,
          data: data
        }));
      }
    }
    
    setState(prev => ({ ...prev, status: 'complete' }));
  };
  
  return { state, startSimulation };
}
```

---

## Styling Strategy

### Theme ("Deep Void")

```css
:root {
  /* Colors */
  --bg-void: #0a0e1a;
  --bg-surface: #141824;
  --text-primary: #e2e8f0;
  --text-secondary: #94a3b8;
  
  --accent-blue: #3b82f6;
  --accent-orange: #f97316;
  --accent-green: #10b981;
  
  /* Typography */
  --font-header: 'Michroma', sans-serif;
  --font-body: 'Barlow', sans-serif;
  
  /* Spacing */
  --spacing-xs: 0.5rem;
  --spacing-sm: 1rem;
  --spacing-md: 1.5rem;
  --spacing-lg: 2rem;
}
```

### Component Example

```jsx
// ControlPanel.jsx
import { motion } from 'framer-motion';

export function ControlPanel({ children }) {
  return (
    <motion.aside
      initial={{ x: -300, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="control-panel"
    >
      <h2 className="panel-title">Mission Parameters</h2>
      {children}
    </motion.aside>
  );
}
```

```css
/* ControlPanel.module.css */
.control-panel {
  background: var(--bg-surface);
  border-right: 1px solid rgba(59, 130, 246, 0.2);
  padding: var(--spacing-lg);
  width: 300px;
  height: 100vh;
  overflow-y: auto;
}

.panel-title {
  font-family: var(--font-header);
  font-size: 1.25rem;
  color: var(--accent-blue);
  margin-bottom: var(--spacing-md);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}
```

---

## Canvas Rendering (2D)

```jsx
// VisualizationCanvas.jsx
import { useEffect, useRef } from 'react';

export function VisualizationCanvas({ trajectories, bestTrajectory }) {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    
    // Clear
    ctx.fillStyle = '#0a0e1a';
    ctx.fillRect(0, 0, width, height);
    
    // Constants
    const L_STAR = 384400; // km
    const earthX = -0.01215 * L_STAR;
    const moonX = 0.98785 * L_STAR;
    const scale = width / (2 * L_STAR);
    
    const toCanvas = (x, y) => ({
      x: width / 2 + x * scale,
      y: height / 2 - y * scale
    });
    
    // Draw Earth
    const earthPos = toCanvas(earthX, 0);
    ctx.beginPath();
    ctx.arc(earthPos.x, earthPos.y, 20, 0, 2 * Math.PI);
    ctx.fillStyle = '#3b82f6';
    ctx.fill();
    
    // Draw Moon
    const moonPos = toCanvas(moonX, 0);
    ctx.beginPath();
    ctx.arc(moonPos.x, moonPos.y, 8, 0, 2 * Math.PI);
    ctx.fillStyle = '#94a3b8';
    ctx.fill();
    
    // Draw ghost trajectories
    trajectories.forEach(traj => {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(100, 116, 139, 0.1)';
      ctx.lineWidth = 1;
      
      traj.forEach(([x, y], i) => {
        const pos = toCanvas(x * L_STAR, y * L_STAR);
        i === 0 ? ctx.moveTo(pos.x, pos.y) : ctx.lineTo(pos.x, pos.y);
      });
      
      ctx.stroke();
    });
    
    // Draw best trajectory
    if (bestTrajectory) {
      ctx.beginPath();
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 3;
      ctx.shadowBlur = 10;
      ctx.shadowColor = '#3b82f6';
      
      bestTrajectory.forEach(([x, y], i) => {
        const pos = toCanvas(x * L_STAR, y * L_STAR);
        i === 0 ? ctx.moveTo(pos.x, pos.y) : ctx.lineTo(pos.x, pos.y);
      });
      
      ctx.stroke();
    }
    
  }, [trajectories, bestTrajectory]);
  
  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={800}
      style={{ border: '1px solid rgba(59, 130, 246, 0.2)' }}
    />
  );
}
```

---

## Animation Strategy

```javascript
// useAnimation.js
import { useEffect, useRef } from 'react';

export function useAnimation(callback) {
  const requestRef = useRef();
  const previousTimeRef = useRef();
  
  useEffect(() => {
    const animate = (time) => {
      if (previousTimeRef.current !== undefined) {
        const deltaTime = time - previousTimeRef.current;
        callback(deltaTime);
      }
      previousTimeRef.current = time;
      requestRef.requestRef = requestAnimationFrame(animate);
    };
    
    requestRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(requestRef.current);
  }, [callback]);
}
```

---

## Next Steps (Implementation Order)

1. ✅ Vite + React initialized
2. [ ] Create `SimulationContext.jsx`
3. [ ] Implement `useSimulation.js` hook
4. [ ] Build `ControlPanel` components
5. [ ] Implement 2D `VisualizationCanvas`
6. [ ] Add `StatusBar` metrics
7. [ ] Test with backend streaming
8. [ ] Polish UI/UX (animations, responsiveness)
9. [ ] (Future) Upgrade to Three.js

---

## Testing Strategy

```jsx
// ControlPanel.test.jsx
import { render, screen, fireEvent } from '@testing-library/react';
import { ControlPanel } from './ControlPanel';

test('updates thrust parameter', () => {
  const onUpdate = jest.fn();
  render(<ControlPanel onUpdate={onUpdate} />);
  
  const thrustInput = screen.getByLabelText(/thrust/i);
  fireEvent.change(thrustInput, { target: { value: '20' } });
  
  expect(onUpdate).toHaveBeenCalledWith('thrust', 20);
});
```

---

## Performance Considerations

1. **Debounce parameter updates**: Don't trigger simulation on every keystroke
2. **Virtualize trajectory rendering**: Only render visible subset
3. **Web Workers**: Offload trajectory parsing to background thread
4. **Memoization**: Use `useMemo` for expensive computations
5. **Canvas pooling**: Reuse canvas elements instead of recreating

---

This plan provides a clear path from current state (Vite initialized) to a production-ready React frontend with real-time 3D visualization.
