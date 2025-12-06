import React, { useState } from 'react';
import { SimulationCanvas } from './components/SimulationCanvas';
import { ControlPanel } from './components/ControlPanel';
import { StatsOverlay } from './components/StatsOverlay';
import { useSimulation } from './hooks/useSimulation';

function App() {
  const [params, setParams] = useState({
    num_steps: 5000,
    batch_size: 50,
    coupling_strength: 0.5,
    mass: 1000.0,
    thrust: 10.0,
    isp: 300.0,
    initial_altitude: 400.0,
    method: 'thrml',
    dt: 0.01,
    num_iterations: 50
  });

  const {
    startSimulation,
    stopSimulation,
    isSimulating,
    trajectories,
    bestTrajectory,
    metrics
  } = useSimulation();

  const handleSimulate = () => {
    if (isSimulating) {
      stopSimulation();
    } else {
      startSimulation(params);
    }
  };

  return (
    <div className="relative w-full h-screen overflow-hidden bg-void-900 text-white">
      {/* Background Elements */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-void-800 via-void-900 to-black opacity-80 pointer-events-none" />

      {/* 3D Scene */}
      <SimulationCanvas
        trajectories={trajectories}
        bestTrajectory={bestTrajectory}
      />

      {/* UI Layer */}
      <ControlPanel
        params={params}
        setParams={setParams}
        onSimulate={handleSimulate}
        isSimulating={isSimulating}
      />

      <StatsOverlay metrics={metrics} />

      {/* Footer / Branding */}
      <div className="absolute bottom-4 left-0 right-0 text-center pointer-events-none">
        <p className="text-[10px] text-gray-600 tracking-[0.2em] uppercase">
          Cislunar Trajectory Sandbox • v2.0.0 • Powered by JAX + THRML
        </p>
      </div>
    </div>
  );
}

export default App;
