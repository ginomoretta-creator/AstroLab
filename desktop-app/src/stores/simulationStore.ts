import { create } from 'zustand'

export type SimulationMethod = 'thrml' | 'quantum' | 'random'
export type SimulationStatus = 'idle' | 'running' | 'completed' | 'error'

export interface Trajectory {
    points: [number, number][]
    cost: number
    method: SimulationMethod
}

export interface SimulationParams {
    numSteps: number
    batchSize: number
    couplingStrength: number
    mass: number
    thrust: number
    isp: number
    initialAltitude: number
    dt: number
    numIterations: number
}

export interface SimulationResult {
    iteration: number
    totalIterations: number
    trajectories: number[][][]
    bestTrajectory: number[][]
    bestCost: number
}

interface SimulationState {
    // Status
    status: SimulationStatus
    currentMethod: SimulationMethod
    backendStatus: 'checking' | 'online' | 'offline'
    backendPort: number

    // Parameters
    params: SimulationParams

    // Results
    currentIteration: number
    results: {
        thrml: SimulationResult | null
        quantum: SimulationResult | null
        random: SimulationResult | null
    }

    // History for visualization
    trajectoryHistory: Trajectory[]

    // Actions
    setStatus: (status: SimulationStatus) => void
    setMethod: (method: SimulationMethod) => void
    setBackendStatus: (status: 'checking' | 'online' | 'offline') => void
    setBackendPort: (port: number) => void
    updateParams: (params: Partial<SimulationParams>) => void
    setIteration: (iteration: number) => void
    setResult: (method: SimulationMethod, result: SimulationResult) => void
    addTrajectory: (trajectory: Trajectory) => void
    clearResults: () => void
    reset: () => void
}

const defaultParams: SimulationParams = {
    numSteps: 500,
    batchSize: 50,
    couplingStrength: 1.0,
    mass: 500,
    thrust: 0.5,
    isp: 3000,
    initialAltitude: 200,
    dt: 0.01,
    numIterations: 20,
}

export const useSimulationStore = create<SimulationState>((set) => ({
    // Initial state
    status: 'idle',
    currentMethod: 'thrml',
    backendStatus: 'checking',
    backendPort: 8080,
    params: defaultParams,
    currentIteration: 0,
    results: {
        thrml: null,
        quantum: null,
        random: null,
    },
    trajectoryHistory: [],

    // Actions
    setStatus: (status) => set({ status }),
    setMethod: (method) => set({ currentMethod: method }),
    setBackendStatus: (status) => set({ backendStatus: status }),
    setBackendPort: (port) => set({ backendPort: port }),

    updateParams: (newParams) => set((state) => ({
        params: { ...state.params, ...newParams }
    })),

    setIteration: (iteration) => set({ currentIteration: iteration }),

    setResult: (method, result) => set((state) => ({
        results: { ...state.results, [method]: result }
    })),

    addTrajectory: (trajectory) => set((state) => ({
        trajectoryHistory: [...state.trajectoryHistory.slice(-50), trajectory]
    })),

    clearResults: () => set({
        results: { thrml: null, quantum: null, random: null },
        trajectoryHistory: [],
        currentIteration: 0,
    }),

    reset: () => set({
        status: 'idle',
        params: defaultParams,
        currentIteration: 0,
        results: { thrml: null, quantum: null, random: null },
        trajectoryHistory: [],
    }),
}))
