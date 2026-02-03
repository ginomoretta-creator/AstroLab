import { create } from 'zustand'

export type SimulationMethod = 'classical' | 'hybrid'
export type SimulationStatus = 'idle' | 'running' | 'completed' | 'error'

export interface Trajectory {
    points: [number, number][]  // Historical trajectories stored as 2D for memory efficiency
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
    apogeeAltitude: number
    perigeeAltitude: number
    dt: number
    numIterations: number
    // 3D Parameters (NEW)
    enable3D: boolean
    inclinationDeg: number
    raanDeg: number
    thrustMode: 'velocity_aligned' | 'orbital_plane'
}

export interface SimulationResult {
    iteration: number
    totalIterations: number
    dimension: 2 | 3  // NEW: Track if 2D or 3D
    trajectories: number[][][]
    bestTrajectory: number[][]
    bestCost: number
    bestSchedule?: number[]
    bestThrustFraction?: number
    bestDistance?: number
    method?: SimulationMethod
    metrics?: {
        deltaV?: number
        deltaV_numerical?: number
        fuelConsumed?: number
        timeOfFlight?: number
        costBreakdown?: {
            distance: number
            velocity: number
            fuel: number
            capture_bonus: number
        }
    }
}

export interface RunMetadata {
    run_id: string
    timestamp: string
    method: SimulationMethod
    params: SimulationParams
    finalCost?: number
    deltaV?: number
    timeOfFlight?: number
    captured?: boolean
    num_iterations: number
    filename?: string
}

export interface ComparisonResult {
    runs: Array<{
        run_id: string
        method: SimulationMethod
        timestamp: string
        delta_v?: number
        time_of_flight?: number
        fuel_consumed?: number
        final_cost?: number
        captured: boolean
        num_iterations: number
        delta_v_relative?: number
        time_relative?: number
        cost_relative?: number
    }>
    summary: {
        best_delta_v?: number
        best_time_of_flight?: number
        best_cost?: number
        num_runs_compared: number
    }
}

interface SimulationState {
    // Status
    status: SimulationStatus
    currentMethod: SimulationMethod
    backendStatus: 'checking' | 'online' | 'offline'
    backendPort: number
    errorMessage: string | null

    // Parameters
    params: SimulationParams

    // Results
    currentIteration: number
    results: {
        classical: SimulationResult | null
        hybrid: SimulationResult | null
    }

    // History
    trajectoryHistory: Trajectory[]
    iterationHistory: SimulationResult[]
    selectedResult: SimulationResult | null

    // Run Storage (NEW)
    savedRuns: RunMetadata[]
    selectedRuns: string[]
    comparisonData: ComparisonResult | null
    isLoadingRuns: boolean
    isSavingRun: boolean
    isExporting: boolean

    // Actions
    setStatus: (status: SimulationStatus) => void
    setMethod: (method: SimulationMethod) => void
    setBackendStatus: (status: 'checking' | 'online' | 'offline') => void
    setBackendPort: (port: number) => void
    setError: (message: string | null) => void
    updateParams: (params: Partial<SimulationParams>) => void
    addResult: (method: SimulationMethod, result: SimulationResult) => void
    addTrajectory: (trajectory: Trajectory) => void
    selectResult: (result: SimulationResult | null) => void
    clearResults: () => void
    reset: () => void

    // Run Storage Actions (NEW)
    setSavedRuns: (runs: RunMetadata[]) => void
    toggleRunSelection: (runId: string) => void
    clearRunSelection: () => void
    setComparisonData: (data: ComparisonResult | null) => void
    setIsLoadingRuns: (loading: boolean) => void
    setIsSavingRun: (saving: boolean) => void
    setIsExporting: (exporting: boolean) => void
}

const defaultParams: SimulationParams = {
    numSteps: 120000,      // ~250 days with dt=0.0005
    batchSize: 100,        // Increased for better exploration
    couplingStrength: 1.0,
    mass: 500,             // Smaller sat for realistic low-thrust
    thrust: 0.5,           // Typical Hall thruster
    isp: 3000,             // Hall thruster Isp
    apogeeAltitude: 400,   // Apogee altitude (km)
    perigeeAltitude: 200,  // Perigee altitude (km)
    dt: 0.0005,            // Very small steps for smooth trajectories
    numIterations: 30,     // More iterations for convergence
    // 3D Parameters
    enable3D: false,
    inclinationDeg: 0,
    raanDeg: 0,
    thrustMode: 'orbital_plane',
}

export const useSimulationStore = create<SimulationState>((set) => ({
    // Initial state
    status: 'idle',
    currentMethod: 'classical',
    backendStatus: 'checking',
    backendPort: 8080,
    errorMessage: null,
    params: defaultParams,
    currentIteration: 0,
    results: {
        classical: null,
        hybrid: null,
    },
    trajectoryHistory: [],
    iterationHistory: [],
    selectedResult: null,

    // Run Storage Initial State (NEW)
    savedRuns: [],
    selectedRuns: [],
    comparisonData: null,
    isLoadingRuns: false,
    isSavingRun: false,
    isExporting: false,

    // Actions
    setStatus: (status) => set({ status }),
    setMethod: (method) => set({ currentMethod: method }),
    setBackendStatus: (status) => set({ backendStatus: status }),
    setBackendPort: (port) => set({ backendPort: port }),
    setError: (message) => set({ errorMessage: message }),

    updateParams: (newParams) => set((state) => ({
        params: { ...state.params, ...newParams }
    })),

    addResult: (method, result) => set((state) => {
        // Add to full iteration history if it's a new iteration
        // We use the iteration number to avoid duplicates
        const existingIdx = state.iterationHistory.findIndex(r => r.iteration === result.iteration)
        let newIterationHistory = [...state.iterationHistory]

        if (existingIdx >= 0) {
            // Update existing
            newIterationHistory[existingIdx] = result
        } else {
            newIterationHistory.push(result)
        }
        // Sort by iteration
        newIterationHistory.sort((a, b) => a.iteration - b.iteration)

        return {
            results: { ...state.results, [method]: result },
            currentIteration: result.iteration,
            iterationHistory: newIterationHistory
        }
    }),

    selectResult: (result) => set({ selectedResult: result }),

    addTrajectory: (trajectory) => set((state) => ({
        trajectoryHistory: [...state.trajectoryHistory.slice(-50), trajectory]
    })),

    clearResults: () => set({
        results: { classical: null, hybrid: null },
        trajectoryHistory: [],
        currentIteration: 0,
        errorMessage: null,
    }),

    reset: () => set({
        status: 'idle',
        params: defaultParams,
        currentIteration: 0,
        results: { classical: null, hybrid: null },
        trajectoryHistory: [],
        errorMessage: null,
    }),

    // Run Storage Actions (NEW)
    setSavedRuns: (runs) => set({ savedRuns: runs }),

    toggleRunSelection: (runId) => set((state) => {
        const isSelected = state.selectedRuns.includes(runId)
        return {
            selectedRuns: isSelected
                ? state.selectedRuns.filter(id => id !== runId)
                : [...state.selectedRuns, runId]
        }
    }),

    clearRunSelection: () => set({ selectedRuns: [] }),

    setComparisonData: (data) => set({ comparisonData: data }),

    setIsLoadingRuns: (loading) => set({ isLoadingRuns: loading }),

    setIsSavingRun: (saving) => set({ isSavingRun: saving }),

    setIsExporting: (exporting) => set({ isExporting: exporting }),
}))
