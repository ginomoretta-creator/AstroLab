import { useSimulationStore, SimulationMethod } from '../stores/simulationStore'

const API_TIMEOUT = 30000

export interface SimulationRequest {
    method: SimulationMethod
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
    // 3D Parameters
    enable3D: boolean
    inclinationDeg: number
    raanDeg: number
    thrustMode: 'velocity_aligned' | 'orbital_plane'
}

export interface StreamedResult {
    iteration: number
    totalIterations: number
    dimension?: 2 | 3  // NEW: Track if 2D or 3D
    trajectories: number[][][]
    bestTrajectory: number[][]
    bestCost: number
    meanCost: number
    successRate: number
    bestSchedule?: number[]
    bestThrustFraction?: number
    bestDistance?: number
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

class BackendAPI {
    private baseUrl: string = 'http://127.0.0.1:8080'
    private abortController: AbortController | null = null

    setPort(port: number) {
        this.baseUrl = `http://127.0.0.1:${port}`
    }

    async healthCheck(): Promise<boolean> {
        try {
            const response = await fetch(`${this.baseUrl}/`, {
                signal: AbortSignal.timeout(5000)
            })
            return response.ok
        } catch {
            return false
        }
    }

    async startSimulation(
        request: SimulationRequest,
        onProgress: (result: StreamedResult) => void,
        onComplete: () => void,
        onError: (error: string) => void
    ): Promise<void> {
        // Cancel any existing simulation
        this.cancelSimulation()

        this.abortController = new AbortController()

        try {
            const response = await fetch(`${this.baseUrl}/simulate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    num_steps: request.numSteps,
                    batch_size: request.batchSize,
                    coupling_strength: request.couplingStrength,
                    mass: request.mass,
                    thrust: request.thrust,
                    isp: request.isp,
                    apogee_altitude: request.apogeeAltitude,
                    perigee_altitude: request.perigeeAltitude,
                    dt: request.dt,
                    num_iterations: request.numIterations,
                    method: request.method,
                    // 3D Parameters
                    enable_3d: request.enable3D,
                    inclination_deg: request.inclinationDeg,
                    raan_deg: request.raanDeg,
                    thrust_mode: request.thrustMode,
                }),
                signal: this.abortController.signal,
            })

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`)
            }

            // Handle streaming response
            const reader = response.body?.getReader()
            if (!reader) {
                throw new Error('No response body')
            }

            const decoder = new TextDecoder()
            let buffer = ''

            while (true) {
                const { done, value } = await reader.read()

                if (done) {
                    onComplete()
                    break
                }

                buffer += decoder.decode(value, { stream: true })

                // Parse JSON lines
                const lines = buffer.split('\n')
                buffer = lines.pop() || '' // Keep incomplete line in buffer

                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            const data = JSON.parse(line)

                            // Check for error messages from backend
                            if (data.error) {
                                console.error('[Backend Error]', data.error)
                                if (data.traceback) console.error(data.traceback)
                                onError(data.error)
                                return
                            }

                            // Validate required fields
                            if (typeof data.iteration !== 'number') {
                                console.warn('[Validation] Skipping line - no iteration number:', data)
                                continue
                            }

                            // Deep validation helper
                            const validateTrajectory = (traj: any[]): number[][] => {
                                if (!Array.isArray(traj)) return []
                                return traj.filter(p =>
                                    Array.isArray(p) &&
                                    p.length >= 2 &&
                                    p.every(val => typeof val === 'number' && Number.isFinite(val))
                                )
                            }

                            // Downsample helper to save memory (max 10000 points)
                            const downsample = (points: number[][], target: number = 10000): number[][] => {
                                if (points.length <= target) return points
                                const step = Math.ceil(points.length / target)
                                return points.filter((_, i) => i % step === 0 || i === points.length - 1)
                            }

                            const safeTrajectories = Array.isArray(data.trajectories)
                                ? data.trajectories.map(validateTrajectory).filter((t: any[]) => t.length > 0).map((t: number[][]) => downsample(t))
                                : []

                            const safeBestTrajectory = validateTrajectory(data.best_trajectory)
                            const downsampledBest = downsample(safeBestTrajectory)

                            console.log(`[${data.method?.toUpperCase()}] Iter ${data.iteration}/${data.total_iterations}: ${safeTrajectories.length} trajs, best=${downsampledBest.length} pts, dist=${data.best_distance?.toFixed(4)}`)

                            onProgress({
                                iteration: data.iteration || 0,
                                totalIterations: data.total_iterations || request.numIterations,
                                dimension: data.dimension || 2,  // NEW: Track dimension
                                trajectories: safeTrajectories,
                                bestTrajectory: downsampledBest,
                                bestCost: typeof data.best_cost === 'number' ? data.best_cost : Infinity,
                                meanCost: typeof data.mean_cost === 'number' ? data.mean_cost : Infinity,
                                successRate: typeof data.success_rate === 'number' ? data.success_rate : 0,
                                bestSchedule: Array.isArray(data.best_schedule) ? data.best_schedule.map((v: any) => Number(v) || 0) : undefined,
                                bestThrustFraction: typeof data.best_thrust_fraction === 'number' ? data.best_thrust_fraction : undefined,
                                bestDistance: typeof data.best_distance === 'number' ? data.best_distance : undefined,
                                metrics: data.metrics ? {
                                    deltaV: data.metrics.deltaV,
                                    deltaV_numerical: data.metrics.deltaV_numerical,
                                    fuelConsumed: data.metrics.fuelConsumed,
                                    timeOfFlight: data.metrics.timeOfFlight,
                                    costBreakdown: data.metrics.costBreakdown
                                } : undefined,
                            })
                        } catch (e) {
                            console.warn('Failed to parse line:', line.substring(0, 100) + '...')
                        }
                    }
                }
            }
        } catch (error: any) {
            if (error.name === 'AbortError') {
                console.log('Simulation cancelled')
            } else {
                onError(error.message || 'Unknown error')
            }
        }
    }

    cancelSimulation() {
        if (this.abortController) {
            this.abortController.abort()
            this.abortController = null
        }
    }

    async runBenchmark(
        methods: SimulationMethod[],
        numSamples: number,
        numSteps: number
    ): Promise<Record<string, any>> {
        const response = await fetch(`${this.baseUrl}/benchmark`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                methods,
                num_samples: numSamples,
                num_steps: numSteps,
            }),
            signal: AbortSignal.timeout(API_TIMEOUT * 2),
        })

        if (!response.ok) {
            throw new Error(`Benchmark failed: ${response.statusText}`)
        }

        return response.json()
    }
}

export const backendAPI = new BackendAPI()

// Hook for using the API in components
export function useBackend() {
    const {
        params,
        currentMethod,
        setStatus,
        addResult,
        addTrajectory,
        backendPort,
        setBackendStatus,
        setError,
    } = useSimulationStore()

    const startSimulation = async () => {
        backendAPI.setPort(backendPort)
        setError(null)
        setStatus('running')

        const request: SimulationRequest = {
            method: currentMethod,
            numSteps: params.numSteps,
            batchSize: params.batchSize,
            couplingStrength: params.couplingStrength,
            mass: params.mass,
            thrust: params.thrust,
            isp: params.isp,
            apogeeAltitude: params.apogeeAltitude,
            perigeeAltitude: params.perigeeAltitude,
            dt: params.dt,
            numIterations: params.numIterations,
            // 3D Parameters
            enable3D: params.enable3D,
            inclinationDeg: params.inclinationDeg,
            raanDeg: params.raanDeg,
            thrustMode: params.thrustMode,
        }

        await backendAPI.startSimulation(
            request,
            (result) => {
                // Add best trajectory for visualization
                // It is already downsampled in onProgress
                // Map to [x, y] for 2D compatibility with Trajectory type
                // Note: Full trajectory with z is in result itself
                if (result.bestTrajectory.length > 0) {
                    addTrajectory({
                        points: result.bestTrajectory.map(p => [p[0], p[1]] as [number, number]),
                        cost: result.bestCost,
                        method: currentMethod,
                    })
                }

                // Update result and history
                addResult(currentMethod, {
                    iteration: result.iteration,
                    totalIterations: result.totalIterations,
                    dimension: result.dimension || 2,  // NEW
                    trajectories: result.trajectories,
                    bestTrajectory: result.bestTrajectory,
                    bestCost: result.bestCost,
                    bestSchedule: result.bestSchedule,
                    bestThrustFraction: result.bestThrustFraction,
                    bestDistance: result.bestDistance,
                    method: currentMethod,
                    metrics: result.metrics
                })
            },
            () => {
                setStatus('completed')
            },
            (error) => {
                console.error('Simulation error:', error)
                setStatus('error')
                setBackendStatus('offline')
                setError(error || 'Backend unreachable. Make sure the physics backend is running.')
            }
        )
    }

    const stopSimulation = () => {
        backendAPI.cancelSimulation()
        setStatus('idle')
    }

    const checkHealth = async () => {
        backendAPI.setPort(backendPort)
        return backendAPI.healthCheck()
    }

    return {
        startSimulation,
        stopSimulation,
        checkHealth,
    }
}
