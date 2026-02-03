/**
 * Run Storage Hook - API Integration for Run Management
 *
 * Provides methods for:
 * - Saving simulation runs
 * - Loading saved runs
 * - Deleting runs
 * - Comparing multiple runs
 * - Exporting data (CSV, JSON, MATLAB)
 */

import { useSimulationStore, RunMetadata, ComparisonResult, SimulationParams } from '../stores/simulationStore'

export interface SaveRunData {
    method: string
    params: SimulationParams
    iterations: any[]
    finalMetrics: any
    best_trajectory: number[][]
    best_schedule: number[]
}

export function useRunStorage() {
    const {
        backendPort,
        savedRuns,
        selectedRuns,
        iterationHistory,
        results,
        currentMethod,
        params,
        setSavedRuns,
        setComparisonData,
        setIsLoadingRuns,
        setIsSavingRun,
        setIsExporting,
        setError,
    } = useSimulationStore()

    const baseUrl = `http://127.0.0.1:${backendPort}`

    // =============================================================================
    // Load Saved Runs
    // =============================================================================

    const loadRuns = async (method?: string, limit: number = 50) => {
        setIsLoadingRuns(true)
        setError(null)

        try {
            const url = new URL(`${baseUrl}/runs/list`)
            if (method) url.searchParams.append('method', method)
            url.searchParams.append('limit', limit.toString())

            const response = await fetch(url.toString(), {
                signal: AbortSignal.timeout(10000)
            })

            if (!response.ok) {
                throw new Error(`Failed to load runs: ${response.statusText}`)
            }

            const data = await response.json()
            const runs: RunMetadata[] = data.runs || []

            setSavedRuns(runs)
            console.log(`[RUN STORAGE] Loaded ${runs.length} runs`)

            return runs
        } catch (error: any) {
            console.error('[RUN STORAGE] Failed to load runs:', error)
            setError(error.message || 'Failed to load saved runs')
            return []
        } finally {
            setIsLoadingRuns(false)
        }
    }

    // =============================================================================
    // Save Current Run
    // =============================================================================

    const saveCurrentRun = async () => {
        // Validate that we have completed results
        const currentResult = results[currentMethod]
        if (!currentResult || iterationHistory.length === 0) {
            setError('No completed simulation to save')
            return null
        }

        setIsSavingRun(true)
        setError(null)

        try {
            // Extract final metrics from last iteration
            const lastIteration = iterationHistory[iterationHistory.length - 1]
            const finalMetrics = {
                final_cost: lastIteration.bestCost,
                delta_v_analytical: lastIteration.metrics?.deltaV,
                delta_v_numerical: lastIteration.metrics?.deltaV_numerical,
                fuel_consumed: lastIteration.metrics?.fuelConsumed,
                total_time_days: lastIteration.metrics?.timeOfFlight,
                captured: false, // Will be calculated on backend
                num_iterations: iterationHistory.length,
                convergence_rate: 0, // Will be calculated on backend
                cost_improvement: iterationHistory[0].bestCost - lastIteration.bestCost
            }

            // Prepare run data
            const runData: SaveRunData = {
                method: currentMethod,
                params: params,
                iterations: iterationHistory.map(iter => ({
                    iteration: iter.iteration,
                    bestCost: iter.bestCost,
                    bestDistance: iter.bestDistance,
                    bestThrustFraction: iter.bestThrustFraction,
                    metrics: iter.metrics
                })),
                finalMetrics: finalMetrics,
                best_trajectory: lastIteration.bestTrajectory,
                best_schedule: lastIteration.bestSchedule || []
            }

            const response = await fetch(`${baseUrl}/runs/save`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(runData),
                signal: AbortSignal.timeout(30000)
            })

            if (!response.ok) {
                throw new Error(`Failed to save run: ${response.statusText}`)
            }

            const result = await response.json()
            console.log(`[RUN STORAGE] Run saved with ID: ${result.run_id}`)

            // Reload runs to include the new one
            await loadRuns()

            return result.run_id
        } catch (error: any) {
            console.error('[RUN STORAGE] Failed to save run:', error)
            setError(error.message || 'Failed to save run')
            return null
        } finally {
            setIsSavingRun(false)
        }
    }

    // =============================================================================
    // Load Specific Run
    // =============================================================================

    const loadRun = async (runId: string) => {
        setError(null)

        try {
            const response = await fetch(`${baseUrl}/runs/${runId}`, {
                signal: AbortSignal.timeout(10000)
            })

            if (!response.ok) {
                throw new Error(`Failed to load run ${runId}: ${response.statusText}`)
            }

            const runData = await response.json()
            console.log(`[RUN STORAGE] Loaded run ${runId}`)

            return runData
        } catch (error: any) {
            console.error(`[RUN STORAGE] Failed to load run ${runId}:`, error)
            setError(error.message || `Failed to load run ${runId}`)
            return null
        }
    }

    // =============================================================================
    // Delete Run
    // =============================================================================

    const deleteRun = async (runId: string) => {
        setError(null)

        try {
            const response = await fetch(`${baseUrl}/runs/${runId}`, {
                method: 'DELETE',
                signal: AbortSignal.timeout(10000)
            })

            if (!response.ok) {
                throw new Error(`Failed to delete run ${runId}: ${response.statusText}`)
            }

            console.log(`[RUN STORAGE] Deleted run ${runId}`)

            // Reload runs
            await loadRuns()

            return true
        } catch (error: any) {
            console.error(`[RUN STORAGE] Failed to delete run ${runId}:`, error)
            setError(error.message || `Failed to delete run ${runId}`)
            return false
        }
    }

    // =============================================================================
    // Compare Runs
    // =============================================================================

    const compareRuns = async (runIds?: string[]) => {
        const idsToCompare = runIds || selectedRuns

        if (idsToCompare.length < 2) {
            setError('Select at least 2 runs to compare')
            return null
        }

        setError(null)

        try {
            const response = await fetch(`${baseUrl}/runs/compare`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ run_ids: idsToCompare }),
                signal: AbortSignal.timeout(15000)
            })

            if (!response.ok) {
                throw new Error(`Failed to compare runs: ${response.statusText}`)
            }

            const comparisonData: ComparisonResult = await response.json()
            console.log(`[RUN STORAGE] Compared ${idsToCompare.length} runs`)

            setComparisonData(comparisonData)

            return comparisonData
        } catch (error: any) {
            console.error('[RUN STORAGE] Failed to compare runs:', error)
            setError(error.message || 'Failed to compare runs')
            return null
        }
    }

    // =============================================================================
    // Export Data
    // =============================================================================

    const exportRuns = async (format: 'csv' | 'json' | 'matlab', runIds?: string[]) => {
        const idsToExport = runIds || selectedRuns

        if (idsToExport.length === 0) {
            setError('Select at least 1 run to export')
            return false
        }

        setIsExporting(true)
        setError(null)

        try {
            const response = await fetch(`${baseUrl}/export/${format}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ run_ids: idsToExport }),
                signal: AbortSignal.timeout(30000)
            })

            if (!response.ok) {
                throw new Error(`Failed to export data: ${response.statusText}`)
            }

            // Get filename from Content-Disposition header or use default
            const contentDisposition = response.headers.get('Content-Disposition')
            let filename = `runs_export.${format}`
            if (contentDisposition) {
                const match = contentDisposition.match(/filename="?(.+)"?/)
                if (match) filename = match[1]
            }

            // Download file
            const blob = await response.blob()
            const url = window.URL.createObjectURL(blob)
            const link = document.createElement('a')
            link.href = url
            link.download = filename
            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)
            window.URL.revokeObjectURL(url)

            console.log(`[RUN STORAGE] Exported ${idsToExport.length} runs as ${format}`)

            return true
        } catch (error: any) {
            console.error(`[RUN STORAGE] Failed to export as ${format}:`, error)
            setError(error.message || `Failed to export as ${format}`)
            return false
        } finally {
            setIsExporting(false)
        }
    }

    // =============================================================================
    // Storage Stats
    // =============================================================================

    const getStorageStats = async () => {
        setError(null)

        try {
            const response = await fetch(`${baseUrl}/runs/storage/stats`, {
                signal: AbortSignal.timeout(5000)
            })

            if (!response.ok) {
                throw new Error(`Failed to get storage stats: ${response.statusText}`)
            }

            const stats = await response.json()
            console.log('[RUN STORAGE] Storage stats:', stats)

            return stats
        } catch (error: any) {
            console.error('[RUN STORAGE] Failed to get storage stats:', error)
            return null
        }
    }

    // =============================================================================
    // Return Hook API
    // =============================================================================

    return {
        // State
        savedRuns,
        selectedRuns,

        // Actions
        loadRuns,
        saveCurrentRun,
        loadRun,
        deleteRun,
        compareRuns,
        exportRuns,
        getStorageStats,
    }
}
