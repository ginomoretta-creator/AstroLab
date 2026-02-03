/**
 * Comparison Dashboard Component
 *
 * Side-by-side comparison of multiple runs:
 * - Metric comparison table
 * - Convergence plots
 * - Statistical summary
 * - Relative performance indicators
 */

import { useEffect } from 'react'
import { TrendingDown, Zap, Clock, Target, Award } from 'lucide-react'
import { useSimulationStore } from '../../stores/simulationStore'
import { useRunStorage } from '../../hooks/useRunStorage'

export default function ComparisonDashboard() {
    const { selectedRuns, comparisonData } = useSimulationStore()
    const { compareRuns } = useRunStorage()

    // Load comparison when selected runs change
    useEffect(() => {
        if (selectedRuns.length >= 2) {
            compareRuns()
        }
    }, [selectedRuns])

    if (selectedRuns.length < 2) {
        return (
            <div className="h-full flex items-center justify-center bg-theme-primary">
                <div className="text-center space-y-3">
                    <TrendingDown className="w-12 h-12 text-theme-muted mx-auto" />
                    <div className="text-sm text-theme-muted">
                        Select at least 2 runs from the Run Manager to compare
                    </div>
                </div>
            </div>
        )
    }

    if (!comparisonData) {
        return (
            <div className="h-full flex items-center justify-center bg-theme-primary">
                <div className="text-sm text-theme-muted">Loading comparison...</div>
            </div>
        )
    }

    const { runs, summary } = comparisonData

    return (
        <div className="h-full overflow-y-auto bg-theme-primary p-4">
            <div className="max-w-6xl mx-auto space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <div>
                        <h3 className="text-lg font-bold text-theme-primary">Run Comparison</h3>
                        <p className="text-xs text-theme-muted mt-1">
                            Comparing {runs.length} runs • Best performers highlighted
                        </p>
                    </div>
                </div>

                {/* Summary Cards */}
                <div className="grid grid-cols-3 gap-4">
                    <SummaryCard
                        icon={<Zap className="w-4 h-4 text-white" />}
                        label="Best ΔV"
                        value={summary.best_delta_v ? `${summary.best_delta_v.toFixed(0)} m/s` : '—'}
                        color="var(--accent-blue)"
                    />
                    <SummaryCard
                        icon={<Clock className="w-4 h-4 text-white" />}
                        label="Best Time"
                        value={summary.best_time_of_flight ? `${summary.best_time_of_flight.toFixed(1)} days` : '—'}
                        color="var(--accent-green)"
                    />
                    <SummaryCard
                        icon={<Target className="w-4 h-4 text-white" />}
                        label="Best Cost"
                        value={summary.best_cost ? summary.best_cost.toFixed(4) : '—'}
                        color="var(--accent-purple)"
                    />
                </div>

                {/* Comparison Table */}
                <div className="glass rounded-lg overflow-hidden">
                    <table className="w-full text-xs">
                        <thead className="bg-theme-secondary border-b border-theme">
                            <tr>
                                <th className="text-left p-3 font-medium text-theme-muted">Run ID</th>
                                <th className="text-center p-3 font-medium text-theme-muted">Method</th>
                                <th className="text-right p-3 font-medium text-theme-muted">ΔV (m/s)</th>
                                <th className="text-right p-3 font-medium text-theme-muted">Time (days)</th>
                                <th className="text-right p-3 font-medium text-theme-muted">Fuel (kg)</th>
                                <th className="text-right p-3 font-medium text-theme-muted">Cost</th>
                                <th className="text-center p-3 font-medium text-theme-muted">Status</th>
                                <th className="text-right p-3 font-medium text-theme-muted">Iterations</th>
                            </tr>
                        </thead>
                        <tbody>
                            {runs.map((run, idx) => {
                                // Check if this run has best values
                                const hasBestDeltaV = run.delta_v === summary.best_delta_v
                                const hasBestTime = run.time_of_flight === summary.best_time_of_flight
                                const hasBestCost = run.final_cost === summary.best_cost

                                return (
                                    <tr
                                        key={run.run_id}
                                        className={`border-b border-theme transition-colors ${
                                            idx % 2 === 0 ? 'bg-theme-primary' : 'bg-theme-secondary'
                                        }`}
                                    >
                                        {/* Run ID */}
                                        <td className="p-3 font-mono text-theme-primary">
                                            {run.run_id}
                                        </td>

                                        {/* Method */}
                                        <td className="p-3 text-center">
                                            <span
                                                className="inline-block px-2 py-0.5 rounded text-xs font-medium text-white capitalize"
                                                style={{
                                                    backgroundColor: run.method === 'classical'
                                                        ? 'var(--accent-purple)'
                                                        : 'var(--accent-cyan)'
                                                }}
                                            >
                                                {run.method}
                                            </span>
                                        </td>

                                        {/* Delta-V */}
                                        <td className="p-3 text-right">
                                            <div className="flex items-center justify-end gap-2">
                                                {hasBestDeltaV && <Award className="w-3 h-3 text-accent-blue" />}
                                                <span className="font-mono text-theme-primary">
                                                    {run.delta_v?.toFixed(0) || '—'}
                                                </span>
                                                {run.delta_v_relative !== undefined && (
                                                    <span className={`text-xs ${run.delta_v_relative > 0 ? 'text-accent-red' : 'text-accent-green'}`}>
                                                        ({run.delta_v_relative > 0 ? '+' : ''}{run.delta_v_relative.toFixed(1)}%)
                                                    </span>
                                                )}
                                            </div>
                                        </td>

                                        {/* Time of Flight */}
                                        <td className="p-3 text-right">
                                            <div className="flex items-center justify-end gap-2">
                                                {hasBestTime && <Award className="w-3 h-3 text-accent-green" />}
                                                <span className="font-mono text-theme-primary">
                                                    {run.time_of_flight?.toFixed(1) || '—'}
                                                </span>
                                                {run.time_relative !== undefined && (
                                                    <span className={`text-xs ${run.time_relative > 0 ? 'text-accent-red' : 'text-accent-green'}`}>
                                                        ({run.time_relative > 0 ? '+' : ''}{run.time_relative.toFixed(1)}%)
                                                    </span>
                                                )}
                                            </div>
                                        </td>

                                        {/* Fuel Consumed */}
                                        <td className="p-3 text-right font-mono text-theme-primary">
                                            {run.fuel_consumed?.toFixed(1) || '—'}
                                        </td>

                                        {/* Cost */}
                                        <td className="p-3 text-right">
                                            <div className="flex items-center justify-end gap-2">
                                                {hasBestCost && <Award className="w-3 h-3 text-accent-purple" />}
                                                <span className="font-mono text-theme-primary">
                                                    {run.final_cost?.toFixed(4) || '—'}
                                                </span>
                                                {run.cost_relative !== undefined && (
                                                    <span className={`text-xs ${run.cost_relative > 0 ? 'text-accent-red' : 'text-accent-green'}`}>
                                                        ({run.cost_relative > 0 ? '+' : ''}{run.cost_relative.toFixed(1)}%)
                                                    </span>
                                                )}
                                            </div>
                                        </td>

                                        {/* Captured Status */}
                                        <td className="p-3 text-center">
                                            {run.captured ? (
                                                <span className="inline-block px-2 py-0.5 rounded text-xs bg-accent-green text-white">
                                                    Captured
                                                </span>
                                            ) : (
                                                <span className="text-theme-muted">—</span>
                                            )}
                                        </td>

                                        {/* Iterations */}
                                        <td className="p-3 text-right font-mono text-theme-secondary">
                                            {run.num_iterations}
                                        </td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                </div>

                {/* Method Performance Summary */}
                <div className="grid grid-cols-2 gap-4">
                    <MethodSummary
                        method="classical"
                        runs={runs.filter(r => r.method === 'classical')}
                    />
                    <MethodSummary
                        method="hybrid"
                        runs={runs.filter(r => r.method === 'hybrid')}
                    />
                </div>

                {/* Legend */}
                <div className="glass rounded-lg p-4">
                    <div className="text-xs text-theme-muted space-y-2">
                        <div className="flex items-center gap-2">
                            <Award className="w-3 h-3 text-accent-blue" />
                            <span>Indicates best value across all runs</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="text-accent-green">(-X%)</span>
                            <span>Better than best (lower is better for ΔV, Time, Cost)</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="text-accent-red">(+X%)</span>
                            <span>Worse than best</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

// Helper Components

interface SummaryCardProps {
    icon: React.ReactNode
    label: string
    value: string
    color: string
}

function SummaryCard({ icon, label, value, color }: SummaryCardProps) {
    return (
        <div className="glass rounded-lg p-4 space-y-2">
            <div className="flex items-center gap-2">
                <div className="p-1.5 rounded" style={{ backgroundColor: color }}>
                    {icon}
                </div>
                <span className="text-xs text-theme-muted">{label}</span>
            </div>
            <div className="text-2xl font-bold text-theme-primary">{value}</div>
        </div>
    )
}

interface MethodSummaryProps {
    method: 'classical' | 'hybrid'
    runs: any[]
}

function MethodSummary({ method, runs }: MethodSummaryProps) {
    if (runs.length === 0) return null

    const avgDeltaV = runs.reduce((sum, r) => sum + (r.delta_v || 0), 0) / runs.length
    const avgTime = runs.reduce((sum, r) => sum + (r.time_of_flight || 0), 0) / runs.length
    const avgCost = runs.reduce((sum, r) => sum + (r.final_cost || 0), 0) / runs.length
    const captureRate = runs.filter(r => r.captured).length / runs.length * 100

    return (
        <div className="glass rounded-lg p-4 space-y-3">
            <div className="flex items-center gap-2">
                <div
                    className="px-2 py-1 rounded text-xs font-medium text-white capitalize"
                    style={{
                        backgroundColor: method === 'classical'
                            ? 'var(--accent-purple)'
                            : 'var(--accent-cyan)'
                    }}
                >
                    {method}
                </div>
                <span className="text-xs text-theme-muted">({runs.length} runs)</span>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                    <div className="text-theme-muted mb-0.5">Avg ΔV</div>
                    <div className="font-mono text-theme-primary">{avgDeltaV.toFixed(0)} m/s</div>
                </div>
                <div>
                    <div className="text-theme-muted mb-0.5">Avg Time</div>
                    <div className="font-mono text-theme-primary">{avgTime.toFixed(1)} days</div>
                </div>
                <div>
                    <div className="text-theme-muted mb-0.5">Avg Cost</div>
                    <div className="font-mono text-theme-primary">{avgCost.toFixed(4)}</div>
                </div>
                <div>
                    <div className="text-theme-muted mb-0.5">Capture Rate</div>
                    <div className="font-mono text-theme-primary">{captureRate.toFixed(0)}%</div>
                </div>
            </div>
        </div>
    )
}
