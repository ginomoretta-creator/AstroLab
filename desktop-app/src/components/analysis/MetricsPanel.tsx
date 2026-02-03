/**
 * Metrics Panel Component
 *
 * Displays enhanced metrics for the current simulation:
 * - ΔV total (analytical and numerical)
 * - Time of flight
 * - Fuel consumed
 * - Cost breakdown
 * - Convergence info
 */

import { Fuel, Clock, Target, TrendingDown, Zap } from 'lucide-react'
import { useSimulationStore } from '../../stores/simulationStore'

interface MetricCardProps {
    icon: React.ReactNode
    label: string
    value: string
    unit?: string
    color?: string
}

function MetricCard({ icon, label, value, unit, color }: MetricCardProps) {
    return (
        <div className="glass rounded-lg p-3 space-y-2">
            <div className="flex items-center gap-2">
                <div className="p-1.5 rounded" style={{ backgroundColor: color || 'var(--theme-tertiary)' }}>
                    {icon}
                </div>
                <span className="text-xs text-theme-muted">{label}</span>
            </div>
            <div className="flex items-baseline gap-1">
                <span className="text-2xl font-bold text-theme-primary">{value}</span>
                {unit && <span className="text-xs text-theme-muted">{unit}</span>}
            </div>
        </div>
    )
}

export default function MetricsPanel() {
    const { status, currentMethod, results, iterationHistory } = useSimulationStore()

    const currentResult = results[currentMethod]

    // Get latest iteration metrics
    const latestIteration = iterationHistory.length > 0
        ? iterationHistory[iterationHistory.length - 1]
        : null

    const metrics = latestIteration?.metrics

    // Calculate convergence info
    const convergenceRate = iterationHistory.length >= 2
        ? ((iterationHistory[0].bestCost - (latestIteration?.bestCost || 0)) / iterationHistory[0].bestCost * 100)
        : 0

    // Format values
    const formatNumber = (value?: number, decimals: number = 0) => {
        if (value === undefined || value === null) return '—'
        return value.toFixed(decimals)
    }

    return (
        <div className="h-full overflow-y-auto bg-theme-primary p-4">
            <div className="max-w-4xl mx-auto space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <div>
                        <h3 className="text-lg font-bold text-theme-primary">Enhanced Metrics</h3>
                        <p className="text-xs text-theme-muted mt-1">
                            {status === 'completed'
                                ? 'Final results'
                                : status === 'running'
                                    ? `Live metrics (Iteration ${latestIteration?.iteration || 0})`
                                    : metrics
                                        ? `Last run results (${iterationHistory.length} iterations)`
                                        : 'No data yet'}
                        </p>
                    </div>
                    <div
                        className="px-3 py-1.5 rounded text-xs font-medium text-white capitalize"
                        style={{
                            backgroundColor: currentMethod === 'classical'
                                ? 'var(--accent-purple)'
                                : 'var(--accent-cyan)'
                        }}
                    >
                        {currentMethod}
                    </div>
                </div>

                {/* Main Metrics Grid */}
                {metrics ? (
                    <>
                        <div className="grid grid-cols-3 gap-4">
                            {/* Delta-V */}
                            <MetricCard
                                icon={<Zap className="w-4 h-4 text-white" />}
                                label="ΔV Total"
                                value={formatNumber(metrics.deltaV, 0)}
                                unit="m/s"
                                color="var(--accent-blue)"
                            />

                            {/* Time of Flight */}
                            <MetricCard
                                icon={<Clock className="w-4 h-4 text-white" />}
                                label="Time of Flight"
                                value={formatNumber(metrics.timeOfFlight, 1)}
                                unit="days"
                                color="var(--accent-green)"
                            />

                            {/* Fuel Consumed */}
                            <MetricCard
                                icon={<Fuel className="w-4 h-4 text-white" />}
                                label="Fuel Used"
                                value={formatNumber(metrics.fuelConsumed, 1)}
                                unit="kg"
                                color="var(--accent-orange)"
                            />
                        </div>

                        {/* Cost Breakdown */}
                        {metrics.costBreakdown && (
                            <div className="glass rounded-lg p-4 space-y-3">
                                <div className="flex items-center gap-2 text-sm font-semibold text-theme-primary">
                                    <Target className="w-4 h-4" />
                                    Cost Breakdown
                                </div>

                                <div className="space-y-2">
                                    <CostBar
                                        label="Distance Cost"
                                        value={metrics.costBreakdown.distance}
                                        max={Math.max(
                                            metrics.costBreakdown.distance,
                                            metrics.costBreakdown.velocity,
                                            metrics.costBreakdown.fuel,
                                            Math.abs(metrics.costBreakdown.capture_bonus)
                                        )}
                                        color="var(--accent-purple)"
                                    />
                                    <CostBar
                                        label="Velocity Cost"
                                        value={metrics.costBreakdown.velocity}
                                        max={Math.max(
                                            metrics.costBreakdown.distance,
                                            metrics.costBreakdown.velocity,
                                            metrics.costBreakdown.fuel,
                                            Math.abs(metrics.costBreakdown.capture_bonus)
                                        )}
                                        color="var(--accent-cyan)"
                                    />
                                    <CostBar
                                        label="Fuel Penalty"
                                        value={metrics.costBreakdown.fuel}
                                        max={Math.max(
                                            metrics.costBreakdown.distance,
                                            metrics.costBreakdown.velocity,
                                            metrics.costBreakdown.fuel,
                                            Math.abs(metrics.costBreakdown.capture_bonus)
                                        )}
                                        color="var(--accent-orange)"
                                    />
                                    <CostBar
                                        label="Capture Bonus"
                                        value={Math.abs(metrics.costBreakdown.capture_bonus)}
                                        max={Math.max(
                                            metrics.costBreakdown.distance,
                                            metrics.costBreakdown.velocity,
                                            metrics.costBreakdown.fuel,
                                            Math.abs(metrics.costBreakdown.capture_bonus)
                                        )}
                                        color="var(--accent-green)"
                                        isBonus={metrics.costBreakdown.capture_bonus < 0}
                                    />
                                </div>
                            </div>
                        )}

                        {/* Convergence Info */}
                        <div className="glass rounded-lg p-4 space-y-3">
                            <div className="flex items-center gap-2 text-sm font-semibold text-theme-primary">
                                <TrendingDown className="w-4 h-4" />
                                Convergence
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <div className="text-xs text-theme-muted mb-1">Improvement</div>
                                    <div className="text-xl font-bold text-theme-primary">
                                        {convergenceRate > 0 ? '+' : ''}{formatNumber(convergenceRate, 1)}%
                                    </div>
                                </div>
                                <div>
                                    <div className="text-xs text-theme-muted mb-1">Current Cost</div>
                                    <div className="text-xl font-bold text-theme-primary">
                                        {formatNumber(currentResult?.bestCost, 4)}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Additional Info */}
                        <div className="glass rounded-lg p-4">
                            <div className="grid grid-cols-2 gap-x-6 gap-y-3 text-xs">
                                <InfoRow label="ΔV (Numerical)" value={`${formatNumber(metrics.deltaV_numerical, 0)} m/s`} />
                                <InfoRow label="Iterations" value={iterationHistory.length.toString()} />
                                <InfoRow label="Best Distance" value={`${formatNumber(currentResult?.bestDistance ? currentResult.bestDistance * 384400 : undefined, 0)} km`} />
                                <InfoRow label="Thrust Fraction" value={`${formatNumber(currentResult?.bestThrustFraction ? currentResult.bestThrustFraction * 100 : undefined, 1)}%`} />
                            </div>
                        </div>
                    </>
                ) : (
                    <div className="glass rounded-lg p-8 text-center">
                        <div className="text-theme-muted text-sm">
                            Run a simulation to see enhanced metrics
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}

// Helper Components

interface CostBarProps {
    label: string
    value: number
    max: number
    color: string
    isBonus?: boolean
}

function CostBar({ label, value, max, color, isBonus = false }: CostBarProps) {
    const percentage = max > 0 ? (value / max) * 100 : 0

    return (
        <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
                <span className="text-theme-secondary">{label}</span>
                <span className="font-mono text-theme-primary">
                    {isBonus ? '-' : ''}{value.toFixed(3)}
                </span>
            </div>
            <div className="h-2 bg-theme-tertiary rounded-full overflow-hidden">
                <div
                    className="h-full transition-all duration-300 rounded-full"
                    style={{
                        width: `${Math.min(percentage, 100)}%`,
                        backgroundColor: color
                    }}
                />
            </div>
        </div>
    )
}

interface InfoRowProps {
    label: string
    value: string
}

function InfoRow({ label, value }: InfoRowProps) {
    return (
        <div className="flex items-center justify-between">
            <span className="text-theme-muted">{label}</span>
            <span className="font-mono text-theme-primary">{value}</span>
        </div>
    )
}
