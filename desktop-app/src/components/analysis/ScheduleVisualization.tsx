import { useMemo } from 'react'
import { useSimulationStore } from '../../stores/simulationStore'

export default function ScheduleVisualization() {
    const { results, currentMethod, params, status } = useSimulationStore()

    // Get schedule from best trajectory (approximate thrust pattern)
    const schedule = useMemo(() => {
        const result = results[currentMethod]
        if (!result?.bestTrajectory || result.bestTrajectory.length < 2) {
            return []
        }

        // Compute velocity changes to infer thrust
        const traj = result.bestTrajectory
        const schedule: number[] = []

        for (let i = 1; i < Math.min(traj.length, params.numSteps); i++) {
            const dvx = traj[i][2] - traj[i - 1][2]
            const dvy = traj[i][3] - traj[i - 1][3]
            const dv = Math.sqrt(dvx * dvx + dvy * dvy)

            // Threshold for thrust detection
            schedule.push(dv > 0.001 ? 1 : 0)
        }

        return schedule
    }, [results, currentMethod, params.numSteps])

    if (schedule.length === 0 && status !== 'running') {
        return (
            <div className="h-full flex items-center justify-center text-theme-muted text-sm">
                Run a simulation to see thrust schedule visualization
            </div>
        )
    }

    const thrustCount = schedule.filter(s => s === 1).length
    const thrustFraction = schedule.length > 0 ? (thrustCount / schedule.length) * 100 : 0

    // Downsample for display if too many steps
    const displaySchedule = useMemo(() => {
        const maxBars = 200
        if (schedule.length <= maxBars) return schedule

        const ratio = Math.ceil(schedule.length / maxBars)
        const downsampled: number[] = []

        for (let i = 0; i < schedule.length; i += ratio) {
            const chunk = schedule.slice(i, i + ratio)
            const avg = chunk.reduce((a, b) => a + b, 0) / chunk.length
            downsampled.push(avg > 0.5 ? 1 : 0)
        }

        return downsampled
    }, [schedule])

    return (
        <div className="h-full flex flex-col">
            <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold text-theme-muted uppercase tracking-wider">
                    Thrust Schedule
                </h3>
                <div className="flex items-center gap-4 text-xs">
                    <span className="text-theme-muted">
                        Thrust: <span className="font-mono" style={{ color: 'var(--accent-blue)' }}>
                            {thrustFraction.toFixed(1)}%
                        </span>
                    </span>
                    <span className="text-theme-muted">
                        Coast: <span className="font-mono text-theme-secondary">
                            {(100 - thrustFraction).toFixed(1)}%
                        </span>
                    </span>
                    {status === 'running' && (
                        <span className="flex items-center gap-1">
                            <div className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: 'var(--accent-green)' }}></div>
                            <span style={{ color: 'var(--accent-green)' }}>Live</span>
                        </span>
                    )}
                </div>
            </div>

            {/* Schedule Bars */}
            <div className="flex-1 flex items-center gap-[1px] px-1 min-h-0">
                {displaySchedule.length > 0 ? (
                    displaySchedule.map((value, index) => (
                        <div
                            key={index}
                            className="flex-1 h-full rounded-sm transition-all"
                            style={{
                                backgroundColor: value === 1 ? 'var(--accent-blue)' : 'var(--bg-tertiary)',
                                opacity: value === 1 ? 0.7 + (index / displaySchedule.length) * 0.3 : 0.4
                            }}
                        />
                    ))
                ) : (
                    // Loading placeholder
                    Array.from({ length: 50 }).map((_, i) => (
                        <div
                            key={i}
                            className="flex-1 h-full rounded-sm animate-pulse"
                            style={{
                                backgroundColor: 'var(--bg-tertiary)',
                                animationDelay: `${i * 20}ms`
                            }}
                        />
                    ))
                )}
            </div>

            {/* Legend */}
            <div className="flex items-center gap-4 mt-2 text-xs">
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: 'var(--accent-blue)' }}></div>
                    <span className="text-theme-muted">Thrust</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-sm bg-theme-tertiary"></div>
                    <span className="text-theme-muted">Coast</span>
                </div>
                <div className="ml-auto text-theme-muted font-mono">
                    {schedule.length > 0 ? `${schedule.length} steps` : `${params.numSteps} steps (pending)`}
                </div>
            </div>
        </div>
    )
}
