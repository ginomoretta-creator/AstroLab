/**
 * Run Manager Component
 *
 * Displays list of saved simulation runs with:
 * - Multi-select for comparison
 * - Filter by method (classical/hybrid)
 * - Sort by timestamp/cost
 * - Delete functionality
 * - Save current run button
 */

import { useEffect, useState } from 'react'
import { Save, Trash2, RefreshCw, Filter, Search, Check } from 'lucide-react'
import { useRunStorage } from '../../hooks/useRunStorage'
import { useSimulationStore } from '../../stores/simulationStore'

export default function RunManager() {
    const {
        savedRuns,
        selectedRuns,
        status,
        toggleRunSelection,
        clearRunSelection,
        setError,
    } = useSimulationStore()

    const {
        loadRuns,
        saveCurrentRun,
        deleteRun,
    } = useRunStorage()

    const [filterMethod, setFilterMethod] = useState<'all' | 'classical' | 'hybrid'>('all')
    const [searchQuery, setSearchQuery] = useState('')
    const [isRefreshing, setIsRefreshing] = useState(false)

    // Load runs on mount
    useEffect(() => {
        loadRuns()
    }, [])

    // Handle refresh
    const handleRefresh = async () => {
        setIsRefreshing(true)
        await loadRuns()
        setTimeout(() => setIsRefreshing(false), 500)
    }

    // Handle save current run
    const handleSaveCurrentRun = async () => {
        if (status !== 'completed') {
            setError('Complete a simulation first before saving')
            return
        }

        const runId = await saveCurrentRun()
        if (runId) {
            console.log('Run saved with ID:', runId)
        }
    }

    // Handle delete run
    const handleDeleteRun = async (runId: string, event: React.MouseEvent) => {
        event.stopPropagation() // Prevent row selection

        if (!confirm('Delete this run? This cannot be undone.')) {
            return
        }

        await deleteRun(runId)
    }

    // Filter runs
    const filteredRuns = savedRuns.filter(run => {
        const matchesMethod = filterMethod === 'all' || run.method === filterMethod
        const matchesSearch = searchQuery === '' ||
            run.run_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
            run.timestamp.includes(searchQuery)
        return matchesMethod && matchesSearch
    })

    // Format timestamp
    const formatTimestamp = (timestamp: string) => {
        try {
            const date = new Date(timestamp)
            return date.toLocaleString('en-US', {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            })
        } catch {
            return timestamp.slice(0, 16)
        }
    }

    // Format cost
    const formatCost = (cost?: number) => {
        if (cost === undefined || cost === null) return '—'
        return cost.toFixed(4)
    }

    // Format deltaV
    const formatDeltaV = (deltaV?: number) => {
        if (deltaV === undefined || deltaV === null) return '—'
        return `${deltaV.toFixed(0)} m/s`
    }

    // Format time
    const formatTime = (days?: number) => {
        if (days === undefined || days === null) return '—'
        return `${days.toFixed(1)} days`
    }

    return (
        <div className="h-full flex flex-col bg-theme-primary">
            {/* Toolbar */}
            <div className="flex items-center justify-between p-3 border-b border-theme bg-theme-secondary">
                <div className="flex items-center gap-2">
                    <h3 className="text-sm font-semibold text-theme-primary">Run Manager</h3>
                    <span className="text-xs text-theme-muted">({filteredRuns.length} runs)</span>
                </div>

                <div className="flex items-center gap-2">
                    {/* Save Current Run Button */}
                    <button
                        onClick={handleSaveCurrentRun}
                        disabled={status !== 'completed'}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        style={{
                            backgroundColor: status === 'completed' ? 'var(--accent-green)' : 'var(--theme-tertiary)',
                            color: status === 'completed' ? 'white' : 'var(--theme-muted)'
                        }}
                    >
                        <Save className="w-3.5 h-3.5" />
                        Save Current
                    </button>

                    {/* Refresh Button */}
                    <button
                        onClick={handleRefresh}
                        className="p-1.5 rounded hover:bg-theme-elevated transition-colors"
                        title="Refresh runs"
                    >
                        <RefreshCw className={`w-4 h-4 text-theme-muted ${isRefreshing ? 'animate-spin' : ''}`} />
                    </button>
                </div>
            </div>

            {/* Filters */}
            <div className="flex items-center gap-3 p-3 border-b border-theme bg-theme-secondary">
                {/* Method Filter */}
                <div className="flex items-center gap-2">
                    <Filter className="w-3.5 h-3.5 text-theme-muted" />
                    <select
                        value={filterMethod}
                        onChange={(e) => setFilterMethod(e.target.value as any)}
                        className="text-xs px-2 py-1 rounded bg-theme-tertiary text-theme-primary border border-theme focus:outline-none focus:ring-1 focus:ring-accent-blue"
                    >
                        <option value="all">All Methods</option>
                        <option value="classical">Classical</option>
                        <option value="hybrid">Hybrid</option>
                    </select>
                </div>

                {/* Search */}
                <div className="flex-1 flex items-center gap-2 px-2 py-1 rounded bg-theme-tertiary border border-theme">
                    <Search className="w-3.5 h-3.5 text-theme-muted" />
                    <input
                        type="text"
                        placeholder="Search by ID or date..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="flex-1 text-xs bg-transparent outline-none text-theme-primary placeholder-theme-muted"
                    />
                </div>

                {/* Selection Actions */}
                {selectedRuns.length > 0 && (
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-theme-muted">{selectedRuns.length} selected</span>
                        <button
                            onClick={clearRunSelection}
                            className="text-xs px-2 py-1 rounded hover:bg-theme-elevated transition-colors text-theme-muted"
                        >
                            Clear
                        </button>
                    </div>
                )}
            </div>

            {/* Run List */}
            <div className="flex-1 overflow-y-auto">
                {filteredRuns.length === 0 ? (
                    <div className="h-full flex items-center justify-center text-theme-muted text-sm">
                        {savedRuns.length === 0 ? 'No saved runs yet' : 'No runs match filters'}
                    </div>
                ) : (
                    <table className="w-full text-xs">
                        <thead className="sticky top-0 bg-theme-secondary border-b border-theme">
                            <tr>
                                <th className="w-8 p-2"></th>
                                <th className="text-left p-2 font-medium text-theme-muted">Method</th>
                                <th className="text-left p-2 font-medium text-theme-muted">Timestamp</th>
                                <th className="text-right p-2 font-medium text-theme-muted">Cost</th>
                                <th className="text-right p-2 font-medium text-theme-muted">ΔV</th>
                                <th className="text-right p-2 font-medium text-theme-muted">Time</th>
                                <th className="text-center p-2 font-medium text-theme-muted">Status</th>
                                <th className="w-10 p-2"></th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredRuns.map((run) => {
                                const isSelected = selectedRuns.includes(run.run_id)

                                return (
                                    <tr
                                        key={run.run_id}
                                        onClick={() => toggleRunSelection(run.run_id)}
                                        className={`cursor-pointer border-b border-theme transition-colors ${
                                            isSelected ? 'bg-theme-elevated' : 'hover:bg-theme-secondary'
                                        }`}
                                    >
                                        {/* Checkbox */}
                                        <td className="p-2">
                                            <div className={`w-4 h-4 rounded border-2 flex items-center justify-center transition-colors ${
                                                isSelected
                                                    ? 'border-accent-blue bg-accent-blue'
                                                    : 'border-theme-muted'
                                            }`}>
                                                {isSelected && <Check className="w-3 h-3 text-white" />}
                                            </div>
                                        </td>

                                        {/* Method */}
                                        <td className="p-2">
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

                                        {/* Timestamp */}
                                        <td className="p-2 text-theme-secondary font-mono">
                                            {formatTimestamp(run.timestamp)}
                                        </td>

                                        {/* Cost */}
                                        <td className="p-2 text-right font-mono text-theme-primary">
                                            {formatCost(run.finalCost)}
                                        </td>

                                        {/* Delta-V */}
                                        <td className="p-2 text-right text-theme-secondary">
                                            {formatDeltaV(run.deltaV)}
                                        </td>

                                        {/* Time of Flight */}
                                        <td className="p-2 text-right text-theme-secondary">
                                            {formatTime(run.timeOfFlight)}
                                        </td>

                                        {/* Captured Status */}
                                        <td className="p-2 text-center">
                                            {run.captured ? (
                                                <span className="inline-block px-2 py-0.5 rounded text-xs bg-accent-green text-white">
                                                    Captured
                                                </span>
                                            ) : (
                                                <span className="text-theme-muted">—</span>
                                            )}
                                        </td>

                                        {/* Delete Button */}
                                        <td className="p-2">
                                            <button
                                                onClick={(e) => handleDeleteRun(run.run_id, e)}
                                                className="p-1 rounded hover:bg-theme-elevated transition-colors group"
                                                title="Delete run"
                                            >
                                                <Trash2 className="w-3.5 h-3.5 text-theme-muted group-hover:text-accent-red" />
                                            </button>
                                        </td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    )
}
