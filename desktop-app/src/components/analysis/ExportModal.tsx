/**
 * Export Modal Component
 *
 * Modal for exporting selected runs in multiple formats:
 * - CSV (metrics table)
 * - JSON (full data)
 * - MATLAB (.mat format)
 */

import { useState } from 'react'
import { Download, X, FileText, Code, FileSpreadsheet } from 'lucide-react'
import { useSimulationStore } from '../../stores/simulationStore'
import { useRunStorage } from '../../hooks/useRunStorage'

interface ExportModalProps {
    isOpen: boolean
    onClose: () => void
}

type ExportFormat = 'csv' | 'json' | 'matlab'

export default function ExportModal({ isOpen, onClose }: ExportModalProps) {
    const { selectedRuns, isExporting } = useSimulationStore()
    const { exportRuns } = useRunStorage()

    const [selectedFormat, setSelectedFormat] = useState<ExportFormat>('csv')

    if (!isOpen) return null

    const handleExport = async () => {
        const success = await exportRuns(selectedFormat)
        if (success) {
            onClose()
        }
    }

    const formats: Array<{
        id: ExportFormat
        icon: React.ReactNode
        name: string
        description: string
        fileExtension: string
    }> = [
        {
            id: 'csv',
            icon: <FileSpreadsheet className="w-5 h-5" />,
            name: 'CSV',
            description: 'Metrics table for Excel/spreadsheets',
            fileExtension: '.csv'
        },
        {
            id: 'json',
            icon: <Code className="w-5 h-5" />,
            name: 'JSON',
            description: 'Full data including trajectories',
            fileExtension: '.json'
        },
        {
            id: 'matlab',
            icon: <FileText className="w-5 h-5" />,
            name: 'MATLAB',
            description: 'MATLAB .mat format for analysis',
            fileExtension: '.mat'
        }
    ]

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="glass rounded-lg shadow-2xl w-full max-w-lg mx-4 overflow-hidden animate-fade-in">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-theme bg-theme-secondary">
                    <div className="flex items-center gap-2">
                        <Download className="w-5 h-5 text-theme-primary" />
                        <h3 className="text-lg font-bold text-theme-primary">Export Runs</h3>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1 rounded hover:bg-theme-elevated transition-colors"
                        disabled={isExporting}
                    >
                        <X className="w-5 h-5 text-theme-muted" />
                    </button>
                </div>

                {/* Content */}
                <div className="p-4 space-y-4">
                    {/* Selected Runs Info */}
                    <div className="glass rounded-lg p-3 text-sm">
                        <div className="text-theme-muted mb-1">Selected Runs</div>
                        <div className="text-theme-primary font-medium">
                            {selectedRuns.length} run{selectedRuns.length !== 1 ? 's' : ''} selected
                        </div>
                    </div>

                    {/* Format Selection */}
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-theme-primary">Export Format</label>
                        <div className="space-y-2">
                            {formats.map(format => (
                                <button
                                    key={format.id}
                                    onClick={() => setSelectedFormat(format.id)}
                                    disabled={isExporting}
                                    className={`w-full flex items-start gap-3 p-3 rounded-lg border-2 transition-all ${
                                        selectedFormat === format.id
                                            ? 'border-accent-blue bg-theme-elevated'
                                            : 'border-theme hover:border-theme-muted'
                                    } ${isExporting ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                                >
                                    {/* Icon */}
                                    <div
                                        className={`p-2 rounded ${
                                            selectedFormat === format.id
                                                ? 'bg-accent-blue text-white'
                                                : 'bg-theme-tertiary text-theme-muted'
                                        }`}
                                    >
                                        {format.icon}
                                    </div>

                                    {/* Info */}
                                    <div className="flex-1 text-left">
                                        <div className="flex items-center gap-2">
                                            <span className="font-medium text-theme-primary">{format.name}</span>
                                            <span className="text-xs text-theme-muted font-mono">{format.fileExtension}</span>
                                        </div>
                                        <div className="text-xs text-theme-muted mt-0.5">{format.description}</div>
                                    </div>

                                    {/* Radio indicator */}
                                    <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors ${
                                        selectedFormat === format.id
                                            ? 'border-accent-blue'
                                            : 'border-theme-muted'
                                    }`}>
                                        {selectedFormat === format.id && (
                                            <div className="w-2.5 h-2.5 rounded-full bg-accent-blue" />
                                        )}
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Format Details */}
                    <div className="glass rounded-lg p-3 text-xs space-y-1">
                        <div className="font-medium text-theme-primary mb-1">What's included:</div>
                        {selectedFormat === 'csv' && (
                            <ul className="text-theme-muted space-y-0.5 list-disc list-inside">
                                <li>Run metadata (ID, timestamp, method)</li>
                                <li>Final metrics (ΔV, time, fuel, cost)</li>
                                <li>Convergence statistics</li>
                                <li>Easy to open in Excel or Google Sheets</li>
                            </ul>
                        )}
                        {selectedFormat === 'json' && (
                            <ul className="text-theme-muted space-y-0.5 list-disc list-inside">
                                <li>Complete run data (all iterations)</li>
                                <li>Full trajectory points</li>
                                <li>Thrust schedules</li>
                                <li>All metrics and parameters</li>
                                <li>Suitable for programmatic analysis</li>
                            </ul>
                        )}
                        {selectedFormat === 'matlab' && (
                            <ul className="text-theme-muted space-y-0.5 list-disc list-inside">
                                <li>MATLAB-compatible .mat format</li>
                                <li>Structured arrays for easy access</li>
                                <li>Trajectory matrices</li>
                                <li>Ready for analysis in MATLAB/Octave</li>
                            </ul>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-end gap-2 p-4 border-t border-theme bg-theme-secondary">
                    <button
                        onClick={onClose}
                        disabled={isExporting}
                        className="px-4 py-2 text-sm font-medium rounded transition-colors bg-theme-tertiary text-theme-primary hover:bg-theme-elevated disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleExport}
                        disabled={isExporting || selectedRuns.length === 0}
                        className="px-4 py-2 text-sm font-medium rounded transition-colors text-white flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                        style={{ backgroundColor: 'var(--accent-blue)' }}
                    >
                        {isExporting ? (
                            <>
                                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                Exporting...
                            </>
                        ) : (
                            <>
                                <Download className="w-4 h-4" />
                                Export
                            </>
                        )}
                    </button>
                </div>
            </div>
        </div>
    )
}
