import { useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Stars, PerspectiveCamera } from '@react-three/drei'
import { Moon, History, Activity, Database, TrendingDown, BarChart3, Download } from 'lucide-react'
import { useThemeStore } from '../../stores/themeStore'
import { useSimulationStore } from '../../stores/simulationStore'
import TrajectoryScene from '../visualization/TrajectoryScene'
import StatsPanel from '../analysis/StatsPanel'
import ScheduleVisualization from '../analysis/ScheduleVisualization'
import IterationHistory from '../analysis/IterationHistory'
import RunManager from '../analysis/RunManager'
import MetricsPanel from '../analysis/MetricsPanel'
import ComparisonDashboard from '../analysis/ComparisonDashboard'
import ExportModal from '../analysis/ExportModal'
import FloatingPanel from '../common/FloatingPanel'
import LunarAnalysis from '../analysis/LunarAnalysis'
import { ErrorBoundary } from '../common/ErrorBoundary'

export default function MainContent() {
    const { theme } = useThemeStore()
    const { currentMethod, selectedRuns } = useSimulationStore()
    const [activeTab, setActiveTab] = useState<'schedule' | 'history' | 'runs' | 'metrics' | 'compare'>('schedule')
    const [isLunarPanelOpen, setIsLunarPanelOpen] = useState(false)
    const [isExportModalOpen, setIsExportModalOpen] = useState(false)

    // Adjust star visibility based on theme
    const showStars = theme === 'dark'
    const backgroundColor = theme === 'dark' ? '#0a0a0f' : '#f8fafc'

    return (
        <main className="flex-1 flex flex-col overflow-hidden bg-theme-primary relative">
            {/* Floating Windows */}
            <FloatingPanel
                title="Lunar Analysis"
                isOpen={isLunarPanelOpen}
                onClose={() => setIsLunarPanelOpen(false)}
                initialPosition={{ x: 100, y: 100 }}
                initialSize={{ width: 500, height: 500 }}
            >
                <LunarAnalysis />
            </FloatingPanel>

            {/* 3D Visualization */}
            <div className="flex-1 relative">
                <ErrorBoundary>
                    <Canvas
                        className="bg-theme-primary"
                        style={{ background: backgroundColor }}
                    >
                        <PerspectiveCamera makeDefault position={[0, 0, 3]} />
                        <ambientLight intensity={theme === 'dark' ? 0.2 : 0.5} />
                        <pointLight position={[10, 10, 10]} intensity={theme === 'dark' ? 1 : 0.8} />
                        {showStars && (
                            <Stars
                                radius={100}
                                depth={50}
                                count={5000}
                                factor={4}
                                saturation={0}
                                fade
                                speed={1}
                            />
                        )}
                        <TrajectoryScene />
                        <OrbitControls
                            enablePan={true}
                            enableZoom={true}
                            enableRotate={true}
                            minDistance={0.5}
                            maxDistance={10}
                        />
                    </Canvas>
                </ErrorBoundary>

                {/* Overlay Stats */}
                <div className="absolute top-4 right-4 z-10">
                    <StatsPanel />
                </div>

                {/* Toolbar / Toggles */}
                <div className="absolute top-4 left-4 z-10 flex flex-col gap-2">
                    {/* Toolbar buttons here if any (removed Moon button) */}
                </div>

                {/* Legend */}
                <div className="absolute bottom-4 left-4 z-10 glass rounded-lg p-3 text-xs space-y-1.5 pointer-events-none select-none">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'var(--accent-blue)' }}></div>
                        <span className="text-theme-secondary">Earth</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-gray-400"></div>
                        <span className="text-theme-secondary">Moon</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: currentMethod === 'classical' ? 'var(--accent-purple)' : 'var(--accent-cyan)' }}></div>
                        <span className="text-theme-secondary capitalize">{currentMethod} Trajectory</span>
                    </div>
                </div>

                {/* Zoom/Pan Instructions */}
                <div className="absolute bottom-4 right-4 z-10 text-xs text-theme-muted bg-theme-secondary/80 px-2 py-1 rounded pointer-events-none">
                    Drag to rotate • Scroll to zoom • Right-click to pan
                </div>
            </div>

            {/* Bottom Panel - Wrapper */}
            <div className="h-64 border-t border-theme bg-theme-secondary flex flex-col z-20">
                {/* Tabs & Toolbar */}
                <div className="flex items-center justify-between border-b border-theme bg-theme-secondary pr-2">
                    <div className="flex overflow-x-auto">
                        <TabButton
                            active={activeTab === 'schedule'}
                            onClick={() => setActiveTab('schedule')}
                            icon={<Activity className="w-4 h-4" />}
                            label="Thrust Schedule"
                        />
                        <TabButton
                            active={activeTab === 'history'}
                            onClick={() => setActiveTab('history')}
                            icon={<History className="w-4 h-4" />}
                            label="Iteration History"
                        />
                        <TabButton
                            active={activeTab === 'metrics'}
                            onClick={() => setActiveTab('metrics')}
                            icon={<BarChart3 className="w-4 h-4" />}
                            label="Metrics"
                        />
                        <TabButton
                            active={activeTab === 'runs'}
                            onClick={() => setActiveTab('runs')}
                            icon={<Database className="w-4 h-4" />}
                            label="Run Manager"
                        />
                        <TabButton
                            active={activeTab === 'compare'}
                            onClick={() => setActiveTab('compare')}
                            icon={<TrendingDown className="w-4 h-4" />}
                            label="Comparison"
                            badge={selectedRuns.length > 0 ? selectedRuns.length : undefined}
                        />
                    </div>

                    <div className="flex items-center gap-2">
                        {/* Export Button */}
                        {selectedRuns.length > 0 && (
                            <button
                                onClick={() => setIsExportModalOpen(true)}
                                className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded transition-colors text-white"
                                style={{ backgroundColor: 'var(--accent-blue)' }}
                            >
                                <Download className="w-3.5 h-3.5" />
                                Export ({selectedRuns.length})
                            </button>
                        )}

                        {/* Lunar Analysis Button */}
                        <button
                            onClick={() => setIsLunarPanelOpen(!isLunarPanelOpen)}
                            className={`flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded hover:bg-theme-elevated transition-colors ${isLunarPanelOpen ? 'text-theme-primary bg-theme-tertiary' : 'text-theme-muted'}`}
                        >
                            <Moon className="w-3.5 h-3.5" />
                            {isLunarPanelOpen ? 'Close' : 'Lunar Analysis'}
                        </button>
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 relative overflow-hidden p-0 bg-theme-primary">
                    {activeTab === 'schedule' && <ScheduleVisualization />}
                    {activeTab === 'history' && <IterationHistory />}
                    {activeTab === 'metrics' && <MetricsPanel />}
                    {activeTab === 'runs' && <RunManager />}
                    {activeTab === 'compare' && <ComparisonDashboard />}
                </div>
            </div>

            {/* Export Modal */}
            <ExportModal
                isOpen={isExportModalOpen}
                onClose={() => setIsExportModalOpen(false)}
            />
        </main>
    )
}

// Helper Component: Tab Button
interface TabButtonProps {
    active: boolean
    onClick: () => void
    icon: React.ReactNode
    label: string
    badge?: number
}

function TabButton({ active, onClick, icon, label, badge }: TabButtonProps) {
    return (
        <button
            onClick={onClick}
            className={`px-4 py-2 text-xs font-semibold uppercase tracking-wider flex items-center gap-2 transition-colors whitespace-nowrap ${
                active
                    ? 'bg-theme-tertiary text-theme-primary border-r border-l border-theme'
                    : 'text-theme-muted hover:text-theme-primary hover:bg-theme-elevated'
            }`}
        >
            {icon}
            {label}
            {badge !== undefined && badge > 0 && (
                <span
                    className="px-1.5 py-0.5 rounded-full text-xs font-bold text-white"
                    style={{ backgroundColor: 'var(--accent-blue)' }}
                >
                    {badge}
                </span>
            )}
        </button>
    )
}
