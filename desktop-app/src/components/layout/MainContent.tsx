import { Canvas } from '@react-three/fiber'
import { OrbitControls, Stars, PerspectiveCamera } from '@react-three/drei'
import { useThemeStore } from '../../stores/themeStore'
import TrajectoryScene from '../visualization/TrajectoryScene'
import StatsPanel from '../analysis/StatsPanel'
import ScheduleVisualization from '../analysis/ScheduleVisualization'
import { ErrorBoundary } from '../common/ErrorBoundary'

export default function MainContent() {
    const { theme } = useThemeStore()

    // Adjust star visibility based on theme
    const showStars = theme === 'dark'
    const backgroundColor = theme === 'dark' ? '#0a0a0f' : '#f8fafc'

    return (
        <main className="flex-1 flex flex-col overflow-hidden bg-theme-primary">
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

                {/* Legend */}
                <div className="absolute bottom-4 left-4 z-10 glass rounded-lg p-3 text-xs space-y-1.5">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'var(--accent-blue)' }}></div>
                        <span className="text-theme-secondary">Earth</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-gray-400"></div>
                        <span className="text-theme-secondary">Moon</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'var(--accent-purple)' }}></div>
                        <span className="text-theme-secondary">THRML Trajectory</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'var(--accent-cyan)' }}></div>
                        <span className="text-theme-secondary">Quantum Trajectory</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'var(--accent-orange)' }}></div>
                        <span className="text-theme-secondary">Random Trajectory</span>
                    </div>
                </div>

                {/* Zoom/Pan Instructions */}
                <div className="absolute bottom-4 right-4 z-10 text-xs text-theme-muted bg-theme-secondary/80 px-2 py-1 rounded">
                    Drag to rotate • Scroll to zoom • Right-click to pan
                </div>
            </div>

            {/* Bottom Panel - Schedule Visualization */}
            <div className="h-32 border-t border-theme bg-theme-secondary p-4">
                <ScheduleVisualization />
            </div>
        </main>
    )
}
