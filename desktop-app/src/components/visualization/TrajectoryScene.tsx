import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Line } from '@react-three/drei'
import * as THREE from 'three'
import { useSimulationStore } from '../../stores/simulationStore'
import { useThemeStore } from '../../stores/themeStore'

// Constants (normalized units - Earth at origin, Moon at ~1.0)
const MU = 0.01215
const EARTH_RADIUS = 0.0166 * 5  // Scaled for visibility
const MOON_RADIUS = 0.0045 * 5
const MOON_POSITION: [number, number, number] = [1 - MU, 0, 0]

export default function TrajectoryScene() {
    const { results, currentMethod, trajectoryHistory } = useSimulationStore()
    const { theme } = useThemeStore()

    return (
        <group>
            {/* Earth */}
            <Earth theme={theme} />

            {/* Moon */}
            <Moon theme={theme} />

            {/* Orbit reference circle */}
            <OrbitReference theme={theme} />

            {/* Historical trajectories (faded) */}
            {trajectoryHistory.slice(-20).map((traj, index) => (
                <TrajectoryPath
                    key={index}
                    points={traj.points}
                    color={getMethodColor(traj.method)}
                    opacity={0.2 + (index / 20) * 0.5}
                />
            ))}

            {/* Best trajectory highlight */}
            {results[currentMethod]?.bestTrajectory && (
                <TrajectoryPath
                    points={results[currentMethod]!.bestTrajectory.map(p => [p[0], p[1]] as [number, number])}
                    color={getMethodColor(currentMethod)}
                    opacity={1}
                    lineWidth={3}
                    glow
                />
            )}

            {/* Grid for reference */}
            <gridHelper
                args={[4, 20, theme === 'dark' ? '#333' : '#ccc', theme === 'dark' ? '#222' : '#eee']}
                rotation={[Math.PI / 2, 0, 0]}
            />
        </group>
    )
}

interface PlanetProps {
    theme: 'dark' | 'light'
}

function Earth({ theme }: PlanetProps) {
    const meshRef = useRef<THREE.Mesh>(null)

    useFrame((_, delta) => {
        if (meshRef.current) {
            meshRef.current.rotation.y += delta * 0.3
        }
    })

    const earthColor = theme === 'dark' ? '#3b82f6' : '#2563eb'
    const emissiveColor = theme === 'dark' ? '#1e40af' : '#1d4ed8'
    const glowColor = theme === 'dark' ? '#60a5fa' : '#93c5fd'

    return (
        <group position={[-MU, 0, 0]}>
            {/* Earth sphere */}
            <mesh ref={meshRef}>
                <sphereGeometry args={[EARTH_RADIUS, 32, 32]} />
                <meshStandardMaterial
                    color={earthColor}
                    emissive={emissiveColor}
                    emissiveIntensity={theme === 'dark' ? 0.3 : 0.1}
                />
            </mesh>

            {/* Atmosphere glow */}
            <mesh scale={1.15}>
                <sphereGeometry args={[EARTH_RADIUS, 32, 32]} />
                <meshBasicMaterial
                    color={glowColor}
                    transparent
                    opacity={theme === 'dark' ? 0.15 : 0.08}
                    side={THREE.BackSide}
                />
            </mesh>
        </group>
    )
}

function Moon({ theme }: PlanetProps) {
    const meshRef = useRef<THREE.Mesh>(null)

    useFrame((_, delta) => {
        if (meshRef.current) {
            meshRef.current.rotation.y += delta * 0.1
        }
    })

    const moonColor = theme === 'dark' ? '#94a3b8' : '#64748b'

    return (
        <group position={MOON_POSITION}>
            <mesh ref={meshRef}>
                <sphereGeometry args={[MOON_RADIUS, 32, 32]} />
                <meshStandardMaterial
                    color={moonColor}
                    emissive={moonColor}
                    emissiveIntensity={theme === 'dark' ? 0.1 : 0.05}
                />
            </mesh>
        </group>
    )
}

function OrbitReference({ theme }: PlanetProps) {
    const points = useMemo(() => {
        const pts: [number, number, number][] = []
        for (let i = 0; i <= 64; i++) {
            const angle = (i / 64) * Math.PI * 2
            pts.push([
                -MU + Math.cos(angle) * 0.5,
                Math.sin(angle) * 0.5,
                0
            ])
        }
        return pts
    }, [])

    const lineColor = theme === 'dark' ? '#444' : '#ccc'

    return (
        <Line
            points={points}
            color={lineColor}
            lineWidth={1}
            transparent
            opacity={0.3}
        />
    )
}

interface TrajectoryPathProps {
    points: [number, number][]
    color: string
    opacity?: number
    lineWidth?: number
    glow?: boolean
}

function TrajectoryPath({ points, color, opacity = 1, lineWidth = 2, glow }: TrajectoryPathProps) {
    const points3D = useMemo(() => {
        if (!Array.isArray(points) || points.length < 2) return []

        // Filter out invalid points (NaN, Infinity)
        return points
            .filter(p => Array.isArray(p) && p.length === 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]))
            .map(p => [p[0], p[1], 0] as [number, number, number])
    }, [points])

    if (points3D.length < 2) return null

    return (
        <group>
            <Line
                points={points3D}
                color={color}
                lineWidth={lineWidth}
                transparent
                opacity={opacity}
            />
            {glow && (
                <Line
                    points={points3D}
                    color={color}
                    lineWidth={lineWidth * 2}
                    transparent
                    opacity={opacity * 0.3}
                />
            )}
        </group>
    )
}

function getMethodColor(method: string): string {
    switch (method) {
        case 'thrml': return '#8b5cf6'
        case 'quantum': return '#06b6d4'
        case 'random': return '#f59e0b'
        default: return '#ffffff'
    }
}
