import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Trail } from '@react-three/drei';
import * as THREE from 'three';

const L_STAR = 384400.0; // km
const R_EARTH = 6378.0 / L_STAR;
const R_MOON = 1737.0 / L_STAR;
const MU = 0.01215;

function Earth() {
    return (
        <mesh position={[-MU, 0, 0]}> {/* Correct position: -MU */}
            {/* Actually, let's keep 1 unit = L_STAR. Earth is at -MU. */}
            <sphereGeometry args={[R_EARTH, 32, 32]} />
            <meshStandardMaterial color="#2563eb" emissive="#1e40af" emissiveIntensity={0.5} />
            <pointLight position={[2, 0, 2]} intensity={1.5} />
        </mesh>
    );
}

function Moon() {
    return (
        <mesh position={[(1 - MU), 0, 0]}>
            <sphereGeometry args={[R_MOON, 32, 32]} />
            <meshStandardMaterial color="#94a3b8" />
        </mesh>
    );
}

function TrajectoryLine({ points, color, opacity = 1, width = 1 }) {
    const lineGeometry = useMemo(() => {
        if (!points || points.length === 0) return null;
        const curve = new THREE.CatmullRomCurve3(
            points.map(p => new THREE.Vector3(p[0], p[1], 0))
        );
        return new THREE.TubeGeometry(curve, points.length, width * 0.002, 8, false);
    }, [points, width]);

    if (!lineGeometry) return null;

    return (
        <mesh geometry={lineGeometry}>
            <meshBasicMaterial color={color} transparent opacity={opacity} />
        </mesh>
    );
}

function Scene({ trajectories, bestTrajectory }) {
    return (
        <>
            <ambientLight intensity={0.1} />
            <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

            <Earth />
            <Moon />

            {/* Ghost Trajectories */}
            {trajectories.map((traj, i) => (
                <TrajectoryLine
                    key={i}
                    points={traj}
                    color="#64748b"
                    opacity={0.1}
                    width={0.5}
                />
            ))}

            {/* Best Trajectory */}
            {bestTrajectory && (
                <TrajectoryLine
                    points={bestTrajectory}
                    color="#00f3ff"
                    opacity={1}
                    width={2}
                />
            )}

            <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
        </>
    );
}

export function SimulationCanvas({ trajectories, bestTrajectory }) {
    return (
        <div className="w-full h-screen bg-void-900">
            <Canvas camera={{ position: [0, 0, 2.5], fov: 45 }}>
                <Scene trajectories={trajectories} bestTrajectory={bestTrajectory} />
            </Canvas>
        </div>
    );
}
