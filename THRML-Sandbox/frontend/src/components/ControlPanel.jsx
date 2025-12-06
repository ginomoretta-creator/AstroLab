import React from 'react';
import { Play, Square, Activity, Settings } from 'lucide-react';

export function ControlPanel({ params, setParams, onSimulate, isSimulating }) {
    const handleChange = (key, value) => {
        setParams(prev => ({ ...prev, [key]: parseFloat(value) }));
    };

    return (
        <div className="fixed top-4 left-4 w-80 bg-gray-900 border-2 border-white rounded-xl p-6 text-white z-[100]">
            <div className="flex items-center gap-2 mb-6">
                <Activity className="text-neon-blue" size={24} />
                <h1 className="text-xl font-bold tracking-wider">THRML<span className="text-gray-400 text-sm font-normal ml-2">SANDBOX</span></h1>
            </div>

            <div className="space-y-4">
                {/* Method Selection */}
                <div className="flex bg-black/40 rounded-lg p-1">
                    {['thrml', 'classical'].map((m) => (
                        <button
                            key={m}
                            onClick={() => setParams(prev => ({ ...prev, method: m }))}
                            className={`flex-1 py-2 rounded-md text-sm font-medium transition-all ${params.method === m
                                ? 'bg-neon-blue/20 text-neon-blue shadow-[0_0_10px_rgba(0,243,255,0.2)]'
                                : 'text-gray-400 hover:text-white'
                                }`}
                        >
                            {m.toUpperCase()}
                        </button>
                    ))}
                </div>

                {/* Sliders */}
                <div className="space-y-1">
                    <label className="text-xs text-gray-400 uppercase">Mass (kg)</label>
                    <input
                        type="range" min="100" max="5000" step="100"
                        value={params.mass}
                        onChange={(e) => handleChange('mass', e.target.value)}
                        className="w-full accent-neon-blue h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="text-right text-sm font-mono text-neon-blue">{params.mass}</div>
                </div>

                <div className="space-y-1">
                    <label className="text-xs text-gray-400 uppercase">Thrust (N)</label>
                    <input
                        type="range" min="1" max="100" step="1"
                        value={params.thrust}
                        onChange={(e) => handleChange('thrust', e.target.value)}
                        className="w-full accent-neon-blue h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="text-right text-sm font-mono text-neon-blue">{params.thrust}</div>
                </div>

                <div className="space-y-1">
                    <label className="text-xs text-gray-400 uppercase">Coupling (Smoothness)</label>
                    <input
                        type="range" min="0" max="2" step="0.1"
                        value={params.coupling_strength}
                        onChange={(e) => handleChange('coupling_strength', e.target.value)}
                        className="w-full accent-neon-purple h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="text-right text-sm font-mono text-neon-purple">{params.coupling_strength}</div>
                </div>

                {/* Action Button */}
                <button
                    onClick={onSimulate}
                    className={`w-full mt-4 py-3 rounded-lg font-bold flex items-center justify-center gap-2 transition-all ${isSimulating
                        ? 'bg-red-500/20 text-red-400 border border-red-500/50 hover:bg-red-500/30'
                        : 'bg-neon-blue/20 text-neon-blue border border-neon-blue/50 hover:bg-neon-blue/30 hover:shadow-[0_0_15px_rgba(0,243,255,0.3)]'
                        }`}
                >
                    {isSimulating ? <><Square size={18} /> STOP</> : <><Play size={18} /> SIMULATE</>}
                </button>
            </div>
        </div>
    );
}
