import React from 'react';

export function StatsOverlay({ metrics }) {
    return (
        <div className="fixed top-4 right-4 w-64 bg-gray-900 border-2 border-white rounded-xl p-4 text-white z-[100]">
            <h2 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-3">Live Telemetry</h2>

            <div className="grid grid-cols-2 gap-4">
                <div>
                    <div className="text-xs text-gray-500">ITERATION</div>
                    <div className="text-xl font-mono font-bold text-white">
                        {metrics.iteration}<span className="text-gray-600 text-sm">/{metrics.totalIterations}</span>
                    </div>
                </div>

                <div>
                    <div className="text-xs text-gray-500">BEST DIST</div>
                    <div className="text-xl font-mono font-bold text-neon-green">
                        {(metrics.bestCost * 384400).toFixed(0)}<span className="text-xs text-gray-500 ml-1">km</span>
                    </div>
                </div>
            </div>

            {/* Progress Bar */}
            <div className="mt-4 h-1 w-full bg-gray-800 rounded-full overflow-hidden">
                <div
                    className="h-full bg-neon-blue transition-all duration-300 ease-out"
                    style={{ width: `${(metrics.iteration / Math.max(1, metrics.totalIterations)) * 100}%` }}
                />
            </div>
        </div>
    );
}
