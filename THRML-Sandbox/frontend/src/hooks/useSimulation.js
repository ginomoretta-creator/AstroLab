import { useState, useRef, useCallback } from 'react';

const API_URL = "http://localhost:8000/simulate";

export function useSimulation() {
    const [isConnected, setIsConnected] = useState(false);
    const [isSimulating, setIsSimulating] = useState(false);
    const [trajectories, setTrajectories] = useState([]);
    const [bestTrajectory, setBestTrajectory] = useState(null);
    const [metrics, setMetrics] = useState({ iteration: 0, totalIterations: 0, bestCost: 0 });
    const abortControllerRef = useRef(null);

    const startSimulation = useCallback(async (params) => {
        // Cancel previous request if active
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        abortControllerRef.current = new AbortController();
        setIsSimulating(true);
        setTrajectories([]);
        setBestTrajectory(null);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params),
                signal: abortControllerRef.current.signal,
            });

            if (!response.ok) throw new Error(response.statusText);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);
                        if (data.error) {
                            console.error("Simulation error:", data.error);
                            continue;
                        }

                        // Update state with new chunk
                        // We only keep the latest batch for visualization to avoid memory issues
                        // or we could accumulate if we want trails.
                        // For now, let's just show the current iteration's data.
                        setTrajectories(data.trajectories);
                        setBestTrajectory(data.best_trajectory);
                        setMetrics({
                            iteration: data.iteration,
                            totalIterations: data.total_iterations,
                            bestCost: data.best_cost
                        });

                    } catch (e) {
                        console.error("Error parsing JSON chunk", e);
                    }
                }
            }
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error("Simulation failed:", error);
            }
        } finally {
            setIsSimulating(false);
            abortControllerRef.current = null;
        }
    }, []);

    const stopSimulation = useCallback(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
            setIsSimulating(false);
        }
    }, []);

    return {
        startSimulation,
        stopSimulation,
        isSimulating,
        trajectories,
        bestTrajectory,
        metrics
    };
}
