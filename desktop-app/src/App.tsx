import { useState, useEffect } from 'react'
import TitleBar from './components/layout/TitleBar'
import Sidebar from './components/layout/Sidebar'
import MainContent from './components/layout/MainContent'
import { useSimulationStore } from './stores/simulationStore'
import { initializeTheme } from './stores/themeStore'

function App() {
    const { setBackendStatus, setBackendPort } = useSimulationStore()
    const [isLoading, setIsLoading] = useState(true)

    useEffect(() => {
        // Initialize theme
        initializeTheme()

        // Check backend status on mount
        const checkBackend = async () => {
            try {
                const port = await window.electronAPI?.getBackendPort() || 8080
                setBackendPort(port)

                const status = await window.electronAPI?.getBackendStatus()
                setBackendStatus(status?.status === 'online' ? 'online' : 'offline')
            } catch (error) {
                setBackendStatus('offline')
            } finally {
                setIsLoading(false)
            }
        }

        // Initial check after delay (backend startup time)
        const timer = setTimeout(checkBackend, 1000)

        // Periodic check
        const interval = setInterval(checkBackend, 5000)

        return () => {
            clearTimeout(timer)
            clearInterval(interval)
        }
    }, [setBackendStatus, setBackendPort])

    if (isLoading) {
        return (
            <div className="h-screen bg-theme-primary flex flex-col">
                <TitleBar />
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center fade-in">
                        <div className="w-20 h-20 mx-auto mb-6 relative">
                            <div className="absolute inset-0 rounded-full border-4 border-theme-tertiary"></div>
                            <div className="absolute inset-0 rounded-full border-4 border-t-[var(--accent-blue)] animate-spin"></div>
                            <div className="absolute inset-3 rounded-full bg-gradient-to-br from-[var(--accent-blue)] to-[var(--accent-purple)] opacity-20"></div>
                        </div>
                        <h2 className="text-xl font-semibold text-theme-primary mb-2">Initializing ASL-Sandbox</h2>
                        <p className="text-theme-secondary">Starting physics backend...</p>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="h-screen bg-theme-primary flex flex-col overflow-hidden">
            <TitleBar />
            <div className="flex-1 flex overflow-hidden">
                <Sidebar />
                <MainContent />
            </div>
        </div>
    )
}

export default App
