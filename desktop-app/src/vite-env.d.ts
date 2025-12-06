/// <reference types="vite/client" />

declare global {
    interface Window {
        electronAPI: {
            minimize: () => Promise<void>
            maximize: () => Promise<void>
            close: () => Promise<void>
            getBackendStatus: () => Promise<{ status: string; data?: any; error?: string }>
            getBackendPort: () => Promise<number>
            platform: string
        }
    }
}

export { }
