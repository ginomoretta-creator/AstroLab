# AstroLab

Welcome to the AstroLab project! This repository contains the source code for the application.

## Getting Started for Developers

To collaborate on this project, you'll need to set up your development environment. Since we don't commit the compiled executables (like `.exe` files) to GitHub, you will run the application directly from the source code.

### Prerequisites

- **Node.js**: [Download and install](https://nodejs.org/)
- **Python**: [Download and install](https://www.python.org/)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/ginomoretta-creator/AstroLab.git
    cd AstroLab
    ```

2.  **Frontend Setup**:
    ```bash
    cd THRML-Sandbox/frontend
    # or the appropriate directory for your frontend
    npm install
    ```

3.  **Backend Setup**:
    ```bash
    cd THRML-Sandbox/backend
    # Create a virtual environment (optional but recommended)
    python -m venv .venv
    # Activate source code (Windows)
    .\.venv\Scripts\Activate
    # Install dependencies
    pip install -r requirements.txt
    ```

### Running the Application

**Desktop App (Electron)**:
This is likely the version you want to run if you are looking for the standalone application.
```bash
cd desktop-app
npm install
npm run electron:dev
```

**Web Sandbox (Frontend)**:
```bash
cd THRML-Sandbox/frontend
npm run dev
```

**Backend (Python Server)**:
The desktop app usually manages the backend, but if you need to run it manually:
```bash
cd THRML-Sandbox/backend
# or
python Launcher.py
```

## Collaborating with Antigravity

If you are using Antigravity, simply open this folder and ask the agent to "Start the development server". It will handle running the commands for you!
