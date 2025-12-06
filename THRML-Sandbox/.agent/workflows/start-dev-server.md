---
description: Start backend and frontend development servers
---

# Start Development Servers

This workflow starts both the backend API server and frontend development server for the Cislunar Trajectory Sandbox.

## Prerequisites

Ensure dependencies are installed:
- Python 3.8+ with `pip`
- Node.js 16+ with `npm` (for React frontend)

## Steps

### 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
cd ..
```

**Note**: THRML library is vendored in `thrml-main/`. It will be auto-imported via `sys.path` manipulation in the backend code.

### 2. Start Backend Server

// turbo
```bash
python backend/server.py
```

This will start the FastAPI server on `http://localhost:8000`.

**Verify**: Open `http://localhost:8000` in browser - should see `{"status": "online", "system": "Cislunar Sandbox Backend"}`

### 3. Choose Frontend

You have two frontend options:

#### Option A: Streamlit (Simple, currently working)

// turbo
```bash
streamlit run app.py
```

Streamlit will auto-open browser to `http://localhost:8501`.

#### Option B: React (Advanced, under development)

First-time setup:
```bash
cd frontend
npm install
```

Start dev server:
// turbo
```bash
cd frontend
npm run dev
```

React app will be available at `http://localhost:5173`.

### 4. Verify Integration

1. Open the frontend (Streamlit or React)
2. Adjust simulation parameters
3. Click "Simulate" or trigger auto-run
4. Verify:
   - Network request to `http://localhost:8000/simulate`
   - Streaming response with trajectory data
   - Real-time visualization updates

## Troubleshooting

### Backend won't start
- **Error**: `ModuleNotFoundError: No module named 'jax'`
  - **Fix**: `pip install jax jaxlib`
  
- **Error**: `ModuleNotFoundError: No module named 'thrml'`
  - **Info**: This is expected. The code will fall back to random sampling.
  - **Fix** (optional): `pip install -e thrml-main/` to properly install THRML

### Frontend won't start (Streamlit)
- **Error**: `ModuleNotFoundError: No module named 'streamlit'`
  - **Fix**: `pip install streamlit requests plotly`

### Frontend won't start (React)
- **Error**: `Cannot find module 'vite'`
  - **Fix**: `cd frontend && npm install`

### CORS Errors
- **Symptom**: Browser console shows "CORS policy blocked"
- **Fix**: Ensure backend `server.py` has `allow_origins=["*"]` in CORS middleware (already configured)

### No Response from API
- **Check**: Is backend running? Verify `http://localhost:8000` shows status message
- **Check**: Network tab in browser DevTools - request should show `200 OK` status
