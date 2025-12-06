"""
Simple launcher script for the bundled backend.
This is the entry point for PyInstaller.
"""
import sys
import os

# Set up paths for bundled modules
if getattr(sys, 'frozen', False):
    # Running as bundled executable
    bundle_dir = sys._MEIPASS
    os.environ['PYTHONPATH'] = bundle_dir
else:
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

# Add paths
sys.path.insert(0, bundle_dir)

# Import and run the server
import uvicorn
from server import app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    args = parser.parse_args()
    
    print(f"Starting ASL-Sandbox Backend on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
