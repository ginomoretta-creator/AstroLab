# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for bundling the ASL-Sandbox Python backend.

This creates a standalone executable that includes:
- Python runtime
- JAX, NumPy, and other dependencies
- The core physics module
- Backend server

Usage:
    cd desktop-app/python
    python -m PyInstaller backend.spec --noconfirm --clean
"""

import sys
import os
from PyInstaller.utils.hooks import collect_all

# ============================================================================
# HARDCODED PATHS - Edit these if your project moves
# ============================================================================
# This file is at: C:\Users\ginom\Desktop\ASL-Sandbox\desktop-app\python\backend.spec
# The project structure is:
#   ASL-Sandbox/
#     ├── core/                    <- core_dir
#     ├── THRML-Sandbox/backend/   <- backend_dir
#     └── desktop-app/python/      <- this spec file

# Use the spec file's directory to find project root
THIS_FILE = os.path.abspath(__file__ if '__file__' in dir() else 'backend.spec')
SPEC_DIR = os.path.dirname(THIS_FILE)
DESKTOP_APP_DIR = os.path.dirname(SPEC_DIR)
PROJECT_ROOT = os.path.dirname(DESKTOP_APP_DIR)

# Verify we found the right directory by checking for expected folders
if not os.path.exists(os.path.join(PROJECT_ROOT, 'core')):
    # Fallback: try common locations
    possible_roots = [
        r'C:\Users\ginom\Desktop\ASL-Sandbox',
        os.path.expanduser('~/Desktop/ASL-Sandbox'),
    ]
    for root in possible_roots:
        if os.path.exists(os.path.join(root, 'core')):
            PROJECT_ROOT = root
            break

backend_dir = os.path.join(PROJECT_ROOT, 'THRML-Sandbox', 'backend')
core_dir = os.path.join(PROJECT_ROOT, 'core')

print("=" * 60)
print("ASL-Sandbox Backend Build")
print("=" * 60)
print(f"Project Root:  {PROJECT_ROOT}")
print(f"Backend Dir:   {backend_dir}")
print(f"Core Dir:      {core_dir}")
print(f"Server.py:     {os.path.join(backend_dir, 'server.py')}")
print("=" * 60)

# Verify paths exist
assert os.path.exists(backend_dir), f"backend_dir not found: {backend_dir}"
assert os.path.exists(core_dir), f"core_dir not found: {core_dir}"
assert os.path.exists(os.path.join(backend_dir, 'server.py')), f"server.py not found"

# ============================================================================
# JAX collection (needed for JIT)
# ============================================================================
try:
    jax_datas, jax_binaries, jax_hiddenimports = collect_all('jax')
    jaxlib_datas, jaxlib_binaries, jaxlib_hiddenimports = collect_all('jaxlib')
except Exception as e:
    print(f"Warning: Could not collect JAX: {e}")
    jax_datas, jax_binaries, jax_hiddenimports = [], [], []
    jaxlib_datas, jaxlib_binaries, jaxlib_hiddenimports = [], [], []

# ============================================================================
# Hidden imports
# ============================================================================
hiddenimports = [
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'fastapi',
    'pydantic',
    'pydantic_core',
    'starlette',
    'starlette.routing',
    'starlette.responses',
    'starlette.middleware',
    'starlette.middleware.cors',
    'httptools',
    'watchfiles',
    'websockets',
    'jax',
    'jax.numpy',
    'jaxlib',
    'numpy',
    # Core modules
    'core',
    'core.constants',
    'core.physics_core',
    'core.energy_model',
    'core.classical_solver',
    # Backend modules
    'physics',
    'generative',
] + jax_hiddenimports + jaxlib_hiddenimports

# ============================================================================
# Data files
# ============================================================================
datas = [
    (core_dir, 'core'),
    (backend_dir, 'backend'),
] + jax_datas + jaxlib_datas

# ============================================================================
# Binary files
# ============================================================================
binaries = jax_binaries + jaxlib_binaries

# ============================================================================
# Analysis
# ============================================================================
a = Analysis(
    [os.path.join(backend_dir, 'server.py')],
    pathex=[PROJECT_ROOT, backend_dir, core_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'PIL',
        'cv2',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='asl-sandbox-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep console for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
