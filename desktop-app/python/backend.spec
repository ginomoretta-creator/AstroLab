# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for bundling the ASL-Sandbox Python backend.

This creates a standalone executable that includes:
- Python runtime
- JAX, NumPy, and other dependencies
- The core physics module
- Backend server

Usage:
    pip install pyinstaller
    pyinstaller backend.spec
"""

import sys
import os
from PyInstaller.utils.hooks import collect_all

# Get the project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(SPECPATH)))
backend_dir = os.path.join(project_root, 'THRML-Sandbox', 'backend')
core_dir = os.path.join(project_root, 'core')

# Collect all data for JAX (needed for JIT compilation)
jax_datas, jax_binaries, jax_hiddenimports = collect_all('jax')
jaxlib_datas, jaxlib_binaries, jaxlib_hiddenimports = collect_all('jaxlib')

# Hidden imports
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
    'starlette',
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
] + jax_hiddenimports + jaxlib_hiddenimports

# Data files
datas = [
    (core_dir, 'core'),
    (backend_dir, 'backend'),
] + jax_datas + jaxlib_datas

# Binary files
binaries = jax_binaries + jaxlib_binaries

a = Analysis(
    [os.path.join(backend_dir, 'server.py')],
    pathex=[project_root, backend_dir, core_dir],
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
