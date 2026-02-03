"""
Run Storage System for Trajectory Analysis
===========================================

File-based storage system for simulation runs using JSON format.

Features:
- Store up to 50 runs with auto-cleanup
- JSON file storage for easy sharing
- Run metadata and full iteration history
- Filtering and search capabilities

Author: ASL-Sandbox Team
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import uuid


class RunStorage:
    """
    Manages storage of simulation runs as JSON files.

    Stores runs in `backend/runs/` directory with automatic cleanup
    when exceeding max_runs limit (default 50).
    """

    def __init__(self, storage_dir: Optional[str] = None, max_runs: int = 50):
        """
        Initialize run storage.

        Args:
            storage_dir: Directory to store runs (default: backend/runs/)
            max_runs: Maximum number of runs to keep (default: 50)
        """
        if storage_dir is None:
            # Default: backend/runs/
            backend_dir = Path(__file__).parent
            storage_dir = backend_dir / "runs"

        self.storage_dir = Path(storage_dir)
        self.max_runs = max_runs

        # Create directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, run_data: Dict, run_id: Optional[str] = None) -> str:
        """
        Save a simulation run to disk.

        Args:
            run_data: Dictionary containing run information:
                - method: "classical" or "hybrid"
                - params: simulation parameters
                - iterations: list of iteration results
                - finalMetrics: comprehensive metrics
                - timestamp: ISO-8601 timestamp (optional, will be added)
            run_id: Optional custom run ID (default: auto-generated UUID)

        Returns:
            run_id: Unique identifier for the saved run
        """
        # Generate run ID if not provided
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]  # Short UUID

        # Add metadata
        if 'timestamp' not in run_data:
            run_data['timestamp'] = datetime.utcnow().isoformat() + 'Z'

        run_data['run_id'] = run_id

        # Filename format: run_{timestamp}_{method}_{id}.json
        timestamp_str = run_data['timestamp'].replace(':', '-').replace('.', '-')
        method = run_data.get('method', 'unknown')
        filename = f"run_{timestamp_str}_{method}_{run_id}.json"

        filepath = self.storage_dir / filename

        # Save to disk
        with open(filepath, 'w') as f:
            json.dump(run_data, f, indent=2)

        print(f"[STORAGE] Saved run {run_id} to {filename}")

        # Cleanup old runs if needed
        self._cleanup_old_runs()

        return run_id

    def load_run(self, run_id: str) -> Optional[Dict]:
        """
        Load a specific run by ID.

        Args:
            run_id: Run identifier

        Returns:
            Run data dictionary, or None if not found
        """
        # Find file with matching run_id
        for filepath in self.storage_dir.glob(f"run_*_{run_id}.json"):
            with open(filepath, 'r') as f:
                return json.load(f)

        return None

    def list_runs(
        self,
        method: Optional[str] = None,
        limit: Optional[int] = None,
        sort_by: str = 'timestamp',
        ascending: bool = False
    ) -> List[Dict]:
        """
        List all stored runs with metadata.

        Args:
            method: Filter by method ("classical", "hybrid", or None for all)
            limit: Maximum number of runs to return (None for all)
            sort_by: Sort key ('timestamp', 'cost', 'method')
            ascending: Sort order (False = newest/best first)

        Returns:
            List of run metadata dictionaries
        """
        runs = []

        # Load all run files
        for filepath in self.storage_dir.glob("run_*.json"):
            try:
                with open(filepath, 'r') as f:
                    run_data = json.load(f)

                # Filter by method if specified
                if method is not None and run_data.get('method') != method:
                    continue

                # Extract metadata
                metadata = {
                    'run_id': run_data.get('run_id'),
                    'timestamp': run_data.get('timestamp'),
                    'method': run_data.get('method'),
                    'params': run_data.get('params', {}),
                    'finalCost': run_data.get('finalMetrics', {}).get('final_cost', None),
                    'deltaV': run_data.get('finalMetrics', {}).get('delta_v_analytical', None),
                    'timeOfFlight': run_data.get('finalMetrics', {}).get('total_time_days', None),
                    'captured': run_data.get('finalMetrics', {}).get('captured', False),
                    'num_iterations': len(run_data.get('iterations', [])),
                    'filename': filepath.name
                }

                runs.append(metadata)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"[STORAGE WARNING] Failed to load {filepath.name}: {e}")
                continue

        # Sort runs
        if sort_by == 'timestamp':
            runs.sort(key=lambda x: x.get('timestamp', ''), reverse=not ascending)
        elif sort_by == 'cost':
            runs.sort(key=lambda x: x.get('finalCost', float('inf')), reverse=not ascending)
        elif sort_by == 'method':
            runs.sort(key=lambda x: x.get('method', ''), reverse=not ascending)

        # Apply limit
        if limit is not None:
            runs = runs[:limit]

        return runs

    def delete_run(self, run_id: str) -> bool:
        """
        Delete a specific run by ID.

        Args:
            run_id: Run identifier

        Returns:
            True if deleted, False if not found
        """
        # Find file with matching run_id
        for filepath in self.storage_dir.glob(f"run_*_{run_id}.json"):
            filepath.unlink()
            print(f"[STORAGE] Deleted run {run_id}")
            return True

        print(f"[STORAGE WARNING] Run {run_id} not found")
        return False

    def _cleanup_old_runs(self):
        """
        Remove oldest runs if count exceeds max_runs.

        Keeps most recent runs based on file modification time.
        """
        run_files = list(self.storage_dir.glob("run_*.json"))

        if len(run_files) <= self.max_runs:
            return

        # Sort by modification time (oldest first)
        run_files.sort(key=lambda f: f.stat().st_mtime)

        # Delete oldest runs
        num_to_delete = len(run_files) - self.max_runs
        for filepath in run_files[:num_to_delete]:
            filepath.unlink()
            print(f"[STORAGE] Auto-cleanup: Deleted {filepath.name}")

    def get_storage_stats(self) -> Dict:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage info
        """
        run_files = list(self.storage_dir.glob("run_*.json"))

        total_size = sum(f.stat().st_size for f in run_files)

        # Count by method
        methods = {'classical': 0, 'hybrid': 0, 'other': 0}
        for filepath in run_files:
            if 'classical' in filepath.name:
                methods['classical'] += 1
            elif 'hybrid' in filepath.name:
                methods['hybrid'] += 1
            else:
                methods['other'] += 1

        return {
            'total_runs': len(run_files),
            'total_size_mb': total_size / (1024 * 1024),
            'storage_dir': str(self.storage_dir),
            'max_runs': self.max_runs,
            'methods': methods
        }


# =============================================================================
# Comparison Utilities
# =============================================================================

def compare_runs(run_ids: List[str], storage: RunStorage) -> Dict:
    """
    Generate comparison statistics for multiple runs.

    Args:
        run_ids: List of run IDs to compare
        storage: RunStorage instance

    Returns:
        Dictionary with comparison data
    """
    runs = []

    # Load all runs
    for run_id in run_ids:
        run_data = storage.load_run(run_id)
        if run_data is not None:
            runs.append(run_data)

    if len(runs) < 2:
        return {'error': 'Need at least 2 runs for comparison'}

    # Extract metrics for comparison
    comparison = {
        'runs': [],
        'summary': {}
    }

    for run in runs:
        metrics = run.get('finalMetrics', {})

        run_summary = {
            'run_id': run.get('run_id'),
            'method': run.get('method'),
            'timestamp': run.get('timestamp'),
            'delta_v': metrics.get('delta_v_analytical'),
            'time_of_flight': metrics.get('total_time_days'),
            'fuel_consumed': metrics.get('fuel_consumed'),
            'final_cost': metrics.get('final_cost'),
            'captured': metrics.get('captured', False),
            'num_iterations': len(run.get('iterations', []))
        }

        comparison['runs'].append(run_summary)

    # Calculate relative performance
    # Find best values across all runs
    best_delta_v = min(r['delta_v'] for r in comparison['runs'] if r['delta_v'] is not None)
    best_time = min(r['time_of_flight'] for r in comparison['runs'] if r['time_of_flight'] is not None)
    best_cost = min(r['final_cost'] for r in comparison['runs'] if r['final_cost'] is not None)

    comparison['summary'] = {
        'best_delta_v': best_delta_v,
        'best_time_of_flight': best_time,
        'best_cost': best_cost,
        'num_runs_compared': len(runs)
    }

    # Add relative performance (% difference from best)
    for run_summary in comparison['runs']:
        if run_summary['delta_v'] is not None:
            run_summary['delta_v_relative'] = ((run_summary['delta_v'] - best_delta_v) / best_delta_v) * 100
        if run_summary['time_of_flight'] is not None:
            run_summary['time_relative'] = ((run_summary['time_of_flight'] - best_time) / best_time) * 100
        if run_summary['final_cost'] is not None:
            run_summary['cost_relative'] = ((run_summary['final_cost'] - best_cost) / best_cost) * 100

    return comparison


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'RunStorage',
    'compare_runs'
]
