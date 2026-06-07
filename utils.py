"""Utility helpers for the project.

Contains helpers to ensure directory structure, find latest runs,
and simple file utilities used across modules.
"""
from pathlib import Path
import shutil
import time


ROOT = Path(__file__).parent


def ensure_dirs():
    """Ensure required folders exist.

    Creates `dataset/`, `runs/`, `hasil_prediksi/`, `laporan/`, and `input_images/` if missing.
    """
    dirs = [
        ROOT / 'dataset',
        ROOT / 'runs',
        ROOT / 'hasil_prediksi',
        ROOT / 'laporan',
        ROOT / 'input_images',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def latest_run_dir(project_root: Path = None):
    """Return the most recent run directory under `runs/`.

    Ultralytics saves runs under the `project` folder; this helper finds the
    newest directory inside `runs/` (by modification time).
    """
    if project_root is None:
        project_root = ROOT / 'runs'
    runs = [p for p in project_root.iterdir() if p.is_dir()]
    if not runs:
        return None
    runs_sorted = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
    return runs_sorted[0]


def copy_weights_from_run(run_dir: Path, target_root: Path = None):
    """Copy `weights/best.pt` and `weights/last.pt` from a run to `runs/` root.

    Returns tuple (best_path, last_path) pointing to copied files.
    """
    if target_root is None:
        target_root = ROOT / 'runs'
    weights_dir = run_dir / 'weights'
    best_src = weights_dir / 'best.pt'
    last_src = weights_dir / 'last.pt'
    ts = int(time.time())
    best_dst = target_root / f'best_{ts}.pt'
    last_dst = target_root / f'last_{ts}.pt'
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
    else:
        best_dst = None
    if last_src.exists():
        shutil.copy2(last_src, last_dst)
    else:
        last_dst = None
    return best_dst, last_dst
