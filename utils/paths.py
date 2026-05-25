from __future__ import annotations
import os
from pathlib import Path
_THIS_FILE = Path(__file__).resolve()
_DEFAULT_PROJECT_ROOT = _THIS_FILE.parent.parent

def get_project_root() -> Path:
    env = os.environ.get('TSS_PROJECT_ROOT')
    return Path(env).expanduser().resolve() if env else _DEFAULT_PROJECT_ROOT

def get_raw_dir(override: str | os.PathLike | None=None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get('TSS_RAW_DIR')
    if env:
        return Path(env).expanduser().resolve()
    return get_project_root() / 'dataset' / 'raw' / 'TTML-IDN'

def get_synthetic_dir(name: str | None=None, override: str | os.PathLike | None=None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get('TSS_SYNTHETIC_DIR')
    base = Path(env).expanduser().resolve() if env else get_project_root() / 'dataset' / 'synthetic'
    return base / name if name else base

def get_checkpoint_dir(*parts: str, override: str | os.PathLike | None=None) -> Path:
    if override:
        base = Path(override).expanduser().resolve()
    else:
        env = os.environ.get('TSS_CHECKPOINT_DIR')
        if env:
            base = Path(env).expanduser().resolve()
        else:
            base = get_project_root() / 'checkpoints'
    for p in parts:
        base = base / p
    return base
