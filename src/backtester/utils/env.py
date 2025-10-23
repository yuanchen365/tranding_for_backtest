from __future__ import annotations

from pathlib import Path
from typing import Dict


def load_env_file(path: Path | str = ".env") -> Dict[str, str]:
    """
    Minimal .env loader.

    Reads KEY=VALUE pairs, ignoring blank lines and comments. Quoted values are
    unwrapped. Parsed variables are injected into os.environ.
    """
    from os import environ

    env_path = Path(path)
    if not env_path.exists():
        return {}

    loaded: Dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        environ[key] = value
        loaded[key] = value
    return loaded
