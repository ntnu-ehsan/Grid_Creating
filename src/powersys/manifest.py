"""What a run wrote into its output folder, and the guard that reads it back.

``run.name`` is free text and ``run.overwrite`` defaults to true, so two runs
at different spatial levels can be pointed at the same folder and the second
silently replaces the first. The CLI derives the folder name from the level
(see :func:`powersys.config._retarget_run_name`), which covers ``--level``, but
nothing stops a config file from being edited to a new level while keeping its
old ``run.name``. Every run therefore drops a small manifest, and the next run
refuses to overwrite a folder whose manifest records a different level.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Config, ConfigError

MANIFEST_NAME = "run.json"


class OutputConflictError(ConfigError):
    """The target folder holds the results of a different run."""


def manifest_path(out_dir: Path) -> Path:
    return Path(out_dir) / MANIFEST_NAME


def write_manifest(cfg: Config, extra: Optional[Dict[str, Any]] = None) -> Path:
    """Record what this run produced. Written last, so a crash leaves the
    previous manifest in place rather than claiming the folder."""
    payload = {
        "name": cfg.run.name,
        "level": cfg.spatial.level,
        "countries": list(cfg.countries),
        "years": list(cfg.years),
        "model": cfg.iamc.model,
        "scenario": cfg.iamc.scenario,
        "written": dt.datetime.now().isoformat(timespec="seconds"),
    }
    payload.update(extra or {})

    path = manifest_path(cfg.run.dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_manifest(out_dir: Path) -> Optional[Dict[str, Any]]:
    """The folder's manifest, or None when there is none to read.

    A folder written before manifests existed, or one left by a crashed run,
    reads as None - the caller treats that as "nothing to protect".
    """
    path = manifest_path(out_dir)
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return loaded if isinstance(loaded, dict) else None


def guard_output_dir(cfg: Config, force: bool = False) -> None:
    """Refuse to write over a run that is not the one being repeated.

    Raises :class:`OutputConflictError` when the folder holds a different
    spatial level, or when ``run.overwrite`` is false and the folder is
    already populated. ``force`` skips both checks.
    """
    if force:
        return

    out_dir = cfg.run.dir
    previous = read_manifest(out_dir)

    if previous is not None:
        previous_level = str(previous.get("level", "")).lower()
        if previous_level and previous_level != cfg.spatial.level:
            raise OutputConflictError(
                f"{out_dir} holds a '{previous_level}' run "
                f"(written {previous.get('written', 'unknown')}), but this run is "
                f"'{cfg.spatial.level}'. Overwriting would destroy it.\n"
                f"  Use --name <folder> to write elsewhere, or --force to overwrite."
            )

    if not cfg.run.overwrite and out_dir.exists() and any(out_dir.iterdir()):
        raise OutputConflictError(
            f"{out_dir} already exists and run.overwrite is false.\n"
            f"  Use --name <folder> to write elsewhere, or --force to overwrite."
        )
