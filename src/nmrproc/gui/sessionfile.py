"""Saving and reopening a processing session as a small JSON file (no Qt, so testable).

A session file records the processing choices, chemical-shift reference, integration ranges and
view, plus where the NMR data live. It never contains the data themselves.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import fields
from importlib import metadata
from pathlib import Path

from nmrproc.gui.session import ZERO_FILL_CHOICES, Region, Session, Settings
from nmrproc.processing import WINDOWS

FORMAT = "nmrproc-session"
VERSION = 1
EXTENSION = ".nmrproc"


class SessionFileError(Exception):
    """A session file could not be read or used; the message is written for the user."""


def _program_version() -> str:
    try:
        return metadata.version("nmrproc")
    except metadata.PackageNotFoundError:
        return "unknown"


def _relative_path(data_path: Path, session_path: Path):
    try:
        return Path(os.path.relpath(data_path, session_path.resolve().parent)).as_posix()
    except ValueError:  # different drives on Windows: no relative path exists
        return None


def save_session(session: Session, path, view=None) -> None:
    """Write the session to ``path`` (atomically, so a failure cannot corrupt an older file)."""
    if session.data_path is None:
        raise ValueError("There is no data set loaded to save a session for.")
    path = Path(path)
    payload = {
        "format": FORMAT,
        "version": VERSION,
        "program_version": _program_version(),
        "data_path": str(session.data_path),
        "data_path_relative": _relative_path(session.data_path, path),
        **session.snapshot(),
        "view": view,
    }
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def read_session_file(path) -> dict:
    """Read and check a session file, returning its contents as a dict."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise SessionFileError(f"The session file could not be opened: {exc}") from exc
    except ValueError as exc:  # includes JSON decoding and Unicode errors
        raise SessionFileError("This file is not a valid nmrproc session: it could not be "
                               "read as session data.") from exc
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise SessionFileError("This file is not an nmrproc session file.")
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise SessionFileError("This session file has no valid version number.")
    if version > VERSION:
        raise SessionFileError(f"This session was saved by a newer version of nmrproc (session "
                               f"format {version}). Please update the program to open it.")
    return data


def resolve_data_path(data: dict, session_path) -> Path | None:
    """Find the data folder: first relative to the session file, then the saved full path."""
    candidates = []
    relative = data.get("data_path_relative")
    if isinstance(relative, str) and relative:
        candidates.append((Path(session_path).resolve().parent / relative).resolve())
    absolute = data.get("data_path")
    if isinstance(absolute, str) and absolute:
        candidates.append(Path(absolute))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _number(value, name, low=None, high=None, integer=False):
    ok = isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
    if ok and integer:
        ok = float(value).is_integer()
    if not ok or (low is not None and value < low) or (high is not None and value > high):
        raise SessionFileError(f"The session file has an invalid value for '{name}'.")
    return int(value) if integer else float(value)


def _checked_settings(saved: dict) -> Settings:
    """Settings from a saved dict: missing keys keep their defaults, unknown keys are ignored."""
    settings = Settings()
    for field in fields(Settings):
        if field.name not in saved:
            continue
        value, default = saved[field.name], getattr(settings, field.name)
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise SessionFileError(f"The session file has an invalid value for "
                                       f"'{field.name}'.")
        elif isinstance(default, str):
            if value not in WINDOWS:
                raise SessionFileError(f"The session file uses an unknown window function "
                                       f"'{value}'.")
        elif isinstance(default, int):
            value = _number(value, field.name, integer=True)
        else:
            value = _number(value, field.name)
        setattr(settings, field.name, value)
    if settings.zero_fill not in ZERO_FILL_CHOICES:
        raise SessionFileError("The session file has an invalid value for 'zero_fill'.")
    if not 0.0 <= settings.pivot <= 1.0:
        raise SessionFileError("The session file has an invalid value for 'pivot'.")
    if not 1 <= settings.baseline_order <= 12:
        raise SessionFileError("The session file has an invalid value for 'baseline_order'.")
    if not 0.0 < settings.gb <= 1.0 or not 0.0 <= settings.ssb <= 16.0:
        raise SessionFileError("The session file has an invalid window parameter.")
    if not -50.0 <= settings.lb_hz <= 50.0 or settings.peak_snr <= 0:
        raise SessionFileError("The session file has an invalid line broadening or threshold.")
    if settings.window == "gaussian" and settings.lb_hz >= 0:
        raise SessionFileError("The session file asks for a Gaussian window with a "
                               "non-negative line broadening, which is not valid.")
    return settings


def apply_session_data(session: Session, data: dict) -> None:
    """Put the saved processing choices, reference and integrals into a loaded session.

    Everything is checked before anything is changed, so a bad file leaves the session as it was.
    """
    if not session.loaded:
        raise ValueError("Load the data set before applying a session to it.")
    saved = data.get("settings", {})
    if not isinstance(saved, dict):
        raise SessionFileError("The session file's settings are not in the expected form.")
    settings = _checked_settings(saved)
    reference_value = _number(data.get("reference_value", 1.0), "reference_value", low=1e-9)
    regions = []
    for item in data.get("regions", []):
        if not isinstance(item, dict):
            raise SessionFileError("An integration range in the session file is not valid.")
        a = _number(item.get("start_ppm"), "start_ppm")
        b = _number(item.get("end_ppm"), "end_ppm")
        reference = item.get("reference", False)
        if not isinstance(reference, bool):
            raise SessionFileError("An integration range in the session file is not valid.")
        regions.append(Region(max(a, b), min(a, b), reference))
    regions.sort(key=lambda r: -r.start_ppm)
    seen = False
    for region in regions:  # at most one range may be the reference
        region.reference, seen = region.reference and not seen, seen or region.reference
    session.settings = settings
    session.regions = regions
    session.reference_value = reference_value
