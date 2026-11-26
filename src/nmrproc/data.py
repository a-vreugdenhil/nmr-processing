"""Plain data containers shared by the I/O, processing and GUI layers (no GUI imports)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FID:
    """A 1D time-domain signal (digital filter already removed) plus its acquisition parameters."""

    data: np.ndarray  # complex
    sw_hz: float  # spectral width, Hz
    bf_mhz: float  # base frequency (BF1), MHz
    sfo_mhz: float  # frequency at the carrier (SFO1), MHz
    o1_hz: float  # carrier offset from the base frequency, Hz
    nucleus: str = ""
    solvent: str = ""
    title: str = ""
    path: str = ""


@dataclass
class Spectrum:
    """A 1D frequency-domain spectrum, ordered like TopSpin: highest ppm first (left to right)."""

    data: np.ndarray  # complex
    ppm: np.ndarray  # chemical-shift axis, same length as data, decreasing
    sw_hz: float
    sf_mhz: float  # frequency of 0 ppm, MHz
    nucleus: str = ""
    info: dict = field(default_factory=dict)

    @property
    def real(self) -> np.ndarray:
        return self.data.real
