"""State and processing pipeline behind the interactive 1D view (no Qt imports, so testable)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nmrproc.baseline import correct_baseline
from nmrproc.data import FID, Spectrum
from nmrproc.io import load_fid_1d
from nmrproc.peaks import Peak, pick_peaks
from nmrproc.phasing import auto_phase
from nmrproc.processing import default_si, process_1d

ZERO_FILL_CHOICES = (1, 2, 4, 8)  # multiples of the next power of two above the FID length


def wrap_angle(angle: float) -> float:
    """Wrap an angle in degrees into the range [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


@dataclass
class Settings:
    lb_hz: float = 0.3
    zero_fill: int = 2
    p0: float = 0.0  # zero-order phase at the pivot, degrees
    p1: float = 0.0  # first-order phase across the full spectral width, degrees
    pivot: float = 0.0  # fraction of the spectrum (0 = left edge, 1 = right edge)
    baseline: bool = False
    baseline_order: int = 5
    show_peaks: bool = True
    peak_snr: float = 6.0


class Session:
    """Holds the loaded FID and the current processing settings, and computes the spectrum."""

    def __init__(self):
        self.fid: FID | None = None
        self.settings = Settings()

    @property
    def loaded(self) -> bool:
        return self.fid is not None

    def load(self, path) -> FID:
        """Load a 1D Bruker experiment folder and reset all settings."""
        self.fid = load_fid_1d(path)
        self.settings = Settings()
        self.settings.pivot = self._tallest_peak_fraction()
        return self.fid

    @property
    def si(self) -> int:
        return (default_si(len(self.fid.data)) // 2) * self.settings.zero_fill

    def _unphased(self) -> Spectrum:
        s = self.settings
        return process_1d(self.fid, lb_hz=s.lb_hz, si=self.si)

    def _tallest_peak_fraction(self) -> float:
        data = self._unphased().data
        return float(np.argmax(np.abs(data))) / len(data)

    def spectrum(self) -> Spectrum:
        """The processed spectrum for the current settings."""
        s = self.settings
        spec = process_1d(self.fid, lb_hz=s.lb_hz, si=self.si, p0=s.p0, p1=s.p1, pivot=s.pivot)
        if s.baseline:
            spec = correct_baseline(spec, order=s.baseline_order)
        return spec

    def peaks(self, spectrum: Spectrum) -> list[Peak]:
        if not self.settings.show_peaks:
            return []
        return pick_peaks(spectrum, snr=self.settings.peak_snr)

    def topspin_phase(self) -> tuple[float, float]:
        """Current phase as TopSpin-style PHC0/PHC1 (pivot at the left edge)."""
        s = self.settings
        return s.p0 - s.p1 * s.pivot, s.p1

    def auto_phase(self) -> None:
        """Set the phase from the automatic (ACME) algorithm, keeping the current pivot."""
        p0, p1 = auto_phase(self._unphased())
        s = self.settings
        s.p1 = p1
        s.p0 = wrap_angle(p0 + p1 * s.pivot)

    def reset_phase(self) -> None:
        self.settings.p0 = 0.0
        self.settings.p1 = 0.0

    def set_pivot(self, fraction: float) -> None:
        """Move the pivot without changing the phase of the displayed spectrum."""
        s = self.settings
        fraction = min(max(float(fraction), 0.0), 1.0)
        s.p0 = wrap_angle(s.p0 + s.p1 * (fraction - s.pivot))
        s.pivot = fraction
