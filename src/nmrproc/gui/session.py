"""State and processing pipeline behind the interactive 1D view (no Qt imports, so testable)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nmrproc.baseline import correct_baseline
from nmrproc.data import FID, Spectrum
from nmrproc.integration import Integral, find_integral_regions, integrate, normalize_integrals
from nmrproc.io import load_fid_1d
from nmrproc.peaks import Peak, pick_peaks
from nmrproc.phasing import auto_phase
from nmrproc.processing import default_si, process_1d, set_reference

ZERO_FILL_CHOICES = (1, 2, 4, 8)  # multiples of the next power of two above the FID length
MIN_REGION_POINTS = 3  # an integration range must span at least this many data points

# Reference peaks commonly used for chemical-shift referencing: (label, nucleus, ppm).
COMMON_REFERENCES = (
    ("TMS / DSS", "1H", 0.00),
    ("CDCl3 (residual CHCl3)", "1H", 7.26),
    ("DMSO-d6", "1H", 2.50),
    ("Acetone-d6", "1H", 2.05),
    ("Methanol-d4", "1H", 3.31),
    ("Benzene-d6", "1H", 7.16),
    ("D2O (HDO)", "1H", 4.79),
    ("TMS", "13C", 0.00),
    ("CDCl3", "13C", 77.16),
    ("DMSO-d6", "13C", 39.52),
    ("Acetone-d6", "13C", 29.84),
    ("Methanol-d4", "13C", 49.00),
    ("Benzene-d6", "13C", 128.06),
    ("85% H3PO4", "31P", 0.00),
)


def wrap_angle(angle: float) -> float:
    """Wrap an angle in degrees into the range [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


@dataclass
class Settings:
    window: str = "exponential"  # 'none', 'exponential', 'gaussian', 'sine' or 'qsine'
    lb_hz: float = 0.3
    gb: float = 0.1  # Gaussian maximum as a fraction of the acquisition time
    ssb: float = 2.0  # sine-bell start (TopSpin's SSB)
    zero_fill: int = 2
    p0: float = 0.0  # zero-order phase at the pivot, degrees
    p1: float = 0.0  # first-order phase across the full spectral width, degrees
    pivot: float = 0.0  # fraction of the spectrum (0 = left edge, 1 = right edge)
    baseline: bool = False
    baseline_order: int = 5
    reference_shift: float = 0.0  # ppm added to the axis by referencing
    show_peaks: bool = True
    peak_snr: float = 6.0


@dataclass
class Region:
    """An integration range, in the spectrum's current ppm scale."""

    start_ppm: float  # higher ppm edge
    end_ppm: float  # lower ppm edge
    reference: bool = False  # True for the integral that is set to ``reference_value``


class Session:
    """Holds the loaded FID, the processing settings and the integration ranges."""

    def __init__(self):
        self.fid: FID | None = None
        self.settings = Settings()
        self.regions: list[Region] = []
        self.reference_value = 1.0

    @property
    def loaded(self) -> bool:
        return self.fid is not None

    def load(self, path) -> FID:
        """Load a 1D Bruker experiment folder and reset all settings and integrals."""
        self.fid = load_fid_1d(path)
        self.settings = Settings()
        self.regions = []
        self.reference_value = 1.0
        self.settings.pivot = self._tallest_peak_fraction()
        return self.fid

    @property
    def si(self) -> int:
        return (default_si(len(self.fid.data)) // 2) * self.settings.zero_fill

    def set_window(self, window: str) -> None:
        """Choose the window function, adjusting line broadening to a sensible sign."""
        s = self.settings
        s.window = window
        if window == "gaussian" and s.lb_hz >= 0:
            s.lb_hz = -1.0
        elif window == "exponential" and s.lb_hz < 0:
            s.lb_hz = 0.3

    def _process(self, p0=0.0, p1=0.0, pivot=0.0) -> Spectrum:
        s = self.settings
        return process_1d(self.fid, lb_hz=s.lb_hz, si=self.si, p0=p0, p1=p1, pivot=pivot,
                          window=s.window, gb=s.gb, ssb=s.ssb)

    def _unphased(self) -> Spectrum:
        return self._process()

    def _tallest_peak_fraction(self) -> float:
        data = self._unphased().data
        return float(np.argmax(np.abs(data))) / len(data)

    def spectrum(self) -> Spectrum:
        """The processed spectrum for the current settings."""
        s = self.settings
        spec = self._process(s.p0, s.p1, s.pivot)
        if s.baseline:
            spec = correct_baseline(spec, order=s.baseline_order)
        if s.reference_shift:
            spec = set_reference(spec, 0.0, s.reference_shift)
        return spec

    def all_peaks(self, spectrum: Spectrum) -> list[Peak]:
        """Peaks above the current signal-to-noise threshold (whether or not labels are shown)."""
        return pick_peaks(spectrum, snr=self.settings.peak_snr)

    def peaks(self, spectrum: Spectrum) -> list[Peak]:
        """Peaks to label on screen: none if labels are switched off."""
        return self.all_peaks(spectrum) if self.settings.show_peaks else []

    # ---- phase -------------------------------------------------------------------------

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

    # ---- chemical-shift referencing ----------------------------------------------------

    def nearest_peak(self, spectrum: Spectrum, ppm: float, window_ppm: float = 0.15):
        """The tallest peak within ``window_ppm`` of ``ppm``, or None if there is none."""
        found = pick_peaks(spectrum, snr=3.0, ppm_range=(ppm - window_ppm, ppm + window_ppm))
        return max(found, key=lambda p: p.height) if found else None

    def reference_to(self, observed_ppm: float, target_ppm: float) -> None:
        """Shift the axis so the peak now at ``observed_ppm`` sits at ``target_ppm``."""
        shift = target_ppm - observed_ppm
        self._shift_axis(shift)

    def clear_reference(self) -> None:
        self._shift_axis(-self.settings.reference_shift)

    def _shift_axis(self, shift: float) -> None:
        self.settings.reference_shift += shift
        if abs(self.settings.reference_shift) < 1e-12:
            self.settings.reference_shift = 0.0
        for region in self.regions:  # integration ranges follow the peaks they enclose
            region.start_ppm += shift
            region.end_ppm += shift

    # ---- integration ranges ------------------------------------------------------------

    def add_region(self, spectrum: Spectrum, ppm_a: float, ppm_b: float) -> Region | None:
        """Add an integration range (clipped to the spectrum); None if it is too narrow."""
        low, high = float(spectrum.ppm.min()), float(spectrum.ppm.max())
        start = min(max(max(ppm_a, ppm_b), low), high)
        end = min(max(min(ppm_a, ppm_b), low), high)
        spacing = abs(float(spectrum.ppm[1] - spectrum.ppm[0]))
        if start - end < MIN_REGION_POINTS * spacing:
            return None
        region = Region(start, end)
        self.regions.append(region)
        self.regions.sort(key=lambda r: -r.start_ppm)
        return region

    def remove_region(self, index: int) -> None:
        del self.regions[index]

    def clear_regions(self) -> None:
        self.regions = []

    def auto_regions(self, spectrum: Spectrum) -> None:
        """Replace the integration ranges with ones suggested from the signals present."""
        self.regions = [Region(a, b) for a, b in find_integral_regions(spectrum)]

    def set_reference_region(self, index: int, value: float) -> None:
        """Make integral ``index`` the reference, with the given value (e.g. 3.00 for a CH3)."""
        for i, region in enumerate(self.regions):
            region.reference = i == index
        self.reference_value = float(value)

    def integrals(self, spectrum: Spectrum) -> list[Integral]:
        """Integrals of all ranges, scaled so the reference range has the reference value.

        With no range marked as the reference, the first (highest-ppm) range is used.
        """
        if not self.regions:
            return []
        ranges = [(r.start_ppm, r.end_ppm) for r in self.regions]
        results = integrate(spectrum, ranges)
        which = next((i for i, r in enumerate(self.regions) if r.reference), 0)
        if results[which].area == 0:
            return results  # nothing to scale by; values stay at zero
        return normalize_integrals(results, which=which, value=self.reference_value)
