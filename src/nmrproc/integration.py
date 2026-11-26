"""Integration of 1D spectra."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nmrproc.baseline import baseline_noise
from nmrproc.data import Spectrum


@dataclass
class Integral:
    start_ppm: float  # higher ppm edge
    end_ppm: float  # lower ppm edge
    area: float  # raw area (sum of intensity times point spacing in ppm)
    value: float = 0.0  # relative value after normalize_integrals()


def _bounds(spectrum: Spectrum, start_ppm, end_ppm):
    hi, lo = max(start_ppm, end_ppm), min(start_ppm, end_ppm)
    ppm = spectrum.ppm
    sel = np.where((ppm <= hi) & (ppm >= lo))[0]
    return hi, lo, sel


def integrate(spectrum: Spectrum, ranges) -> list[Integral]:
    """Integrate the real part over each (start_ppm, end_ppm) range.

    Ranges can be given in either order. Returns a list of Integral with raw areas; use
    ``normalize_integrals`` to scale them (for example so that one integral equals 1.00).
    """
    y = spectrum.real
    step = abs(float(spectrum.ppm[1] - spectrum.ppm[0]))
    out = []
    for start, end in ranges:
        hi, lo, sel = _bounds(spectrum, start, end)
        area = float(y[sel].sum() * step) if len(sel) else 0.0
        out.append(Integral(start_ppm=hi, end_ppm=lo, area=area))
    return out


def normalize_integrals(integrals: list[Integral], which=0, value=1.0) -> list[Integral]:
    """Scale all integrals so that integrals[which] has the given value (e.g. 3.00 for a CH3)."""
    ref = integrals[which].area
    if ref == 0:
        raise ValueError("The reference integral is zero and cannot be used to normalize")
    for integral in integrals:
        integral.value = integral.area / ref * value
    return integrals


def find_integral_regions(spectrum: Spectrum, snr=5.0, merge_hz=20.0, pad_hz=10.0):
    """Suggest integration ranges around the signals in the spectrum.

    Points higher than ``snr`` times the noise level are signal. Signals closer together than
    ``merge_hz`` are merged into one region (so multiplets stay whole), and each region is
    widened by ``pad_hz`` on both sides. Returns a list of (start_ppm, end_ppm), high to low ppm.
    """
    y = spectrum.real
    n = len(y)
    hz_per_point = spectrum.sw_hz / n
    hot = np.where(y > snr * baseline_noise(y))[0]
    if len(hot) == 0:
        return []
    gap = max(1, round(merge_hz / hz_per_point))
    pad = round(pad_hz / hz_per_point)
    breaks = np.where(np.diff(hot) > gap)[0]
    starts = np.concatenate([[hot[0]], hot[breaks + 1]])
    ends = np.concatenate([hot[breaks], [hot[-1]]])
    regions = []
    for s, e in zip(starts, ends):
        s, e = max(0, s - pad), min(n - 1, e + pad)
        regions.append((float(spectrum.ppm[s]), float(spectrum.ppm[e])))
    return regions


def integral_trace(spectrum: Spectrum, ranges) -> np.ndarray:
    """Cumulative integral curve for display: rises inside the ranges and is flat elsewhere."""
    y = spectrum.real
    masked = np.zeros_like(y)
    for start, end in ranges:
        _, _, sel = _bounds(spectrum, start, end)
        masked[sel] = y[sel]
    return np.cumsum(masked) * abs(float(spectrum.ppm[1] - spectrum.ppm[0]))
