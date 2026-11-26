"""Peak picking for 1D spectra."""
from __future__ import annotations

from dataclasses import dataclass

from scipy.signal import find_peaks

from nmrproc.baseline import baseline_noise
from nmrproc.data import Spectrum


@dataclass
class Peak:
    index: int  # nearest data point
    ppm: float  # position refined by parabolic interpolation
    hz: float  # position in Hz (ppm * spectrometer frequency)
    height: float  # intensity at the refined position


def pick_peaks(spectrum: Spectrum, snr=6.0, threshold=None, min_rel_height=0.0,
               ppm_range=None) -> list[Peak]:
    """Find peaks in the real part of the spectrum, ordered from high to low ppm.

    A point is a peak if it is a local maximum that is taller than ``threshold`` and rises at
    least ``threshold`` above its surroundings. By default the
    threshold is ``snr`` times the noise level measured in the signal-free regions. ``min_rel_height`` (a fraction of the
    tallest peak) can be used to hide small peaks, and ``ppm_range=(low, high)`` limits the
    search to part of the spectrum. Positions are refined to better than the point spacing.
    """
    y = spectrum.real
    if threshold is None:
        threshold = snr * baseline_noise(y)
    if min_rel_height:
        threshold = max(threshold, min_rel_height * float(y.max()))
    # Requiring prominence as well as height rejects noise bumps riding on the tails of big peaks.
    idx, _ = find_peaks(y, height=threshold, prominence=threshold)
    peaks = []
    dppm = float(spectrum.ppm[1] - spectrum.ppm[0])
    for i in idx:
        if i == 0 or i == len(y) - 1:
            continue
        ppm_i = float(spectrum.ppm[i])
        if ppm_range is not None and not (min(ppm_range) <= ppm_i <= max(ppm_range)):
            continue
        a, b, c = y[i - 1], y[i], y[i + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        height = float(b - 0.25 * (a - c) * delta)
        ppm = ppm_i + delta * dppm
        peaks.append(Peak(index=int(i), ppm=float(ppm), hz=float(ppm * spectrum.sf_mhz),
                          height=height))
    return peaks
