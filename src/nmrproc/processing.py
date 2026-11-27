"""GUI-independent 1D processing functions (pure numpy).

Conventions (chosen to match TopSpin so results can be compared directly):

* Spectra are stored highest chemical shift first (left to right as displayed).
* Phase angles are TopSpin's PHC0 / PHC1 values in degrees. The first-order phase is applied
  linearly across the whole spectral width, starting at the left edge (x = 0) unless a
  ``pivot`` (0 = left edge, 1 = right edge) is given.
"""
from __future__ import annotations

import numpy as np

from nmrproc.data import FID, Spectrum


def exponential_apodization(fid, lb_hz, sw_hz):
    """Multiply the FID by exp(-pi*lb*t). lb_hz = line broadening, sw_hz = spectral width."""
    t = np.arange(len(fid)) / sw_hz
    return fid * np.exp(-np.pi * lb_hz * t)


WINDOWS = ("none", "exponential", "gaussian", "sine", "qsine")


def gaussian_apodization(fid, lb_hz, gb, sw_hz):
    """Lorentz-to-Gauss resolution enhancement (TopSpin's GM), with TopSpin's LB and GB meaning.

    ``lb_hz`` must be negative: it sets a rising exponential that cancels the natural line
    width. A Gaussian then brings the signal back down, with its maximum at the fraction ``gb``
    (0 < gb <= 1) of the acquisition time. The window equals 1 at t = 0.
    """
    if lb_hz >= 0:
        raise ValueError("Gaussian apodization needs a negative line broadening, e.g. -1 Hz")
    if not 0 < gb <= 1:
        raise ValueError("The Gaussian maximum (GB) must be greater than 0 and at most 1")
    t = np.arange(len(fid)) / sw_hz
    aq = len(fid) / sw_hz
    growth = -np.pi * lb_hz
    return fid * np.exp(growth * t - growth / (2 * gb * aq) * t**2)


def sine_apodization(fid, ssb=2.0, power=1):
    """Sine-bell window (TopSpin's SINE; ``power=2`` gives QSINE, the squared sine bell).

    The bell runs over the whole FID. ``ssb`` sets where it starts: the window begins at a
    phase of pi/ssb, so ssb = 2 gives a cosine-like decay from 1 to 0, and larger values start
    nearer zero. Values below 2 (TopSpin's 0 and 1) give a pure sine, starting at zero.
    """
    fraction = np.arange(len(fid)) / len(fid)
    start = np.pi / ssb if ssb >= 2 else 0.0
    return fid * np.sin(start + (np.pi - start) * fraction) ** power


def apodize(fid, window, sw_hz, lb_hz=0.0, gb=0.1, ssb=2.0):
    """Apply the named window function ('none', 'exponential', 'gaussian', 'sine', 'qsine')."""
    if window == "none":
        return fid
    if window == "exponential":
        return exponential_apodization(fid, lb_hz, sw_hz)
    if window == "gaussian":
        return gaussian_apodization(fid, lb_hz, gb, sw_hz)
    if window == "sine":
        return sine_apodization(fid, ssb, power=1)
    if window == "qsine":
        return sine_apodization(fid, ssb, power=2)
    raise ValueError(f"Unknown window function {window!r}; choose from {WINDOWS}")


def zero_fill(fid, n_points):
    """Pad the FID with zeros up to n_points (no-op if already that long)."""
    if n_points <= len(fid):
        return fid
    return np.concatenate([fid, np.zeros(n_points - len(fid), dtype=fid.dtype)])


def fourier_transform(fid):
    """Complex FFT returned highest frequency first, with the carrier at index n // 2.

    This is the orientation (and one-point alignment) used by TopSpin, so spectra line up
    point for point with Bruker's own processed data.
    """
    spectrum = np.fft.fftshift(np.fft.fft(fid))[::-1]
    return np.roll(spectrum, 1)


def phase_correct(spectrum, p0_deg=0.0, p1_deg=0.0, pivot=0.0):
    """Apply zero- and first-order phase correction using TopSpin's PHC0/PHC1 convention."""
    n = len(spectrum)
    x = np.arange(n) / n - pivot
    return spectrum * np.exp(-1j * np.deg2rad(p0_deg + p1_deg * x))


def ppm_axis(n, sw_hz, sf_mhz, left_ppm):
    """Chemical-shift axis of n points spanning sw_hz, starting at left_ppm and decreasing."""
    return left_ppm - (sw_hz / sf_mhz / n) * np.arange(n)


def default_si(n_fid):
    """Default number of spectrum points: twice the next power of two above the FID length."""
    return 2 * (1 << max(int(n_fid) - 1, 1).bit_length())


def process_1d(fid: FID, lb_hz=0.3, si=None, p0=0.0, p1=0.0, pivot=0.0,
               window="exponential", gb=0.1, ssb=2.0) -> Spectrum:
    """Apodization, zero fill, Fourier transform and phasing of a 1D FID.

    ``window`` is one of 'none', 'exponential' (uses ``lb_hz``), 'gaussian' (``lb_hz``, negative,
    and ``gb``), 'sine' or 'qsine' (``ssb``).

    The chemical-shift axis is unreferenced: 0 ppm is the base frequency (BF1), as in TopSpin
    before ``sref`` is used.
    """
    si = int(si) if si else default_si(len(fid.data))
    data = apodize(fid.data, window, fid.sw_hz, lb_hz=lb_hz, gb=gb, ssb=ssb)
    data = zero_fill(data, si)[:si]
    spec = phase_correct(fourier_transform(data), p0, p1, pivot)
    left_ppm = (fid.o1_hz + fid.sw_hz / 2) / fid.bf_mhz
    return Spectrum(
        data=spec,
        ppm=ppm_axis(si, fid.sw_hz, fid.bf_mhz, left_ppm),
        sw_hz=fid.sw_hz,
        sf_mhz=fid.bf_mhz,
        nucleus=fid.nucleus,
        info={"window": window, "lb_hz": lb_hz, "gb": gb, "ssb": ssb, "si": si, "p0": p0,
              "p1": p1, "pivot": pivot},
    )


def set_reference(spectrum: Spectrum, observed_ppm, target_ppm) -> Spectrum:
    """Shift the ppm axis so that the peak seen at observed_ppm sits at target_ppm."""
    shift = target_ppm - observed_ppm
    info = dict(spectrum.info)
    info["reference_shift_ppm"] = info.get("reference_shift_ppm", 0.0) + shift
    return Spectrum(
        data=spectrum.data,
        ppm=spectrum.ppm + shift,
        sw_hz=spectrum.sw_hz,
        sf_mhz=spectrum.sf_mhz,
        nucleus=spectrum.nucleus,
        info=info,
    )


def estimate_noise(y):
    """Robust noise level (standard deviation) of a spectrum, ignoring signals and slow baseline.

    Uses the median absolute deviation of point-to-point differences, which signals and gentle
    baseline curvature barely affect.
    """
    d = np.diff(np.asarray(y, dtype=float))
    return float(np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2))


def phase_spectrum(spectrum: Spectrum, p0=0.0, p1=0.0, pivot=0.0) -> Spectrum:
    """Apply an additional phase correction to a Spectrum and keep a running total in ``info``.

    The totals (``info["p0"]``, ``info["p1"]``) are always expressed with the pivot at the left
    edge, i.e. in TopSpin's PHC0/PHC1 convention.
    """
    info = dict(spectrum.info)
    info["p0"] = info.get("p0", 0.0) + p0 - p1 * pivot
    info["p1"] = info.get("p1", 0.0) + p1
    info["pivot"] = 0.0
    return Spectrum(
        data=phase_correct(spectrum.data, p0, p1, pivot),
        ppm=spectrum.ppm,
        sw_hz=spectrum.sw_hz,
        sf_mhz=spectrum.sf_mhz,
        nucleus=spectrum.nucleus,
        info=info,
    )
