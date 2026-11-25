"""GUI-independent 1D processing functions (pure numpy)."""
import numpy as np


def exponential_apodization(fid, lb_hz, sw_hz):
    """Multiply the FID by exp(-pi*lb*t). lb_hz = line broadening, sw_hz = spectral width."""
    t = np.arange(len(fid)) / sw_hz
    return fid * np.exp(-np.pi * lb_hz * t)


def zero_fill(fid, n_points):
    """Pad the FID with zeros up to n_points (no-op if already that long)."""
    if n_points <= len(fid):
        return fid
    return np.concatenate([fid, np.zeros(n_points - len(fid), dtype=fid.dtype)])


def fourier_transform(fid):
    """Complex FFT, shifted so frequency is centered on zero."""
    return np.fft.fftshift(np.fft.fft(fid))


def phase_correct(spectrum, p0_deg=0.0, p1_deg=0.0):
    """Apply zero- and first-order phase correction (degrees)."""
    n = len(spectrum)
    x = np.linspace(-0.5, 0.5, n)
    phase = np.deg2rad(p0_deg + p1_deg * x)
    return spectrum * np.exp(1j * phase)
