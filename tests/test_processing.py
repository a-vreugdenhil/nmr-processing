import numpy as np

from nmrproc.processing import (
    exponential_apodization,
    fourier_transform,
    phase_correct,
    zero_fill,
)


def make_fid(freq_hz=100.0, sw_hz=1000.0, n=1024):
    t = np.arange(n) / sw_hz
    return np.exp(2j * np.pi * freq_hz * t) * np.exp(-t / 0.2)


def test_peak_position():
    sw, n = 1000.0, 1024
    spec = fourier_transform(make_fid(100.0, sw, n))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1 / sw))
    assert abs(freqs[np.argmax(spec.real)] - 100.0) < sw / n


def test_zero_fill_length():
    assert len(zero_fill(make_fid(), 4096)) == 4096


def test_apodization_decays():
    fid = np.ones(100, dtype=complex)
    out = exponential_apodization(fid, 5.0, 1000.0)
    assert out[-1].real < out[0].real


def test_phase_roundtrip():
    spec = fourier_transform(make_fid())
    back = phase_correct(phase_correct(spec, 30, 10), -30, -10)
    assert np.allclose(spec, back)
