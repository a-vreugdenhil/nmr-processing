import numpy as np

from nmrproc.processing import (
    default_si,
    exponential_apodization,
    fourier_transform,
    phase_correct,
    ppm_axis,
    zero_fill,
)


def make_fid(freq_hz=100.0, sw_hz=1000.0, n=1024):
    t = np.arange(n) / sw_hz
    return np.exp(2j * np.pi * freq_hz * t) * np.exp(-t / 0.2)


def test_peak_position():
    # Highest frequency comes first and the carrier (0 Hz) sits at index n // 2.
    sw, n = 1000.0, 1024
    spec = fourier_transform(make_fid(100.0, sw, n))
    peak = int(np.argmax(spec.real))
    assert peak == n // 2 - round(100.0 / (sw / n))


def test_carrier_at_center():
    n = 1024
    spec = fourier_transform(make_fid(0.0, 1000.0, n))
    assert int(np.argmax(spec.real)) == n // 2


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


def test_phase_pivot_leaves_pivot_point_unchanged_by_p1():
    spec = np.ones(100, dtype=complex)
    out = phase_correct(spec, 0.0, 90.0, pivot=0.5)
    assert np.isclose(out[50], 1.0)


def test_ppm_axis_spacing_and_direction():
    ppm = ppm_axis(1000, 4000.0, 400.0, 10.0)
    assert ppm[0] == 10.0
    assert np.all(np.diff(ppm) < 0)
    assert np.isclose(ppm[0] - ppm[-1], 10.0 * 999 / 1000)  # SW is 10 ppm across n points


def test_default_si():
    assert default_si(32768) == 65536
    assert default_si(32690) == 65536
    assert default_si(1000) == 2048
