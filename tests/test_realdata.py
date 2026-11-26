"""Reprocess the real Bruker test data sets and compare with TopSpin's own spectra."""
from pathlib import Path

import numpy as np
import pytest

from nmrproc.baseline import correct_baseline
from nmrproc.integration import integrate
from nmrproc.io import load_fid_1d, load_processed_1d, read_proc_params
from nmrproc.peaks import pick_peaks
from nmrproc.phasing import auto_phase
from nmrproc.processing import phase_spectrum, process_1d, set_reference

DATA = Path(__file__).resolve().parent.parent / "testdata"
EXPERIMENTS = [
    DATA / "triphenylphosphine-1H" / "10",
    DATA / "dppe-31P" / "10",
]

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="testdata/ folder not present")


def reprocess(path):
    fid = load_fid_1d(path)
    procs = read_proc_params(path)
    ours = process_1d(fid, lb_hz=procs["LB"], si=procs["SI"], p0=procs["PHC0"], p1=procs["PHC1"])
    return fid, procs, ours, load_processed_1d(path)


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.parent.name)
def test_fid_parameters(path):
    fid = load_fid_1d(path)
    assert fid.data.ndim == 1 and np.iscomplexobj(fid.data)
    assert fid.sw_hz > 0 and fid.bf_mhz > 0
    assert fid.nucleus in {"1H", "31P"}


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.parent.name)
def test_ppm_axis_matches_topspin(path):
    _, _, ours, ref = reprocess(path)
    assert len(ours.ppm) == len(ref.ppm)
    # Same left edge and spacing as TopSpin (the data were not referenced in TopSpin).
    assert abs(ours.ppm[0] - ref.ppm[0]) < 1e-3
    assert abs(ours.ppm[-1] - ref.ppm[-1]) < 1e-3


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.parent.name)
def test_spectrum_matches_topspin_shape(path):
    _, _, ours, ref = reprocess(path)
    a, b = ours.real, ref.real
    assert len(a) == len(b)
    assert np.corrcoef(a, b)[0, 1] > 0.999
    scale = (a @ b) / (a @ a)
    rel_err = np.linalg.norm(b - scale * a) / np.linalg.norm(b)
    assert rel_err < 0.05


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.parent.name)
def test_tallest_peak_position_matches_topspin(path):
    _, _, ours, ref = reprocess(path)
    i_ours, i_ref = int(np.argmax(ours.real)), int(np.argmax(ref.real))
    assert abs(i_ours - i_ref) <= 1
    assert abs(ours.ppm[i_ours] - ref.ppm[i_ref]) < 0.01


def test_set_reference_moves_axis():
    path = EXPERIMENTS[0]
    _, _, ours, _ = reprocess(path)
    peak_ppm = ours.ppm[int(np.argmax(ours.real))]
    shifted = set_reference(ours, peak_ppm, 7.26)
    assert abs(shifted.ppm[int(np.argmax(shifted.real))] - 7.26) < 1e-9
    assert shifted.info["reference_shift_ppm"] == pytest.approx(7.26 - peak_ppm)


# --- automatic phasing, baseline correction, peak picking and integration -------------------

# Integration ranges (ppm, unreferenced axis) used to compare ratios with TopSpin's spectrum.
RANGES = {
    "triphenylphosphine-1H": [(7.79, 7.25), (1.80, 1.40)],  # aromatic region, water
    "dppe-31P": [(-4.5, -6.2), (-12.0, -13.2)],
}
RATIO_TOLERANCE = {"triphenylphosphine-1H": 0.06, "dppe-31P": 0.02}


def rel_error(ours, ref):
    scale = (ours @ ref) / (ours @ ours)
    return np.linalg.norm(ref - scale * ours) / np.linalg.norm(ref)


def auto_pipeline(path):
    """Unphased spectrum -> automatic phasing -> baseline correction (no TopSpin parameters)."""
    fid = load_fid_1d(path)
    procs = read_proc_params(path)
    raw = process_1d(fid, lb_hz=procs["LB"], si=procs["SI"])
    p0, p1 = auto_phase(raw)
    spec = correct_baseline(phase_spectrum(raw, p0, p1))
    return raw, procs, (p0, p1), spec, load_processed_1d(path)


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.parent.name)
def test_automatic_pipeline_matches_topspin(path):
    _, _, _, spec, ref = auto_pipeline(path)
    assert np.corrcoef(spec.real, ref.real)[0, 1] > 0.998
    assert rel_error(spec.real, ref.real) < 0.06


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.parent.name)
def test_auto_phase_close_to_topspin_at_tallest_peak(path):
    raw, procs, (p0, p1), _, _ = auto_pipeline(path)
    x = np.argmax(np.abs(raw.data)) / len(raw.data)
    ours = p0 + p1 * x
    theirs = procs["PHC0"] + procs["PHC1"] * x
    assert abs((ours - theirs + 180.0) % 360.0 - 180.0) < 15.0


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.parent.name)
def test_baseline_correction_improves_agreement(path):
    ours, ref = reprocess(path)[2:]  # phased with TopSpin's own values
    before = rel_error(ours.real, ref.real)
    after = rel_error(correct_baseline(ours).real, ref.real)
    assert after < 0.6 * before


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.parent.name)
def test_peak_picking_agrees_with_topspin(path):
    _, _, _, spec, ref = auto_pipeline(path)
    spacing = abs(ref.ppm[1] - ref.ppm[0])
    ours = pick_peaks(spec, min_rel_height=0.05)
    theirs = pick_peaks(ref, min_rel_height=0.05)
    assert len(ours) == len(theirs) >= 1
    for t in theirs:
        nearest = min(abs(o.ppm - t.ppm) for o in ours)
        assert nearest < 2 * spacing
        height_ours = min(ours, key=lambda o: abs(o.ppm - t.ppm)).height
        assert height_ours == pytest.approx(t.height, rel=0.1)


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.parent.name)
def test_integral_ratio_agrees_with_topspin(path):
    _, _, _, spec, ref = auto_pipeline(path)
    ranges = RANGES[path.parent.name]
    a = [i.area for i in integrate(spec, ranges)]
    b = [i.area for i in integrate(ref, ranges)]
    assert a[1] / a[0] == pytest.approx(b[1] / b[0], rel=RATIO_TOLERANCE[path.parent.name])
