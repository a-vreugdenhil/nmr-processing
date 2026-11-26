"""Reprocess the real Bruker test data sets and compare with TopSpin's own spectra."""
from pathlib import Path

import numpy as np
import pytest

from nmrproc.io import load_fid_1d, load_processed_1d, read_proc_params
from nmrproc.processing import process_1d, set_reference

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
