"""Saving and reopening sessions (no Qt needed)."""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from nmrproc.export import spectrum_to_csv
from nmrproc.gui.session import Session
from nmrproc.gui.sessionfile import (
    FORMAT,
    VERSION,
    SessionFileError,
    apply_session_data,
    read_session_file,
    resolve_data_path,
    save_session,
)

DATA = Path(__file__).resolve().parent.parent / "testdata"
PATH = DATA / "triphenylphosphine-1H" / "10"

pytestmark = pytest.mark.skipif(not PATH.exists(), reason="testdata/ folder not present")


def worked_session(path=PATH):
    """A session with every kind of saved information changed from its default."""
    session = Session()
    session.load(path)
    session.set_window("gaussian")
    session.settings.gb = 0.2
    session.auto_phase()
    session.settings.baseline = True
    session.settings.baseline_order = 3
    session.settings.zero_fill = 4
    session.settings.peak_snr = 8.0
    spec = session.spectrum()
    session.reference_to(session.nearest_peak(spec, 7.284).ppm, 7.26)
    spec = session.spectrum()
    session.add_region(spec, 7.62, 7.20)
    session.add_region(spec, 1.75, 1.40)
    session.set_reference_region(1, 2.0)
    return session


def test_snapshot_changes_exactly_when_the_work_changes():
    session = Session()
    session.load(PATH)
    fresh = session.snapshot()
    assert session.snapshot() == fresh
    session.settings.p0 = 12.5
    assert session.snapshot() != fresh
    session.settings.p0 = 0.0
    assert session.snapshot() == fresh
    session.add_region(session.spectrum(), 7.5, 7.1)
    assert session.snapshot() != fresh
    session.clear_regions()
    assert session.snapshot() == fresh


def test_round_trip_restores_everything(tmp_path):
    original = worked_session()
    saved = tmp_path / "work.nmrproc"
    save_session(original, saved, view={"xlim": [7.7, 7.1]})
    data = read_session_file(saved)
    assert data["format"] == FORMAT and data["version"] == VERSION
    assert data["view"] == {"xlim": [7.7, 7.1]}
    restored = Session()
    restored.load(resolve_data_path(data, saved))
    apply_session_data(restored, data)
    assert restored.snapshot() == original.snapshot()
    assert np.allclose(restored.spectrum().data, original.spectrum().data)
    assert [i.value for i in restored.integrals(restored.spectrum())] == pytest.approx(
        [i.value for i in original.integrals(original.spectrum())])


def test_saved_file_is_readable_text_with_a_relative_path(tmp_path):
    # The data are copied next to the session file: on Windows the repository and the temporary
    # folder can be on different drives, where no relative path exists (tested separately below).
    data_dir = tmp_path / "data" / "10"
    shutil.copytree(PATH, data_dir)
    saved = tmp_path / "work.nmrproc"
    save_session(worked_session(data_dir), saved)
    data = json.loads(saved.read_text(encoding="utf-8"))
    assert data["settings"]["window"] == "gaussian" and data["reference_value"] == 2.0
    assert Path(data["data_path"]) == data_dir.resolve()
    assert (saved.parent / data["data_path_relative"]).resolve() == data_dir.resolve()
    assert not (tmp_path / "work.nmrproc.tmp").exists()  # the temporary file is gone


def test_data_on_another_drive_is_recorded_by_full_path_only(tmp_path, monkeypatch):
    """On Windows, os.path.relpath fails between drives (e.g. C: and D:); saving must still work."""
    def different_drives(*args, **kwargs):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(os.path, "relpath", different_drives)
    saved = tmp_path / "work.nmrproc"
    save_session(worked_session(), saved)
    data = read_session_file(saved)
    assert data["data_path_relative"] is None
    assert resolve_data_path(data, saved) == PATH.resolve()  # found by the full path
    restored = Session()
    restored.load(resolve_data_path(data, saved))
    apply_session_data(restored, data)
    assert restored.snapshot() == worked_session().snapshot()


def test_session_and_data_can_be_moved_together(tmp_path):
    folder = tmp_path / "student-work"
    shutil.copytree(PATH, folder / "data" / "10")
    session = worked_session(folder / "data" / "10")
    save_session(session, folder / "lab1.nmrproc")
    moved = tmp_path / "moved-to-usb"
    shutil.move(str(folder), str(moved))  # the whole folder moves; the old absolute path is gone
    data = read_session_file(moved / "lab1.nmrproc")
    found = resolve_data_path(data, moved / "lab1.nmrproc")
    assert found == (moved / "data" / "10").resolve()


def test_absolute_path_is_the_fallback_and_missing_data_is_reported(tmp_path):
    saved = tmp_path / "work.nmrproc"
    save_session(worked_session(), saved)
    data = read_session_file(saved)
    data["data_path_relative"] = "no/such/folder"
    assert resolve_data_path(data, saved) == PATH.resolve()  # falls back to the full path
    data["data_path"] = str(tmp_path / "gone")
    assert resolve_data_path(data, saved) is None
    assert resolve_data_path({}, saved) is None


def test_saving_needs_loaded_data(tmp_path):
    with pytest.raises(ValueError):
        save_session(Session(), tmp_path / "x.nmrproc")


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def valid_payload(tmp_path):
    saved = tmp_path / "ok.nmrproc"
    save_session(worked_session(), saved)
    return json.loads(saved.read_text(encoding="utf-8"))


@pytest.mark.parametrize("content", ["", "not json at all", "[1, 2, 3]", '{"hello": 1}'])
def test_files_that_are_not_sessions_are_rejected_politely(tmp_path, content):
    bad = tmp_path / "bad.nmrproc"
    bad.write_text(content, encoding="utf-8")
    with pytest.raises(SessionFileError):
        read_session_file(bad)


def test_binary_and_missing_files_are_rejected_politely(tmp_path):
    binary = tmp_path / "binary.nmrproc"
    binary.write_bytes(b"\xff\xfe\x00\x01garbage")
    with pytest.raises(SessionFileError):
        read_session_file(binary)
    with pytest.raises(SessionFileError):
        read_session_file(tmp_path / "does-not-exist.nmrproc")


def test_a_newer_session_format_asks_for_a_program_update(tmp_path):
    payload = valid_payload(tmp_path)
    payload["version"] = VERSION + 1
    with pytest.raises(SessionFileError, match="newer version"):
        read_session_file(write_json(tmp_path / "new.nmrproc", payload))
    payload["version"] = "one"
    with pytest.raises(SessionFileError, match="version"):
        read_session_file(write_json(tmp_path / "odd.nmrproc", payload))


@pytest.mark.parametrize("change", [
    {"window": "hamming"}, {"window": 3}, {"zero_fill": 3}, {"pivot": 1.5}, {"baseline_order": 0},
    {"baseline_order": 2.5}, {"lb_hz": "big"}, {"lb_hz": float("nan")}, {"lb_hz": 500.0},
    {"gb": 0.0}, {"ssb": -1.0}, {"peak_snr": 0.0}, {"baseline": "yes"}, {"p0": True},
    {"window": "gaussian", "lb_hz": 1.0},
])
def test_invalid_settings_are_rejected_without_touching_the_session(tmp_path, change):
    payload = valid_payload(tmp_path)
    payload["settings"].update(change)
    session = Session()
    session.load(PATH)
    before = session.snapshot()
    with pytest.raises(SessionFileError):
        apply_session_data(session, payload)
    assert session.snapshot() == before  # nothing was half-applied


@pytest.mark.parametrize("regions", [
    [{"start_ppm": "a", "end_ppm": 1.0}], ["not a dict"], [{"start_ppm": 5.0}],
    [{"start_ppm": 5.0, "end_ppm": 4.0, "reference": "maybe"}],
])
def test_invalid_ranges_are_rejected(tmp_path, regions):
    payload = valid_payload(tmp_path)
    payload["regions"] = regions
    session = Session()
    session.load(PATH)
    with pytest.raises(SessionFileError):
        apply_session_data(session, payload)


def test_unknown_and_missing_keys_are_tolerated(tmp_path):
    payload = valid_payload(tmp_path)
    payload["settings"]["setting_from_the_future"] = 42
    del payload["settings"]["peak_snr"]
    payload["something_new"] = {"a": 1}
    del payload["reference_value"]
    session = Session()
    session.load(PATH)
    apply_session_data(session, payload)
    assert session.settings.peak_snr == 6.0 and session.reference_value == 1.0
    assert session.settings.window == "gaussian"  # everything else was applied


def test_reversed_ranges_are_put_right_and_only_one_reference_is_kept(tmp_path):
    payload = valid_payload(tmp_path)
    payload["regions"] = [
        {"start_ppm": 1.4, "end_ppm": 1.75, "reference": True},  # given low-to-high
        {"start_ppm": 7.6, "end_ppm": 7.2, "reference": True},
    ]
    session = Session()
    session.load(PATH)
    apply_session_data(session, payload)
    assert [(r.start_ppm, r.end_ppm) for r in session.regions] == [(7.6, 7.2), (1.75, 1.4)]
    assert sum(r.reference for r in session.regions) == 1


def test_apply_needs_loaded_data(tmp_path):
    with pytest.raises(ValueError):
        apply_session_data(Session(), valid_payload(tmp_path))


def test_a_file_with_non_ascii_text_in_its_path_works(tmp_path):
    folder = tmp_path / "Étude – données"
    folder.mkdir()
    saved = folder / "résumé.nmrproc"
    save_session(worked_session(), saved)
    assert read_session_file(saved)["format"] == FORMAT


def test_exported_spectrum_matches_what_the_session_shows(tmp_path):
    import csv

    session = worked_session()
    spec = session.spectrum()
    out = tmp_path / "spectrum.csv"
    spectrum_to_csv(out, spec)
    with open(out, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["ppm", "intensity"] and len(rows) == len(spec.ppm) + 1
    assert float(rows[1][0]) == pytest.approx(spec.ppm[0], abs=1e-6)
    peak = max(range(1, len(rows)), key=lambda i: float(rows[i][1]))
    assert float(rows[peak][0]) == pytest.approx(7.26, abs=0.002)  # the referenced CHCl3 peak
