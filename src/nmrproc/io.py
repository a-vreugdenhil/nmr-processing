"""Reading Bruker data (1D)."""
from __future__ import annotations

import warnings
from pathlib import Path

import nmrglue as ng
import numpy as np

from nmrproc.data import FID, Spectrum
from nmrproc.processing import ppm_axis


def _read(path):
    # nmrglue warns when the pulse program text cannot be located; this is harmless for us.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return ng.bruker.read(str(path))


def load_bruker(path):
    """Load a Bruker experiment directory (the folder containing fid/ser).

    Returns (dic, data) with the digital filter removed for 1D data.
    """
    path = Path(path)
    dic, data = _read(path)
    data = ng.bruker.remove_digital_filter(dic, data)
    return dic, data


def load_fid_1d(path) -> FID:
    """Load a 1D Bruker experiment folder as an FID with its acquisition parameters."""
    path = Path(path)
    dic, data = load_bruker(path)
    if data.ndim != 1:
        raise ValueError(f"{path} is not a 1D data set (data has {data.ndim} dimensions)")
    acqus = dic["acqus"]
    return FID(
        data=data,
        sw_hz=float(acqus["SW_h"]),
        bf_mhz=float(acqus["BF1"]),
        sfo_mhz=float(acqus["SFO1"]),
        o1_hz=float(acqus["O1"]),
        nucleus=str(acqus.get("NUC1", "")).strip("<>"),
        solvent=str(acqus.get("SOLVENT", "")).strip("<>"),
        title=_read_title(path),
        path=str(path),
    )


def _read_title(path: Path, procno: int = 1) -> str:
    title_file = path / "pdata" / str(procno) / "title"
    try:
        return title_file.read_text(errors="replace").strip()
    except OSError:
        return ""


def read_proc_params(path, procno: int = 1) -> dict:
    """Return TopSpin's processing parameters (procs) for a processed data set as a dict.

    Useful keys: LB, SI, PHC0, PHC1, WDW, OFFSET, SF.
    """
    pdic, _ = _read_pdata(Path(path) / "pdata" / str(procno))
    return pdic["procs"]


def _read_pdata(pdata_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return ng.bruker.read_pdata(str(pdata_dir))


def load_processed_1d(path, procno: int = 1) -> Spectrum:
    """Load the spectrum TopSpin already processed (real part only), e.g. for validation."""
    path = Path(path)
    pdic, data = _read_pdata(path / "pdata" / str(procno))
    procs = pdic["procs"]
    acqus = ng.bruker.read_jcamp(str(path / "acqus"))
    sw_hz = float(procs["SW_p"])
    sf_mhz = float(procs["SF"])
    n = len(data)
    # OFFSET is the shift (ppm) of the left-most point; points are evenly spaced across SW.
    ppm = ppm_axis(n, sw_hz, sf_mhz, float(procs["OFFSET"]))
    return Spectrum(
        data=np.asarray(data, dtype=complex),
        ppm=ppm,
        sw_hz=sw_hz,
        sf_mhz=sf_mhz,
        nucleus=str(acqus.get("NUC1", "")).strip("<>"),
        info={"procno": procno, "source": "TopSpin"},
    )

