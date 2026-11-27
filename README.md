# nmrproc

A simple, cross-platform (Windows + macOS) program for processing 1D and 2D Bruker NMR
data, intended as a lightweight replacement for SpinWorks for undergraduate use.

**Scope:** loading, apodization, zero filling, Fourier transform, phasing, baseline
correction, referencing, peak picking/integration. No spectral simulation and no advanced
multi-dimensional experiments.

**Status:** very preliminary. The 1D processing engine is in place and checked against
TopSpin on real data (loading, apodization, zero filling, Fourier transform, phasing,
baseline correction, referencing, peak picking, integration), and a first 1D window
(PySide6 + matplotlib) lets you open a Bruker folder and process it interactively. Integration
in the window, 2D processing and packaged installers are still to come.

## Development setup

```
git clone https://github.com/a-vreugdenhil/nmr-processing.git
cd nmr-processing
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS:    source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Running the program

```
python -m nmrproc                      # then File > Open (or drag an experiment folder onto the window)
python -m nmrproc testdata/dppe-31P/10 # or open a data set directly
```

Choose the Bruker *experiment* folder, the one that directly contains `fid` and `acqus`.
The window offers line broadening, zero filling, manual and automatic phasing (the dotted red
line is the pivot for first-order phase; tick "Click spectrum to set pivot" and click a peak to
move it), baseline correction, and peak labels. Zoom and pan with the toolbar, and use the disk
icon to save a figure.

## Quick example

```python
from nmrproc.baseline import correct_baseline
from nmrproc.integration import find_integral_regions, integrate, normalize_integrals
from nmrproc.io import load_fid_1d
from nmrproc.peaks import pick_peaks
from nmrproc.phasing import auto_phase
from nmrproc.processing import phase_spectrum, process_1d

fid = load_fid_1d("testdata/triphenylphosphine-1H/10")
spec = process_1d(fid, lb_hz=0.3, si=65536)          # apodize, zero fill, FT
p0, p1 = auto_phase(spec)                            # automatic phasing (refine by hand if needed)
spec = correct_baseline(phase_spectrum(spec, p0, p1))

peaks = pick_peaks(spec)                             # list of Peak(ppm, hz, height, ...)
regions = find_integral_regions(spec)                # suggested integration ranges
integrals = normalize_integrals(integrate(spec, regions), which=0, value=1.0)
```

Phase angles use TopSpin's PHC0/PHC1 convention, so values from TopSpin can be passed straight
to `process_1d(..., p0=..., p1=...)`.

## Layout

- `src/nmrproc/data.py` – plain containers (`FID`, `Spectrum`)
- `src/nmrproc/io.py` – reading Bruker data
- `src/nmrproc/processing.py` – apodization, zero filling, FT, phasing, referencing (pure numpy)
- `src/nmrproc/phasing.py` – automatic phase correction (ACME)
- `src/nmrproc/baseline.py` – baseline correction and noise estimation
- `src/nmrproc/peaks.py` – peak picking
- `src/nmrproc/integration.py` – integration ranges, integrals, integral trace
- `src/nmrproc/gui/` – user interface, kept separate from the processing code: `session.py`
  (settings and pipeline, no Qt), `main_window.py`, `widgets.py`, `app.py`
- `tests/` – automated tests (run on Windows, macOS, and Linux in CI)
- `testdata/` – real Bruker data sets (1H and 31P) used to check results against TopSpin

## Conventions

Spectra are ordered like TopSpin (highest ppm first), and phase angles use TopSpin's
PHC0/PHC1 convention, so results can be compared directly with Bruker's own processing.

## Cross-platform rules

Use `pathlib` for all paths, no Windows-only libraries, no hard-coded path separators.

## License

MIT, see `LICENSE`.
