# nmrproc

A simple, cross-platform (Windows + macOS) program for processing 1D and 2D Bruker NMR
data, intended as a lightweight replacement for SpinWorks for undergraduate use.

**Scope:** loading, apodization, zero filling, Fourier transform, phasing, baseline
correction, referencing, peak picking/integration. No spectral simulation and no advanced
multi-dimensional experiments.

**Status:** very preliminary. The 1D processing engine is in place and checked against
TopSpin on real data (loading, window functions, zero filling, Fourier transform, phasing,
baseline correction, referencing, peak picking, integration), and a 1D window (PySide6 +
matplotlib) lets you open a Bruker folder and do all of this interactively, including
integration, chemical-shift referencing, saving and reopening sessions, and CSV export of the
spectrum, peaks and integrals. 2D processing and packaged installers are still to come.

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

**Process tab:** window function (none, exponential, Gaussian resolution enhancement, sine
bell, squared sine bell; a short explanation is shown for each), zero filling, manual and
automatic phasing (the dotted red line is the pivot for first-order phase; tick "Click
spectrum to set pivot" and click a peak to move it), and baseline correction.

**Analyze tab:**
- *Chemical-shift reference:* click a peak and choose a standard (for example CDCl3 at 7.26 ppm)
  or type a shift. Integration ranges move with the peaks.
- *Peaks:* labels with an adjustable signal-to-noise threshold.
- *Integrals:* "Draw integrals", then drag across a peak or multiplet to add a range, drag a
  range's edge to adjust it, or click inside a range to select it. "Suggest ranges" proposes
  ranges automatically. Select a row and set it to a known value (for example 3.00 for a CH3)
  to calibrate the others.

A yellow banner above the plot shows which click mode is active (pivot, reference or integrate);
press Esc to leave it. Starting a mode switches off the toolbar's zoom/pan tool, which would
otherwise swallow the clicks.

## Saving your work

- **Save session / Open session** (File menu, Ctrl+S): a session file (`.nmrproc`) records your
  processing choices, chemical-shift reference, integrals and zoom, and where the data folder is.
  It does not contain the data. Reopening it restores everything, and you can also double-click
  it in the file dialog, drag it onto the window, or run `python -m nmrproc my_lab.nmrproc`.
  Keep the session file and the data folder together (for example both in one folder): the
  session finds its data relative to its own location, so the folder can be moved or copied to
  another computer. If the data cannot be found, the program asks you to locate the folder.
- **Export processed spectrum (CSV):** ppm and intensity of the spectrum as currently processed,
  for plotting in Excel, Origin or Python. Peak lists and integrals can be exported too.
- A `*` in the window title means there are unsaved changes; closing the window or opening other
  data then asks whether to save them first.
- Session files are plain text (JSON), so they are safe to keep in version control or email.
  A damaged or newer-format file is reported with a message and never disturbs the work you
  have open.

Zoom and pan with the toolbar, and use the disk icon to save a figure.

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
- `src/nmrproc/processing.py` – window functions, zero filling, FT, phasing, referencing (pure numpy)
- `src/nmrproc/phasing.py` – automatic phase correction (ACME)
- `src/nmrproc/baseline.py` – baseline correction and noise estimation
- `src/nmrproc/peaks.py` – peak picking
- `src/nmrproc/integration.py` – integration ranges, integrals, integral curves
- `src/nmrproc/export.py` – CSV export of peaks, integrals and spectra
- `src/nmrproc/gui/` – user interface, kept separate from the processing code: `session.py`
  (settings and pipeline, no Qt), `sessionfile.py` (save/reopen sessions, no Qt),
  `main_window.py`, `widgets.py`, `app.py`
- `tests/` – automated tests (run on Windows, macOS, and Linux in CI)
- `testdata/` – real Bruker data sets (1H and 31P) used to check results against TopSpin

## Conventions

Spectra are ordered like TopSpin (highest ppm first), and phase angles use TopSpin's
PHC0/PHC1 convention, so results can be compared directly with Bruker's own processing.

## Cross-platform rules

Use `pathlib` for all paths, no Windows-only libraries, no hard-coded path separators.

## License

MIT, see `LICENSE`.
