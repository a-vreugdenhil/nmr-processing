# nmrproc

A simple, cross-platform (Windows + macOS) program for processing 1D and 2D Bruker NMR
data, intended as a lightweight replacement for SpinWorks for undergraduate use.

**Scope:** loading, apodization, zero filling, Fourier transform, phasing, baseline
correction, referencing, peak picking/integration. No spectral simulation and no advanced
multi-dimensional experiments.

**Status:** very preliminary. 1D loading, apodization, zero filling, Fourier transform,
phasing and referencing work and are checked against TopSpin. Baseline correction, peak
picking, integration, 2D processing and the user interface are still to come.

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

## Quick example

```python
from nmrproc.io import load_fid_1d
from nmrproc.processing import process_1d

fid = load_fid_1d("testdata/triphenylphosphine-1H/10")
spectrum = process_1d(fid, lb_hz=0.3, si=65536, p0=29.6)   # phase values as in TopSpin PHC0/PHC1
print(spectrum.ppm[0], spectrum.real.max())
```

## Layout

- `src/nmrproc/data.py` – plain containers (`FID`, `Spectrum`)
- `src/nmrproc/io.py` – reading Bruker data
- `src/nmrproc/processing.py` – GUI-independent processing functions (pure numpy)
- `src/nmrproc/gui/` – user interface (kept separate from processing code)
- `tests/` – automated tests (run on Windows, macOS, and Linux in CI)
- `testdata/` – real Bruker data sets (1H and 31P) used to check results against TopSpin

## Conventions

Spectra are ordered like TopSpin (highest ppm first), and phase angles use TopSpin's
PHC0/PHC1 convention, so results can be compared directly with Bruker's own processing.

## Cross-platform rules

Use `pathlib` for all paths, no Windows-only libraries, no hard-coded path separators.

## License

MIT, see `LICENSE`.
