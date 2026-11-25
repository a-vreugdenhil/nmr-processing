# nmrproc

A simple, cross-platform (Windows + macOS) program for processing 1D and 2D Bruker NMR
data, intended as a lightweight replacement for SpinWorks for undergraduate use.

**Scope:** loading, apodization, zero filling, Fourier transform, phasing, baseline
correction, referencing, peak picking/integration. No spectral simulation and no
advanced multi-dimensional experiments.

## Development setup

```bash
git clone https://github.com/<your-username>/nmr-processing.git
cd nmr-processing
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS:    source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Layout

- `src/nmrproc/io.py` – reading Bruker data
- `src/nmrproc/processing.py` – GUI-independent processing functions (pure numpy)
- `src/nmrproc/gui/` – user interface (kept separate from processing code)
- `tests/` – automated tests (run on Windows, macOS, and Linux in CI)

## Cross-platform rules

Use `pathlib` for all paths, no Windows-only libraries, no hard-coded path separators.
