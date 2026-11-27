"""Writing results to plain files that open in Excel, R, Python, etc."""
from __future__ import annotations

import csv

from nmrproc.data import Spectrum


def peaks_to_csv(path, peaks) -> None:
    """Write a peak list (ppm, Hz, height) as CSV."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ppm", "hz", "height"])
        for p in peaks:
            writer.writerow([f"{p.ppm:.5f}", f"{p.hz:.3f}", f"{p.height:.6g}"])


def integrals_to_csv(path, integrals) -> None:
    """Write integrals (range edges, raw area, calibrated value) as CSV."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_ppm", "end_ppm", "area", "value"])
        for i in integrals:
            writer.writerow([f"{i.start_ppm:.5f}", f"{i.end_ppm:.5f}", f"{i.area:.6g}",
                             f"{i.value:.4f}"])


def spectrum_to_csv(path, spectrum: Spectrum) -> None:
    """Write the processed spectrum as CSV: ppm, real part, imaginary part."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ppm", "real", "imaginary"])
        for ppm, value in zip(spectrum.ppm, spectrum.data):
            writer.writerow([f"{ppm:.6f}", f"{value.real:.6g}", f"{value.imag:.6g}"])
