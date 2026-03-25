"""
Parser for BWTek .txtr Raman spectrometer files.

Extracts metadata from headers and spectral data (Raman shift, intensity)
from the data section.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Column indices in the semicolon-delimited data section (0-based)
_COL_PIXEL = 0
_COL_WAVELENGTH_NM = 1
_COL_WAVENUMBER = 2
_COL_RAMAN_SHIFT = 3
_COL_DARK = 4
_COL_REFERENCE = 5
_COL_RAW = 6
_COL_DARK_SUBTRACTED = 7


@dataclass
class TxtrMetadata:
    """Metadata extracted from .txtr file header."""
    filepath: str
    date: str
    instrument_model: str
    laser_wavelength_nm: float
    integration_time_sec: float
    pixel_count: int
    raman_shift_range: Tuple[float, float]  # (min, max) cm⁻¹
    gain: int
    raw_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class TxtrSpectrum:
    """Parsed spectrum from a .txtr file."""
    metadata: TxtrMetadata
    raman_shift_cm1: List[float]
    intensity: List[float]  # dark-subtracted
    raw_counts: List[float]
    dark_counts: List[float]


def parse_txtr(filepath: str, min_raman_shift: float = 100.0) -> TxtrSpectrum:
    """
    Parse a .txtr file into structured spectrum data.

    Args:
        filepath: Path to the .txtr file.
        min_raman_shift: Discard data below this Raman shift (cm⁻¹).
            Negative shifts and near-Rayleigh data are noise.

    Returns:
        TxtrSpectrum with metadata, Raman shifts, and intensities.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is unrecognizable.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"TXTR file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw_text = f.read()

    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    # Parse header (key;value pairs until we hit the column header line)
    headers: Dict[str, str] = {}
    data_start_line = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(";", 1)
        if len(parts) == 2:
            key = parts[0].strip()
            val = parts[1].strip().rstrip(";")
            # Detect the column header line
            if key == "Pixel" or (key.isdigit() and data_start_line == -1):
                if key == "Pixel":
                    data_start_line = i + 1
                else:
                    data_start_line = i
                break
            headers[key] = val

    if data_start_line == -1:
        raise ValueError(
            f"Could not find data section in {filepath}. "
            f"Parsed {len(headers)} header entries but no data rows."
        )

    # Extract metadata
    metadata = TxtrMetadata(
        filepath=filepath,
        date=headers.get("Date", ""),
        instrument_model=headers.get("model", ""),
        laser_wavelength_nm=_float_or(headers.get("laser_wavelength", ""), 0.0),
        integration_time_sec=_float_or(headers.get("integration times(sec)", ""), 0.0),
        pixel_count=int(_float_or(headers.get("pixel_num", ""), 0)),
        raman_shift_range=(
            _float_or(headers.get("xaxis_min", ""), 0.0),
            _float_or(headers.get("xaxis_max", ""), 0.0),
        ),
        gain=int(_float_or(headers.get("gain", ""), 0)),
        raw_headers=headers,
    )

    # Parse data rows
    raman_shifts = []
    intensities = []
    raw_counts = []
    dark_counts = []

    for i in range(data_start_line, len(lines)):
        stripped = lines[i].strip().rstrip(";")
        if not stripped:
            continue
        parts = stripped.split(";")
        if len(parts) < 8:
            continue
        try:
            rs = float(parts[_COL_RAMAN_SHIFT])
            if rs < min_raman_shift:
                continue
            dark_sub = float(parts[_COL_DARK_SUBTRACTED])
            raw = float(parts[_COL_RAW])
            dark = float(parts[_COL_DARK])
            raman_shifts.append(rs)
            intensities.append(dark_sub)
            raw_counts.append(raw)
            dark_counts.append(dark)
        except (ValueError, IndexError):
            continue

    if not raman_shifts:
        raise ValueError(
            f"No valid spectral data found in {filepath} "
            f"(data_start_line={data_start_line}, total_lines={len(lines)}, "
            f"min_raman_shift={min_raman_shift})"
        )

    return TxtrSpectrum(
        metadata=metadata,
        raman_shift_cm1=raman_shifts,
        intensity=intensities,
        raw_counts=raw_counts,
        dark_counts=dark_counts,
    )


def _float_or(val: str, default: float) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def extract_scan_info(filepath: str) -> Dict[str, str]:
    """
    Extract experiment info from filepath conventions.

    Expects patterns like:
        CF41_-1.2V_Scan 5 .txtr  -> potential=-1.2V, scan_number=5
        CF41_initial_0.1M_HCO3.txtr -> type=initial, electrolyte=0.1M_HCO3
        CF41_material.txtr -> type=material

    Returns dict with extracted fields.
    """
    basename = os.path.splitext(os.path.basename(filepath))[0].strip()
    info: Dict[str, str] = {"filename": basename, "filepath": filepath}

    lower = basename.lower()

    if "initial" in lower:
        info["scan_type"] = "initial"
    elif "material" in lower:
        info["scan_type"] = "material"
    elif "scan" in lower:
        info["scan_type"] = "electrolysis"
        # Extract scan number
        parts = lower.split("scan")
        if len(parts) > 1:
            num_str = parts[1].strip().split("_")[0].split(" ")[0]
            if not num_str:
                # Handle "Scan 5 " pattern
                remainder = parts[1].strip()
                for token in remainder.split():
                    if token.isdigit():
                        num_str = token
                        break
            if num_str.isdigit():
                info["scan_number"] = num_str
    else:
        info["scan_type"] = "unknown"

    # Extract potential from filename (e.g., -1.2V, -1.4V)
    import re
    pot_match = re.search(r"(-?\d+\.?\d*)\s*V", basename, re.IGNORECASE)
    if pot_match:
        info["potential_V"] = pot_match.group(1)

    return info
