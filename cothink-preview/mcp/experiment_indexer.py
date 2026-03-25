"""
Experiment indexer for CO2RR Raman spectroscopy data.

Walks a data directory with structure:
    data/<application>/<date>/<condition_folder>/
        *.txtr files (scans)

Generates:
    - Per-condition metadata.json (extracted from first scan header + folder name)
    - Top-level experiment_manifest.json
"""

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from txtr_parser import parse_txtr, extract_scan_info, TxtrMetadata


@dataclass
class ConditionMetadata:
    """Metadata for one experimental condition (one folder of scans)."""
    condition_id: str  # sanitized folder name
    folder_path: str
    folder_name: str
    sample_name: str  # e.g. "CF41"
    potential_vs_ref_V: Optional[float] = None  # From filename, vs Ag/AgCl
    potential_vs_rhe_V: Optional[float] = None  # Computed from pH
    reference_electrode: str = "Ag/AgCl"
    is_repeat: bool = False
    ph: Optional[float] = None
    electrolyte: Optional[str] = None
    temperature_C: Optional[float] = None
    laser_wavelength_nm: Optional[float] = None
    integration_time_sec: Optional[float] = None
    n_scans: int = 0
    scan_files: List[str] = field(default_factory=list)
    special_files: Dict[str, str] = field(default_factory=dict)  # "initial", "material"


@dataclass
class ExperimentManifest:
    """Top-level manifest for an experiment directory."""
    application: str  # e.g. "co2_acetate"
    date: str
    data_root: str
    conditions: List[ConditionMetadata]
    hypothesis_file: Optional[str] = None


def index_experiment(
    data_root: str,
    ph: float = 6.8,
    electrolyte: str = "CO2-saturated 0.1M KHCO3",
    reference_electrode: str = "Ag/AgCl",
) -> ExperimentManifest:
    """
    Walk the data directory and build a manifest.

    Args:
        data_root: Top-level data directory (e.g. "data/co2_acetate/2026-03-24")
        ph: Electrolyte pH (for RHE conversion). Extracted from scan headers if available.
        electrolyte: Electrolyte description.
        reference_electrode: Reference electrode type.

    Returns:
        ExperimentManifest with all conditions indexed.

    Raises:
        FileNotFoundError: If data_root doesn't exist.
        ValueError: If no scan files found.
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # Detect application and date from path
    path_parts = os.path.normpath(data_root).split(os.sep)
    application = ""
    date = ""
    for i, part in enumerate(path_parts):
        if re.match(r"\d{4}-\d{2}-\d{2}", part):
            date = part
            if i > 0:
                application = path_parts[i - 1]
            break

    # Find condition folders (directories containing .txtr files)
    conditions = []
    for entry in sorted(os.listdir(data_root)):
        cond_path = os.path.join(data_root, entry)
        if not os.path.isdir(cond_path):
            continue
        if entry.startswith("."):
            continue

        txtr_files = sorted([
            f for f in os.listdir(cond_path)
            if f.lower().endswith(".txtr") and not f.startswith(".")
        ])
        if not txtr_files:
            continue

        cond = _index_condition(
            cond_path, entry, txtr_files,
            ph=ph,
            electrolyte=electrolyte,
            reference_electrode=reference_electrode,
        )
        conditions.append(cond)

    if not conditions:
        raise ValueError(
            f"No condition folders with .txtr files found in {data_root}. "
            f"Expected structure: {data_root}/<condition_folder>/*.txtr"
        )

    # Check for hypothesis file
    hyp_file = None
    hyp_dir = os.path.join(os.path.dirname(data_root), "hypothesis")
    if os.path.isdir(hyp_dir):
        for f in os.listdir(hyp_dir):
            if f.endswith(".json"):
                hyp_file = os.path.join(hyp_dir, f)
                break

    return ExperimentManifest(
        application=application,
        date=date,
        data_root=data_root,
        conditions=conditions,
        hypothesis_file=hyp_file,
    )


def _index_condition(
    folder_path: str,
    folder_name: str,
    txtr_files: List[str],
    ph: float,
    electrolyte: str,
    reference_electrode: str,
) -> ConditionMetadata:
    """Index a single condition folder."""
    # Parse folder name for potential and sample info
    # Expected: "Cu-Fe 41_-0.56V" or "Cu-Fe 41_-0.8V_repeat"
    is_repeat = "repeat" in folder_name.lower()
    condition_id = _sanitize_id(folder_name)

    # Extract potential from folder name (this is vs RHE typically)
    pot_rhe = None
    pot_match = re.search(r"(-?\d+\.?\d*)\s*V", folder_name)
    if pot_match:
        pot_rhe = float(pot_match.group(1))

    # Classify scans
    scan_files = []
    special_files: Dict[str, str] = {}
    for f in txtr_files:
        fl = f.lower()
        if "initial" in fl:
            special_files["initial"] = f
        elif "material" in fl:
            special_files["material"] = f
        elif "scan" in fl:
            scan_files.append(f)
        else:
            scan_files.append(f)

    # Sort scan files by scan number
    scan_files = _sort_by_scan_number(scan_files)

    # Read metadata from first available scan
    laser_wl = None
    int_time = None
    pot_ref = None
    first_file = scan_files[0] if scan_files else list(special_files.values())[0]
    first_path = os.path.join(folder_path, first_file)
    try:
        spectrum = parse_txtr(first_path)
        laser_wl = spectrum.metadata.laser_wavelength_nm
        int_time = spectrum.metadata.integration_time_sec

        # Extract potential from scan filename (vs Ag/AgCl)
        info = extract_scan_info(first_path)
        if "potential_V" in info:
            pot_ref = float(info["potential_V"])
    except Exception:
        pass  # Metadata extraction is best-effort

    # Compute RHE potential from reference potential if we have pH
    # V_RHE = V_Ag/AgCl + 0.197 + 0.059 × pH
    computed_rhe = None
    if pot_ref is not None:
        computed_rhe = pot_ref + 0.197 + 0.059 * ph

    # Extract sample name
    sample_name = ""
    name_match = re.match(r"([\w-]+\s*\d+)", folder_name)
    if name_match:
        sample_name = name_match.group(1).strip()

    return ConditionMetadata(
        condition_id=condition_id,
        folder_path=folder_path,
        folder_name=folder_name,
        sample_name=sample_name,
        potential_vs_ref_V=pot_ref,
        potential_vs_rhe_V=pot_rhe if pot_rhe is not None else computed_rhe,
        reference_electrode=reference_electrode,
        is_repeat=is_repeat,
        ph=ph,
        electrolyte=electrolyte,
        temperature_C=25.0,
        laser_wavelength_nm=laser_wl,
        integration_time_sec=int_time,
        n_scans=len(scan_files),
        scan_files=scan_files,
        special_files=special_files,
    )


def _sanitize_id(name: str) -> str:
    """Convert folder name to a clean ID."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_").lower()


def _sort_by_scan_number(files: List[str]) -> List[str]:
    """Sort scan files by numeric scan number."""
    def _key(f: str) -> int:
        match = re.search(r"[Ss]can\s*(\d+)", f)
        return int(match.group(1)) if match else 0
    return sorted(files, key=_key)


def write_manifest(manifest: ExperimentManifest, output_path: str):
    """Write manifest to JSON file."""
    data = asdict(manifest)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_manifest(path: str) -> Dict:
    """Load manifest from JSON file."""
    with open(path) as f:
        return json.load(f)
