"""
MCP server for CO2RR hypothesis testing.

Exposes tools for:
  - Experiment discovery and metadata
  - Spectral data access (raw + processed)
  - Peak detection and assignment
  - Hypothesis cross-referencing
  - Blind-spot guard (unassigned peaks + reference library)
  - Temporal evolution analysis
  - Cross-condition comparison
  - Hypothesis/network state persistence
"""

import json
import os
import glob
import re
from typing import Optional

from mcp.server.fastmcp import FastMCP

from txtr_parser import parse_txtr, extract_scan_info, TxtrSpectrum
from experiment_indexer import index_experiment, load_manifest, write_manifest
from peak_detection import (
    analyze_spectrum,
    compare_scans,
    peak_analysis_to_dict,
    detect_peaks,
)
from reference_library import get_reference_library, get_hypothesis_species, get_guard_species
from hypothesis_schema import load_hypothesis, save_hypothesis

# =============================================================================
# SERVER SETUP
# =============================================================================

DATA_ROOT = os.environ.get("CO2RR_DATA_ROOT", "../demo_data/co2_acetate")

mcp = FastMCP(
    "ChemReasoner",
    instructions=(
        "Query CO2 electroreduction experimental data and cross-reference "
        "against hypothesized reaction networks. Tools cover spectral access, "
        "peak analysis, hypothesis comparison, and blind-spot detection."
    ),
)


# =============================================================================
# HELPERS
# =============================================================================


def _resolve_experiment_dir(date: Optional[str] = None) -> str:
    """Find the experiment directory. If date given, use it; else find the only one."""
    if date:
        path = os.path.join(DATA_ROOT, date)
        if os.path.isdir(path):
            return path
        raise ValueError(f"No experiment directory for date {date} under {DATA_ROOT}")

    # Find date-named directories
    candidates = []
    for entry in os.listdir(DATA_ROOT):
        full = os.path.join(DATA_ROOT, entry)
        if os.path.isdir(full) and re.match(r"\d{4}-\d{2}-\d{2}", entry):
            candidates.append(full)
    if not candidates:
        raise ValueError(f"No date-named experiment directories found under {DATA_ROOT}")
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Multiple experiment dates found, specify date parameter. "
        f"Available: {[os.path.basename(c) for c in candidates]}"
    )


def _resolve_condition_dir(condition: str, date: Optional[str] = None) -> str:
    """Find a condition folder by name or partial match."""
    exp_dir = _resolve_experiment_dir(date)
    # Exact match first
    exact = os.path.join(exp_dir, condition)
    if os.path.isdir(exact):
        return exact

    # Partial match
    condition_lower = condition.lower().replace(" ", "").replace("_", "")
    for entry in os.listdir(exp_dir):
        full = os.path.join(exp_dir, entry)
        if not os.path.isdir(full):
            continue
        entry_clean = entry.lower().replace(" ", "").replace("_", "")
        if condition_lower in entry_clean or entry_clean in condition_lower:
            return full

    available = [e for e in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, e))]
    raise ValueError(
        f"Condition '{condition}' not found under {exp_dir}. "
        f"Available: {available}"
    )


def _get_scan_path(condition_dir: str, scan_number: int) -> str:
    """Find a scan file by number within a condition folder."""
    for f in os.listdir(condition_dir):
        if not f.lower().endswith(".txtr"):
            continue
        match = re.search(r"[Ss]can\s*(\d+)", f)
        if match and int(match.group(1)) == scan_number:
            return os.path.join(condition_dir, f)
    available = sorted([
        f for f in os.listdir(condition_dir)
        if f.lower().endswith(".txtr") and "scan" in f.lower()
    ])
    raise ValueError(
        f"Scan {scan_number} not found in {condition_dir}. "
        f"Available: {available}"
    )


def _get_special_scan_path(condition_dir: str, scan_type: str) -> str:
    """Find a special scan (initial, material) within a condition folder."""
    for f in os.listdir(condition_dir):
        if not f.lower().endswith(".txtr"):
            continue
        if scan_type.lower() in f.lower():
            return os.path.join(condition_dir, f)
    raise ValueError(
        f"No '{scan_type}' scan found in {condition_dir}. "
        f"Files: {os.listdir(condition_dir)}"
    )


def _load_hypothesis() -> dict:
    """Load the hypothesis JSON. Checks standard locations."""
    # Check hypothesis subdirectory
    hyp_dir = os.path.join(DATA_ROOT, "hypothesis")
    if os.path.isdir(hyp_dir):
        for f in sorted(os.listdir(hyp_dir)):
            if f.endswith(".json"):
                return load_hypothesis(os.path.join(hyp_dir, f))

    # Check DATA_ROOT directly
    for f in sorted(os.listdir(DATA_ROOT)):
        if "hypothesis" in f.lower() and f.endswith(".json"):
            return load_hypothesis(os.path.join(DATA_ROOT, f))

    raise ValueError(
        f"No hypothesis JSON found. Expected at {hyp_dir}/*.json "
        f"or {DATA_ROOT}/*hypothesis*.json"
    )


def _parse_and_analyze(filepath: str) -> dict:
    """Parse a .txtr file and run full peak analysis."""
    spectrum = parse_txtr(filepath)
    analysis = analyze_spectrum(
        spectrum.raman_shift_cm1,
        spectrum.intensity,
    )
    result = peak_analysis_to_dict(analysis)
    result["metadata"] = {
        "filepath": filepath,
        "date": spectrum.metadata.date,
        "integration_time_sec": spectrum.metadata.integration_time_sec,
        "laser_wavelength_nm": spectrum.metadata.laser_wavelength_nm,
        "n_datapoints": len(spectrum.raman_shift_cm1),
        "raman_shift_range": [
            spectrum.raman_shift_cm1[0],
            spectrum.raman_shift_cm1[-1],
        ],
    }
    return result, spectrum, analysis


# =============================================================================
# TOOLS: EXPERIMENT DISCOVERY
# =============================================================================


@mcp.tool()
def list_experiments(date: Optional[str] = None) -> str:
    """List available experiments and conditions.
    Returns experiment dates, condition folders, scan counts."""
    if date:
        dirs = [_resolve_experiment_dir(date)]
    else:
        dirs = []
        for entry in sorted(os.listdir(DATA_ROOT)):
            full = os.path.join(DATA_ROOT, entry)
            if os.path.isdir(full) and re.match(r"\d{4}-\d{2}-\d{2}", entry):
                dirs.append(full)
        if not dirs:
            # Check if DATA_ROOT itself contains condition folders
            has_txtr = any(
                f.endswith(".txtr")
                for d in os.listdir(DATA_ROOT)
                if os.path.isdir(os.path.join(DATA_ROOT, d))
                for f in os.listdir(os.path.join(DATA_ROOT, d))
            )
            if has_txtr:
                dirs = [DATA_ROOT]

    results = []
    for exp_dir in dirs:
        conditions = []
        for entry in sorted(os.listdir(exp_dir)):
            cond_path = os.path.join(exp_dir, entry)
            if not os.path.isdir(cond_path) or entry.startswith("."):
                continue
            txtr_files = [f for f in os.listdir(cond_path) if f.lower().endswith(".txtr")]
            scan_files = [f for f in txtr_files if "scan" in f.lower()]
            special_files = [f for f in txtr_files if "scan" not in f.lower()]
            if txtr_files:
                conditions.append({
                    "folder": entry,
                    "n_scans": len(scan_files),
                    "special_files": special_files,
                })
        results.append({
            "date": os.path.basename(exp_dir),
            "path": exp_dir,
            "conditions": conditions,
        })

    return json.dumps(results, indent=2)


@mcp.tool()
def get_experiment_metadata(
    condition: str,
    date: Optional[str] = None,
) -> str:
    """Get detailed metadata for an experimental condition.
    Extracts instrument settings, potentials, and scan inventory from .txtr headers."""
    cond_dir = _resolve_condition_dir(condition, date)
    manifest = index_experiment(
        os.path.dirname(cond_dir),
        ph=6.8,  # CO2-saturated KHCO3
    )
    for cond in manifest.conditions:
        if cond.folder_path == cond_dir:
            from dataclasses import asdict
            return json.dumps(asdict(cond), indent=2)

    raise ValueError(f"Condition {condition} indexed but not found in manifest")


# =============================================================================
# TOOLS: SPECTRAL DATA ACCESS
# =============================================================================


@mcp.tool()
def get_scan(
    condition: str,
    scan_number: int,
    date: Optional[str] = None,
    downsample: int = 1,
) -> str:
    """Get raw Raman spectrum for a specific scan.
    Returns Raman shift (cm⁻¹) and intensity arrays.
    Use downsample > 1 to reduce data size (e.g., downsample=4 returns every 4th point)."""
    cond_dir = _resolve_condition_dir(condition, date)
    filepath = _get_scan_path(cond_dir, scan_number)
    spectrum = parse_txtr(filepath)

    rs = spectrum.raman_shift_cm1[::downsample]
    inten = spectrum.intensity[::downsample]

    return json.dumps({
        "condition": condition,
        "scan_number": scan_number,
        "filepath": filepath,
        "integration_time_sec": spectrum.metadata.integration_time_sec,
        "n_points": len(rs),
        "raman_shift_cm1": rs,
        "intensity": inten,
    }, indent=2)


@mcp.tool()
def get_special_scan(
    condition: str,
    scan_type: str,
    date: Optional[str] = None,
    downsample: int = 1,
) -> str:
    """Get a special scan (initial, material) for a condition.
    scan_type: 'initial' for pre-electrolysis baseline, 'material' for bare electrode."""
    cond_dir = _resolve_condition_dir(condition, date)
    filepath = _get_special_scan_path(cond_dir, scan_type)
    spectrum = parse_txtr(filepath)

    rs = spectrum.raman_shift_cm1[::downsample]
    inten = spectrum.intensity[::downsample]

    return json.dumps({
        "condition": condition,
        "scan_type": scan_type,
        "filepath": filepath,
        "integration_time_sec": spectrum.metadata.integration_time_sec,
        "n_points": len(rs),
        "raman_shift_cm1": rs,
        "intensity": inten,
    }, indent=2)


@mcp.tool()
def get_scan_region(
    condition: str,
    scan_number: int,
    wavenumber_min: float,
    wavenumber_max: float,
    date: Optional[str] = None,
) -> str:
    """Get a specific spectral region from a scan.
    Useful for examining particular peak regions (e.g., CO stretch 1900-2200 cm⁻¹)."""
    cond_dir = _resolve_condition_dir(condition, date)
    filepath = _get_scan_path(cond_dir, scan_number)
    spectrum = parse_txtr(filepath)

    rs = []
    inten = []
    for wn, i in zip(spectrum.raman_shift_cm1, spectrum.intensity):
        if wavenumber_min <= wn <= wavenumber_max:
            rs.append(wn)
            inten.append(i)

    return json.dumps({
        "condition": condition,
        "scan_number": scan_number,
        "region": [wavenumber_min, wavenumber_max],
        "n_points": len(rs),
        "raman_shift_cm1": rs,
        "intensity": inten,
    }, indent=2)


# =============================================================================
# TOOLS: PEAK ANALYSIS
# =============================================================================


@mcp.tool()
def analyze_scan_peaks(
    condition: str,
    scan_number: int,
    date: Optional[str] = None,
    prominence_factor: float = 3.0,
    min_snr: float = 2.0,
) -> str:
    """Full peak analysis on a scan: detect peaks, assign to known species, flag unknowns.

    Returns:
      - assigned: peaks matched to hypothesis or reference species
      - unassigned: peaks not matching any known species (blind-spot guard)
      - hypothesis_species_detected: which hypothesis species are observed
      - guard_species_detected: non-hypothesis species found (electrode, electrolyte)
    """
    cond_dir = _resolve_condition_dir(condition, date)
    filepath = _get_scan_path(cond_dir, scan_number)
    result, _, _ = _parse_and_analyze(filepath)
    result["condition"] = condition
    result["scan_number"] = scan_number
    return json.dumps(result, indent=2)


@mcp.tool()
def get_unassigned_peaks(
    condition: str,
    scan_number: int,
    date: Optional[str] = None,
    min_snr: float = 3.0,
) -> str:
    """BLIND-SPOT GUARD: Get peaks not matching ANY known species.

    These are spectral features that might indicate:
      - Unexpected reaction products
      - Catalyst degradation products
      - Contamination
      - Species not in the hypothesis

    Peaks with high SNR are flagged for LLM interpretation.
    """
    cond_dir = _resolve_condition_dir(condition, date)
    filepath = _get_scan_path(cond_dir, scan_number)
    result, _, analysis = _parse_and_analyze(filepath)

    unassigned = [u for u in result["unassigned"] if u["snr"] >= min_snr]
    llm_candidates = [u for u in unassigned if u["llm_query_suggested"]]

    return json.dumps({
        "condition": condition,
        "scan_number": scan_number,
        "unassigned_peaks": unassigned,
        "n_unassigned": len(unassigned),
        "llm_query_candidates": llm_candidates,
        "n_llm_candidates": len(llm_candidates),
        "residual_energy_pct": result["residual_energy_pct"],
        "interpretation_hint": (
            "Peaks with llm_query_suggested=true have high SNR and don't match "
            "any species in the hypothesis or reference library. Consider querying "
            "an LLM with the wavenumber, intensity, and experimental context to "
            "identify possible species."
        ),
    }, indent=2)


# =============================================================================
# TOOLS: HYPOTHESIS CROSS-REFERENCE
# =============================================================================


@mcp.tool()
def get_hypothesis() -> str:
    """Get the current hypothesis: reaction network, species, predictions."""
    hyp = _load_hypothesis()
    return json.dumps(hyp, indent=2)


@mcp.tool()
def get_hypothesis_predictions(
    potential_vs_rhe_V: Optional[float] = None,
) -> str:
    """Get hypothesis predictions, optionally filtered to a specific potential."""
    hyp = _load_hypothesis()
    predictions = hyp.get("predictions", [])

    if potential_vs_rhe_V is not None:
        # Find closest matching prediction
        best = min(
            predictions,
            key=lambda p: abs(p["potential_vs_rhe_V"] - potential_vs_rhe_V),
        )
        return json.dumps(best, indent=2)

    return json.dumps(predictions, indent=2)


@mcp.tool()
def compare_to_hypothesis(
    condition: str,
    scan_number: int,
    date: Optional[str] = None,
) -> str:
    """Cross-reference a scan's peak analysis against the hypothesis predictions.

    Returns:
      - predicted_species: what the hypothesis expected
      - detected_species: what was actually observed
      - confirmed: species predicted AND detected
      - missing: species predicted but NOT detected
      - unexpected: species detected but NOT predicted
      - verdict: brief assessment
    """
    cond_dir = _resolve_condition_dir(condition, date)
    filepath = _get_scan_path(cond_dir, scan_number)
    result, _, analysis = _parse_and_analyze(filepath)

    hyp = _load_hypothesis()

    # Determine which hypothesis condition matches this experiment
    # Try to extract potential from condition folder name
    pot_match = re.search(r"(-?\d+\.?\d*)\s*V", condition)
    condition_potential = float(pot_match.group(1)) if pot_match else None

    # Find best matching prediction
    predictions = hyp.get("predictions", [])
    matched_pred = None
    if condition_potential is not None and predictions:
        matched_pred = min(
            predictions,
            key=lambda p: abs(p["potential_vs_rhe_V"] - condition_potential),
        )

    # What hypothesis species were detected?
    hyp_species_ids = {s["id"] for s in hyp.get("species", [])}
    detected_hyp = set(result.get("hypothesis_species_detected", []))
    detected_guard = set(result.get("guard_species_detected", []))

    # What was predicted?
    predicted_species = set()
    if matched_pred:
        predicted_species = set(
            matched_pred.get("dominant_products", []) +
            matched_pred.get("expected_intermediates", [])
        )

    confirmed = detected_hyp & predicted_species
    missing = predicted_species - detected_hyp
    unexpected_hyp = detected_hyp - predicted_species
    unexpected_guard = detected_guard

    # Build verdict
    if confirmed and not missing:
        verdict = "CONSISTENT: All predicted species detected."
    elif confirmed and missing:
        verdict = (
            f"PARTIAL: {len(confirmed)} predicted species confirmed, "
            f"{len(missing)} predicted species not detected ({', '.join(sorted(missing))}). "
            f"Check if missing species are below detection limit or conditions differ."
        )
    elif not confirmed and predicted_species:
        verdict = (
            f"INCONSISTENT: None of the predicted species ({', '.join(sorted(predicted_species))}) "
            f"were detected. Hypothesis may not apply at this condition."
        )
    else:
        verdict = "No predictions available for comparison at this condition."

    if unexpected_guard:
        verdict += (
            f" NOTE: Non-hypothesis species detected: {', '.join(sorted(unexpected_guard))}. "
            f"Check for electrode degradation or contamination."
        )

    return json.dumps({
        "condition": condition,
        "scan_number": scan_number,
        "matched_prediction": matched_pred,
        "predicted_species": sorted(predicted_species),
        "detected_hypothesis_species": sorted(detected_hyp),
        "detected_guard_species": sorted(detected_guard),
        "confirmed": sorted(confirmed),
        "missing_from_prediction": sorted(missing),
        "unexpected_hypothesis_species": sorted(unexpected_hyp),
        "unexpected_guard_species": sorted(unexpected_guard),
        "unassigned_peaks": result.get("unassigned", []),
        "verdict": verdict,
    }, indent=2)


# =============================================================================
# TOOLS: TEMPORAL EVOLUTION
# =============================================================================


@mcp.tool()
def get_temporal_evolution(
    condition: str,
    species_ids: list[str],
    date: Optional[str] = None,
) -> str:
    """Track how specific species peak intensities evolve across scans.
    Returns per-species intensity at each scan number."""
    cond_dir = _resolve_condition_dir(condition, date)

    # Find all electrolysis scan files
    scan_files = []
    for f in sorted(os.listdir(cond_dir)):
        if not f.lower().endswith(".txtr"):
            continue
        match = re.search(r"[Ss]can\s*(\d+)", f)
        if match:
            scan_files.append((int(match.group(1)), os.path.join(cond_dir, f)))

    scan_files.sort(key=lambda x: x[0])

    if not scan_files:
        raise ValueError(f"No scan files found in {cond_dir}")

    # Analyze each scan
    evolution = {sid: [] for sid in species_ids}
    scan_numbers = []

    for scan_num, filepath in scan_files:
        try:
            spectrum = parse_txtr(filepath)
            analysis = analyze_spectrum(
                spectrum.raman_shift_cm1,
                spectrum.intensity,
            )
            scan_numbers.append(scan_num)

            # Aggregate intensity per species
            species_intensity = {}
            for a in analysis.assigned:
                sid = a.species_id
                species_intensity[sid] = species_intensity.get(sid, 0) + a.peak.intensity

            for sid in species_ids:
                evolution[sid].append({
                    "scan_number": scan_num,
                    "total_intensity": float(species_intensity.get(sid, 0.0)),
                    "detected": sid in species_intensity,
                })
        except Exception as e:
            for sid in species_ids:
                evolution[sid].append({
                    "scan_number": scan_num,
                    "total_intensity": 0.0,
                    "detected": False,
                    "error": str(e),
                })

    return json.dumps({
        "condition": condition,
        "species_tracked": species_ids,
        "scan_numbers": scan_numbers,
        "evolution": evolution,
    }, indent=2)


@mcp.tool()
def get_temporal_peak_evolution(
    condition: str,
    wavenumber_center: float,
    wavenumber_width: float = 30.0,
    date: Optional[str] = None,
) -> str:
    """Track the intensity at a specific wavenumber region across all scans.
    Useful for monitoring a particular peak regardless of species assignment."""
    cond_dir = _resolve_condition_dir(condition, date)
    wn_min = wavenumber_center - wavenumber_width / 2
    wn_max = wavenumber_center + wavenumber_width / 2

    scan_files = []
    for f in sorted(os.listdir(cond_dir)):
        if not f.lower().endswith(".txtr"):
            continue
        match = re.search(r"[Ss]can\s*(\d+)", f)
        if match:
            scan_files.append((int(match.group(1)), os.path.join(cond_dir, f)))
    scan_files.sort(key=lambda x: x[0])

    results = []
    for scan_num, filepath in scan_files:
        try:
            spectrum = parse_txtr(filepath)
            intensities_in_region = [
                inten for wn, inten
                in zip(spectrum.raman_shift_cm1, spectrum.intensity)
                if wn_min <= wn <= wn_max
            ]
            if intensities_in_region:
                results.append({
                    "scan_number": scan_num,
                    "max_intensity": max(intensities_in_region),
                    "mean_intensity": sum(intensities_in_region) / len(intensities_in_region),
                    "n_points": len(intensities_in_region),
                })
        except Exception as e:
            results.append({
                "scan_number": scan_num,
                "error": str(e),
            })

    return json.dumps({
        "condition": condition,
        "wavenumber_center_cm1": wavenumber_center,
        "wavenumber_range_cm1": [wn_min, wn_max],
        "scans": results,
    }, indent=2)


# =============================================================================
# TOOLS: CROSS-CONDITION COMPARISON
# =============================================================================


@mcp.tool()
def compare_conditions(
    condition_a: str,
    condition_b: str,
    scan_number: int = 1,
    date: Optional[str] = None,
) -> str:
    """Compare peak analyses between two conditions at the same scan number.
    Shows which species appear/disappear and intensity changes."""
    dir_a = _resolve_condition_dir(condition_a, date)
    dir_b = _resolve_condition_dir(condition_b, date)

    path_a = _get_scan_path(dir_a, scan_number)
    path_b = _get_scan_path(dir_b, scan_number)

    spec_a = parse_txtr(path_a)
    spec_b = parse_txtr(path_b)

    analysis_a = analyze_spectrum(spec_a.raman_shift_cm1, spec_a.intensity)
    analysis_b = analyze_spectrum(spec_b.raman_shift_cm1, spec_b.intensity)

    comparison = compare_scans(
        analysis_a, analysis_b,
        label_a=condition_a,
        label_b=condition_b,
    )
    comparison["scan_number"] = scan_number
    comparison["condition_a"] = condition_a
    comparison["condition_b"] = condition_b

    return json.dumps(comparison, indent=2)


@mcp.tool()
def compare_to_initial(
    condition: str,
    scan_number: int,
    date: Optional[str] = None,
) -> str:
    """Compare an electrolysis scan to the initial (pre-electrolysis) baseline.
    Shows what has changed since electrolysis began."""
    cond_dir = _resolve_condition_dir(condition, date)
    scan_path = _get_scan_path(cond_dir, scan_number)
    initial_path = _get_special_scan_path(cond_dir, "initial")

    spec_scan = parse_txtr(scan_path)
    spec_init = parse_txtr(initial_path)

    analysis_scan = analyze_spectrum(spec_scan.raman_shift_cm1, spec_scan.intensity)
    analysis_init = analyze_spectrum(spec_init.raman_shift_cm1, spec_init.intensity)

    comparison = compare_scans(
        analysis_init, analysis_scan,
        label_a="initial",
        label_b=f"scan_{scan_number}",
    )
    comparison["condition"] = condition
    comparison["scan_number"] = scan_number

    return json.dumps(comparison, indent=2)


# =============================================================================
# TOOLS: HYPOTHESIS / NETWORK STATE MANAGEMENT
# =============================================================================


@mcp.tool()
def save_observed_network_state(
    condition: str,
    scan_number: int,
    observations: dict,
    date: Optional[str] = None,
) -> str:
    """Save observed reaction network state for a condition/scan.

    observations should include:
      - detected_species: list of species IDs observed
      - estimated_selectivity: dict of product ratios
      - active_pathways: list of reaction IDs that appear active
      - notes: any additional observations

    Saved to data/<app>/analysis/<condition>/network_state_scan<N>.json
    """
    analysis_dir = os.path.join(DATA_ROOT, "analysis", _sanitize_condition(condition))
    os.makedirs(analysis_dir, exist_ok=True)

    state = {
        "condition": condition,
        "scan_number": scan_number,
        "date": date,
        **observations,
    }

    path = os.path.join(analysis_dir, f"network_state_scan{scan_number:02d}.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)

    return json.dumps({"saved": path, "state": state}, indent=2)


@mcp.tool()
def get_observed_network_states(
    condition: str,
) -> str:
    """Get all saved network state observations for a condition."""
    analysis_dir = os.path.join(DATA_ROOT, "analysis", _sanitize_condition(condition))
    if not os.path.isdir(analysis_dir):
        return json.dumps({"condition": condition, "states": [], "note": "No saved observations"})

    states = []
    for f in sorted(glob.glob(os.path.join(analysis_dir, "network_state_*.json"))):
        with open(f) as fh:
            states.append(json.load(fh))

    return json.dumps({"condition": condition, "states": states}, indent=2)


@mcp.tool()
def save_hypothesis_update(
    hypothesis_data: dict,
    version_suffix: str = "",
) -> str:
    """Save an updated hypothesis JSON.
    Use after Claude Chat refines the hypothesis based on experimental observations."""
    hyp_dir = os.path.join(DATA_ROOT, "hypothesis")
    os.makedirs(hyp_dir, exist_ok=True)

    hyp_id = hypothesis_data.get("hypothesis_id", "hypothesis")
    filename = f"{hyp_id}{version_suffix}.json"
    path = os.path.join(hyp_dir, filename)

    save_hypothesis(hypothesis_data, path)
    return json.dumps({"saved": path}, indent=2)


# =============================================================================
# TOOLS: REFERENCE LIBRARY
# =============================================================================


@mcp.tool()
def get_reference_species(category: Optional[str] = None) -> str:
    """Get the Raman reference library.
    category: 'hypothesis', 'electrolyte', 'electrode', 'contaminant', or None for all."""
    lib = get_reference_library()
    if category:
        lib = {k: v for k, v in lib.items() if v.category == category}

    result = {}
    for sid, species in lib.items():
        result[sid] = {
            "name": species.name,
            "formula": species.formula,
            "category": species.category,
            "peaks": [
                {
                    "wavenumber_cm1": p.wavenumber_cm1,
                    "assignment": p.assignment,
                    "relative_intensity": p.relative_intensity,
                }
                for p in species.peaks
            ],
            "notes": species.notes,
        }
    return json.dumps(result, indent=2)


@mcp.tool()
def identify_wavenumber(wavenumber: float, tolerance: float = 20.0) -> str:
    """Look up what species could produce a peak at a given wavenumber.
    Searches the full reference library within the given tolerance."""
    lib = get_reference_library()
    matches = []
    for species in lib.values():
        for peak in species.peaks:
            dist = abs(peak.wavenumber_cm1 - wavenumber)
            if dist <= tolerance:
                matches.append({
                    "species_id": species.id,
                    "species_name": species.name,
                    "formula": species.formula,
                    "category": species.category,
                    "reference_wavenumber_cm1": peak.wavenumber_cm1,
                    "distance_cm1": round(dist, 1),
                    "assignment": peak.assignment,
                    "relative_intensity": peak.relative_intensity,
                })

    matches.sort(key=lambda m: m["distance_cm1"])
    return json.dumps({
        "query_wavenumber_cm1": wavenumber,
        "tolerance_cm1": tolerance,
        "n_matches": len(matches),
        "matches": matches,
    }, indent=2)


# =============================================================================
# TOOLS: REPORT GENERATION
# =============================================================================


@mcp.tool()
def generate_hypothesis_report(
    date: Optional[str] = None,
    output_name: str = "hypothesis_test_report",
    conditions: Optional[list[str]] = None,
) -> str:
    """Generate a PDF/HTML report comparing experimental observations to hypothesis.

    Produces a multi-page report with:
      - Reaction network diagram
      - Hypothesis predictions
      - Annotated spectra per condition
      - Temporal evolution plots
      - Hypothesis vs observation comparison matrix
      - Blind-spot guard summary

    Returns paths to generated files.
    """
    from report_generator import generate_report

    exp_dir = _resolve_experiment_dir(date)

    # Find hypothesis
    hyp_dir = os.path.join(DATA_ROOT, "hypothesis")
    hyp_path = None
    if os.path.isdir(hyp_dir):
        for f in sorted(os.listdir(hyp_dir)):
            if f.endswith(".json"):
                hyp_path = os.path.join(hyp_dir, f)
                break

    if hyp_path is None:
        raise ValueError(
            f"No hypothesis JSON found in {hyp_dir}. "
            f"Save one with save_hypothesis_update first."
        )

    analysis_dir = os.path.join(DATA_ROOT, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    html_path = os.path.join(analysis_dir, f"{output_name}.html")
    pdf_path = os.path.join(analysis_dir, f"{output_name}.pdf")

    generate_report(
        data_root=exp_dir,
        hypothesis_path=hyp_path,
        output_html=html_path,
        output_pdf=pdf_path,
        condition_filter=conditions,
    )

    return json.dumps({
        "html_report": html_path,
        "pdf_report": pdf_path if os.path.exists(pdf_path) else None,
        "status": "generated",
    }, indent=2)


# =============================================================================
# HELPERS
# =============================================================================


def _sanitize_condition(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_").lower()


# =============================================================================
# ENTRY POINT
# =============================================================================


def _set_data_root(path: str):
    global DATA_ROOT
    DATA_ROOT = path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ChemReasoner MCP Server")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--transport", choices=["stdio", "sse"], default="sse")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    _set_data_root(args.data_root)

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
