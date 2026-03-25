"""
Peak detection, assignment to known species, and residual analysis.

Three-tier blind-spot guard:
  1. Residual peak detection: peaks not matching any hypothesis species
  2. Reference library matching: check residuals against known non-hypothesis species
  3. LLM fallback: unresolved peaks flagged for LLM interpretation
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import uniform_filter1d

from reference_library import (
    RamanPeak,
    ReferenceSpecies,
    get_reference_library,
    get_hypothesis_species,
    get_guard_species,
)


@dataclass
class DetectedPeak:
    """A peak found in experimental spectrum."""
    wavenumber_cm1: float
    intensity: float
    prominence: float
    width_cm1: float
    snr: float  # signal-to-noise ratio


@dataclass
class PeakAssignment:
    """A detected peak assigned to a species."""
    peak: DetectedPeak
    species_id: str
    species_name: str
    reference_wavenumber_cm1: float
    shift_cm1: float  # observed - reference
    assignment: str  # vibrational mode
    confidence: str  # "high", "medium", "low"
    category: str  # "hypothesis", "electrolyte", "electrode", "contaminant"


@dataclass
class UnassignedPeak:
    """A detected peak not matching any known species."""
    peak: DetectedPeak
    nearest_reference: Optional[str] = None
    nearest_distance_cm1: Optional[float] = None
    llm_query_suggested: bool = False


@dataclass
class PeakAnalysisResult:
    """Complete peak analysis of a spectrum."""
    assigned: List[PeakAssignment]
    unassigned: List[UnassignedPeak]
    hypothesis_matches: List[PeakAssignment]  # subset of assigned: hypothesis species only
    guard_matches: List[PeakAssignment]  # subset: non-hypothesis species
    all_detected: List[DetectedPeak]
    residual_energy_pct: float  # fraction of spectral energy unexplained
    noise_estimate: float


def detect_peaks(
    raman_shift: List[float],
    intensity: List[float],
    smoothing_window: int = 11,
    prominence_factor: float = 3.0,
    min_snr: float = 2.0,
) -> Tuple[List[DetectedPeak], float]:
    """
    Detect peaks in a Raman spectrum.

    Args:
        raman_shift: Wavenumber array (cm⁻¹).
        intensity: Intensity array.
        smoothing_window: Savitzky-Golay window (must be odd).
        prominence_factor: Peaks must be this many × noise above local baseline.
        min_snr: Minimum signal-to-noise ratio to accept a peak.

    Returns:
        (list of DetectedPeak, noise_estimate)
    """
    x = np.array(raman_shift)
    y = np.array(intensity, dtype=float)

    # Smooth for peak detection
    if len(y) < smoothing_window:
        raise ValueError(
            f"Spectrum too short ({len(y)} points) for smoothing window {smoothing_window}"
        )
    y_smooth = savgol_filter(y, smoothing_window, polyorder=3)

    # Estimate noise from residual
    noise = np.std(y - y_smooth)
    if noise <= 0:
        noise = 1.0  # prevent division by zero

    # Estimate local baseline with wide moving average
    baseline = uniform_filter1d(y_smooth, size=min(201, len(y) // 3))

    # Peak detection on baseline-corrected smoothed data
    y_corrected = y_smooth - baseline
    min_prominence = prominence_factor * noise

    indices, properties = find_peaks(
        y_corrected,
        prominence=min_prominence,
        width=2,
        distance=5,
    )

    peaks = []
    for i, idx in enumerate(indices):
        intensity_val = float(y_smooth[idx])
        prominence_val = float(properties["prominences"][i])
        snr = prominence_val / noise

        if snr < min_snr:
            continue

        # Estimate peak width in cm⁻¹
        left_ips = properties["left_ips"][i]
        right_ips = properties["right_ips"][i]
        if right_ips < len(x) and left_ips >= 0:
            left_idx = int(np.floor(left_ips))
            right_idx = int(np.ceil(right_ips))
            right_idx = min(right_idx, len(x) - 1)
            width_cm1 = abs(x[right_idx] - x[left_idx])
        else:
            width_cm1 = 20.0

        peaks.append(DetectedPeak(
            wavenumber_cm1=float(x[idx]),
            intensity=intensity_val,
            prominence=prominence_val,
            width_cm1=width_cm1,
            snr=snr,
        ))

    return peaks, float(noise)


def assign_peaks(
    detected: List[DetectedPeak],
    reference: Dict[str, ReferenceSpecies],
) -> Tuple[List[PeakAssignment], List[DetectedPeak]]:
    """
    Assign detected peaks to reference species.

    Each detected peak is matched to the closest reference peak within tolerance.
    A reference peak can only be claimed once (closest detected peak wins).

    Returns:
        (list of assignments, list of unmatched detected peaks)
    """
    # Build flat list of (species_id, reference_peak) pairs
    ref_entries = []
    for species in reference.values():
        for rp in species.peaks:
            ref_entries.append((species, rp))

    assigned = []
    matched_detected_indices = set()

    # For each reference peak, find the best matching detected peak
    for species, ref_peak in ref_entries:
        best_idx = -1
        best_dist = float("inf")

        for i, dp in enumerate(detected):
            if i in matched_detected_indices:
                continue
            dist = abs(dp.wavenumber_cm1 - ref_peak.wavenumber_cm1)
            if dist <= ref_peak.tolerance_cm1 and dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0:
            dp = detected[best_idx]
            matched_detected_indices.add(best_idx)

            # Confidence based on shift and SNR
            if best_dist <= 5 and dp.snr > 5:
                confidence = "high"
            elif best_dist <= 10 and dp.snr > 3:
                confidence = "medium"
            else:
                confidence = "low"

            assigned.append(PeakAssignment(
                peak=dp,
                species_id=species.id,
                species_name=species.name,
                reference_wavenumber_cm1=ref_peak.wavenumber_cm1,
                shift_cm1=dp.wavenumber_cm1 - ref_peak.wavenumber_cm1,
                assignment=ref_peak.assignment,
                confidence=confidence,
                category=species.category,
            ))

    unmatched = [dp for i, dp in enumerate(detected) if i not in matched_detected_indices]
    return assigned, unmatched


def analyze_spectrum(
    raman_shift: List[float],
    intensity: List[float],
    hypothesis_only: bool = False,
    prominence_factor: float = 3.0,
    min_snr: float = 2.0,
) -> PeakAnalysisResult:
    """
    Full peak analysis: detect, assign, guard.

    Args:
        raman_shift: Wavenumber array (cm⁻¹).
        intensity: Intensity array.
        hypothesis_only: If True, only assign to hypothesis species (skip guard library).
        prominence_factor: Noise multiplier for peak prominence threshold.
        min_snr: Minimum signal-to-noise ratio.

    Returns:
        PeakAnalysisResult with assigned, unassigned, and guard matches.
    """
    # Step 1: Detect peaks
    detected, noise_est = detect_peaks(
        raman_shift, intensity,
        prominence_factor=prominence_factor,
        min_snr=min_snr,
    )

    # Step 2: Assign to hypothesis species
    hypothesis_ref = get_hypothesis_species()
    hyp_assigned, remaining = assign_peaks(detected, hypothesis_ref)

    # Step 3: Assign remaining to guard library (non-hypothesis species)
    guard_assigned = []
    still_unmatched = remaining
    if not hypothesis_only:
        guard_ref = get_guard_species()
        guard_assigned, still_unmatched = assign_peaks(remaining, guard_ref)

    # Step 4: Build unassigned list with nearest-reference hints
    all_ref = get_reference_library()
    all_ref_peaks = []
    for species in all_ref.values():
        for rp in species.peaks:
            all_ref_peaks.append((species.id, rp.wavenumber_cm1))

    unassigned = []
    for dp in still_unmatched:
        nearest_id = None
        nearest_dist = None
        for sid, wn in all_ref_peaks:
            dist = abs(dp.wavenumber_cm1 - wn)
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_id = sid

        unassigned.append(UnassignedPeak(
            peak=dp,
            nearest_reference=nearest_id,
            nearest_distance_cm1=nearest_dist,
            llm_query_suggested=bool(dp.snr > 4.0),  # Suggest LLM query for strong peaks
        ))

    # Step 5: Residual energy estimate
    total_energy = np.sum(np.array(intensity) ** 2)
    assigned_energy = sum(a.peak.intensity ** 2 for a in hyp_assigned + guard_assigned)
    residual_pct = 100.0 * (1.0 - assigned_energy / total_energy) if total_energy > 0 else 100.0

    return PeakAnalysisResult(
        assigned=hyp_assigned + guard_assigned,
        unassigned=unassigned,
        hypothesis_matches=hyp_assigned,
        guard_matches=guard_assigned,
        all_detected=detected,
        residual_energy_pct=residual_pct,
        noise_estimate=noise_est,
    )


def compare_scans(
    analysis_a: PeakAnalysisResult,
    analysis_b: PeakAnalysisResult,
    label_a: str = "scan_a",
    label_b: str = "scan_b",
) -> Dict:
    """
    Compare peak analyses from two scans to detect changes.

    Returns dict with:
      - species appearing/disappearing
      - intensity changes for shared species
      - new unassigned peaks
    """
    # Species detected in each
    species_a = {a.species_id for a in analysis_a.assigned}
    species_b = {a.species_id for a in analysis_b.assigned}

    appeared = species_b - species_a
    disappeared = species_a - species_b
    shared = species_a & species_b

    # Intensity changes for shared species (sum of peak intensities per species)
    def _species_intensity(analysis: PeakAnalysisResult) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for a in analysis.assigned:
            result[a.species_id] = result.get(a.species_id, 0) + a.peak.intensity
        return result

    int_a = _species_intensity(analysis_a)
    int_b = _species_intensity(analysis_b)

    intensity_changes = {}
    for sid in shared:
        ia = int_a.get(sid, 0)
        ib = int_b.get(sid, 0)
        if ia > 0:
            intensity_changes[sid] = {
                label_a: float(ia),
                label_b: float(ib),
                "ratio": float(ib / ia),
                "pct_change": float(100.0 * (ib - ia) / ia),
            }

    return {
        "appeared_in_b": sorted(appeared),
        "disappeared_from_a": sorted(disappeared),
        "shared_species": sorted(shared),
        "intensity_changes": intensity_changes,
        "unassigned_a": len(analysis_a.unassigned),
        "unassigned_b": len(analysis_b.unassigned),
    }


def peak_analysis_to_dict(result: PeakAnalysisResult) -> Dict:
    """Serialize PeakAnalysisResult to a JSON-safe dict."""
    return {
        "assigned": [
            {
                "wavenumber_cm1": float(a.peak.wavenumber_cm1),
                "intensity": float(a.peak.intensity),
                "snr": float(a.peak.snr),
                "species_id": a.species_id,
                "species_name": a.species_name,
                "reference_wavenumber_cm1": float(a.reference_wavenumber_cm1),
                "shift_cm1": float(a.shift_cm1),
                "assignment": a.assignment,
                "confidence": a.confidence,
                "category": a.category,
            }
            for a in result.assigned
        ],
        "unassigned": [
            {
                "wavenumber_cm1": float(u.peak.wavenumber_cm1),
                "intensity": float(u.peak.intensity),
                "snr": float(u.peak.snr),
                "nearest_reference": u.nearest_reference,
                "nearest_distance_cm1": float(u.nearest_distance_cm1) if u.nearest_distance_cm1 is not None else None,
                "llm_query_suggested": bool(u.llm_query_suggested),
            }
            for u in result.unassigned
        ],
        "hypothesis_species_detected": sorted(set(
            a.species_id for a in result.hypothesis_matches
        )),
        "guard_species_detected": sorted(set(
            a.species_id for a in result.guard_matches
        )),
        "n_peaks_total": len(result.all_detected),
        "n_assigned": len(result.assigned),
        "n_unassigned": len(result.unassigned),
        "residual_energy_pct": round(float(result.residual_energy_pct), 2),
        "noise_estimate": round(float(result.noise_estimate), 2),
    }
