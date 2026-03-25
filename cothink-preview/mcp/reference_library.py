"""
Reference library of known Raman-active species for CO2 electroreduction.

Three categories:
  1. Hypothesis species: expected products/intermediates in CO2RR
  2. Electrolyte/environment: bicarbonate, carbonate, water
  3. Electrode materials: Cu oxides, Fe oxides, carbon

Each entry has characteristic peaks with literature assignments.
Sources: NIST spectral database, published CO2RR Raman studies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RamanPeak:
    """A single Raman peak with assignment."""
    wavenumber_cm1: float
    assignment: str  # Vibrational mode description
    relative_intensity: str  # "strong", "medium", "weak"
    width_cm1: float = 20.0  # Typical FWHM
    tolerance_cm1: float = 15.0  # Matching tolerance


@dataclass
class ReferenceSpecies:
    """A chemical species with its Raman signature."""
    id: str
    name: str
    formula: str
    category: str  # "hypothesis", "electrolyte", "electrode", "contaminant"
    peaks: List[RamanPeak]
    notes: str = ""


def get_reference_library() -> Dict[str, ReferenceSpecies]:
    """Return the full reference library keyed by species ID."""
    species = _build_library()
    return {s.id: s for s in species}


def get_hypothesis_species() -> Dict[str, ReferenceSpecies]:
    """Return only hypothesis-relevant species (CO2RR products/intermediates)."""
    lib = get_reference_library()
    return {k: v for k, v in lib.items() if v.category == "hypothesis"}


def get_guard_species() -> Dict[str, ReferenceSpecies]:
    """Return species NOT in the hypothesis but commonly observed in CO2RR experiments.
    These form the 'known unknowns' safety net."""
    lib = get_reference_library()
    return {k: v for k, v in lib.items() if v.category != "hypothesis"}


def _build_library() -> List[ReferenceSpecies]:
    return [
        # =====================================================================
        # HYPOTHESIS SPECIES: CO2RR intermediates and products
        # =====================================================================
        ReferenceSpecies(
            id="CO2_dissolved",
            name="Dissolved CO2",
            formula="CO2(aq)",
            category="hypothesis",
            peaks=[
                RamanPeak(1285, "ν1 symmetric stretch (Fermi dyad)", "strong"),
                RamanPeak(1388, "2ν2 overtone (Fermi dyad)", "strong"),
            ],
            notes="Fermi resonance dyad. Intensity drops as CO2 is consumed at electrode.",
        ),
        ReferenceSpecies(
            id="COOH_ads",
            name="Adsorbed *COOH",
            formula="*COOH",
            category="hypothesis",
            peaks=[
                RamanPeak(1540, "νas(OCO) + δ(OH)", "medium", tolerance_cm1=25),
                RamanPeak(1290, "ν(C-OH)", "weak", tolerance_cm1=25),
                RamanPeak(360, "ν(Cu-C)", "weak", tolerance_cm1=20),
            ],
            notes="Transient intermediate. Broad, often overlaps with carbonate. "
                  "Detection requires SERS-active surface.",
        ),
        ReferenceSpecies(
            id="CO_ads",
            name="Adsorbed *CO",
            formula="*CO",
            category="hypothesis",
            peaks=[
                RamanPeak(2060, "ν(C≡O) atop site", "strong", tolerance_cm1=30),
                RamanPeak(1990, "ν(C≡O) bridge site", "medium", tolerance_cm1=30),
                RamanPeak(360, "ν(Cu-CO) frustrated translation", "medium"),
                RamanPeak(280, "ν(Cu-CO) frustrated rotation", "weak"),
            ],
            notes="Key intermediate. Atop CO (2050-2100) is primary on Cu. "
                  "Bridge CO (1850-2000) indicates multi-coordination.",
        ),
        ReferenceSpecies(
            id="formate",
            name="Formate",
            formula="HCOO⁻",
            category="hypothesis",
            peaks=[
                RamanPeak(1351, "νs(OCO) symmetric stretch", "strong"),
                RamanPeak(1580, "νas(OCO) asymmetric stretch", "medium"),
                RamanPeak(1066, "δ(CH) in-plane bend", "medium"),
                RamanPeak(770, "π(OCO) out-of-plane deformation", "weak"),
            ],
            notes="C1 product. Strong 1351 cm⁻¹ peak is diagnostic.",
        ),
        ReferenceSpecies(
            id="methanol",
            name="Methanol",
            formula="CH3OH",
            category="hypothesis",
            peaks=[
                RamanPeak(1033, "ν(C-O) stretch", "strong"),
                RamanPeak(2945, "νs(CH3) symmetric stretch", "strong"),
                RamanPeak(2833, "νs(CH3) Fermi resonance", "medium"),
                RamanPeak(1450, "δas(CH3) asymmetric deformation", "weak"),
            ],
            notes="C1 product. 1033 cm⁻¹ C-O stretch is diagnostic but can overlap with carbonate.",
        ),
        ReferenceSpecies(
            id="ethylene",
            name="Ethylene",
            formula="C2H4",
            category="hypothesis",
            peaks=[
                RamanPeak(1623, "ν(C=C) stretch", "strong"),
                RamanPeak(1342, "δ(CH2) scissor", "medium"),
                RamanPeak(3019, "ν(C-H) stretch", "weak"),
            ],
            notes="C2 product. May desorb as gas — Raman detection in liquid limited.",
        ),
        ReferenceSpecies(
            id="ethanol",
            name="Ethanol",
            formula="C2H5OH",
            category="hypothesis",
            peaks=[
                RamanPeak(882, "ν(C-C-O) stretch", "strong"),
                RamanPeak(1050, "ν(C-O) stretch", "medium"),
                RamanPeak(1454, "δ(CH3/CH2) deformation", "medium"),
                RamanPeak(2930, "ν(C-H) stretch", "strong"),
            ],
            notes="C2 product. 882 cm⁻¹ is most diagnostic.",
        ),
        ReferenceSpecies(
            id="acetate",
            name="Acetate",
            formula="CH3COO⁻",
            category="hypothesis",
            peaks=[
                RamanPeak(928, "ν(C-C) stretch", "strong"),
                RamanPeak(1413, "νs(COO⁻) symmetric stretch", "strong"),
                RamanPeak(1556, "νas(COO⁻) asymmetric stretch", "medium"),
                RamanPeak(1344, "δs(CH3) symmetric deformation", "medium"),
                RamanPeak(2936, "ν(C-H) stretch", "medium"),
            ],
            notes="C2 product — target for Cu-Fe system. 928 cm⁻¹ C-C stretch is diagnostic.",
        ),

        # =====================================================================
        # ELECTROLYTE / ENVIRONMENT
        # =====================================================================
        ReferenceSpecies(
            id="bicarbonate",
            name="Bicarbonate",
            formula="HCO3⁻",
            category="electrolyte",
            peaks=[
                RamanPeak(1017, "ν1 symmetric stretch", "strong"),
                RamanPeak(1302, "ν3 asymmetric stretch", "weak"),
                RamanPeak(1365, "ν3 asymmetric stretch (2nd component)", "weak"),
                RamanPeak(680, "δ(OCO) in-plane bend", "medium"),
            ],
            notes="0.1M KHCO3 electrolyte. Dominant background signal at 1017 cm⁻¹.",
        ),
        ReferenceSpecies(
            id="carbonate",
            name="Carbonate",
            formula="CO3²⁻",
            category="electrolyte",
            peaks=[
                RamanPeak(1065, "ν1 symmetric stretch", "strong"),
                RamanPeak(880, "ν2 out-of-plane bend", "weak"),
            ],
            notes="Forms at high local pH near electrode during CO2RR.",
        ),
        ReferenceSpecies(
            id="water",
            name="Water",
            formula="H2O",
            category="electrolyte",
            peaks=[
                RamanPeak(1640, "δ(HOH) bend", "medium"),
                RamanPeak(3400, "ν(O-H) stretch (broad)", "strong", width_cm1=200),
            ],
            notes="Always present. Broad OH stretch can obscure C-H region.",
        ),

        # =====================================================================
        # ELECTRODE MATERIALS
        # =====================================================================
        ReferenceSpecies(
            id="Cu2O",
            name="Cuprous oxide",
            formula="Cu2O",
            category="electrode",
            peaks=[
                RamanPeak(218, "2nd order mode", "medium"),
                RamanPeak(520, "T1u(TO)", "medium"),
                RamanPeak(620, "T1u(LO) / IR-active", "strong"),
            ],
            notes="Forms on Cu under reducing conditions. 620 cm⁻¹ is diagnostic. "
                  "Presence indicates incomplete reduction of oxide.",
        ),
        ReferenceSpecies(
            id="CuO",
            name="Cupric oxide",
            formula="CuO",
            category="electrode",
            peaks=[
                RamanPeak(295, "Ag mode", "strong"),
                RamanPeak(345, "Bg mode", "medium"),
                RamanPeak(630, "Bg mode", "medium"),
            ],
            notes="CuO reduction to Cu is expected pre-electrolysis. "
                  "Persistence suggests incomplete activation.",
        ),
        ReferenceSpecies(
            id="Fe2O3",
            name="Iron(III) oxide / hematite",
            formula="α-Fe2O3",
            category="electrode",
            peaks=[
                RamanPeak(225, "A1g", "strong"),
                RamanPeak(290, "Eg", "strong"),
                RamanPeak(410, "Eg", "medium"),
                RamanPeak(500, "A1g", "weak"),
                RamanPeak(610, "Eg", "medium"),
            ],
            notes="Fe component of Cu-Fe catalyst. May indicate Fe segregation or oxidation.",
        ),
        ReferenceSpecies(
            id="FeOOH",
            name="Iron oxyhydroxide",
            formula="FeOOH",
            category="electrode",
            peaks=[
                RamanPeak(250, "Fe-O stretch", "medium"),
                RamanPeak(385, "Fe-OH bend", "medium"),
                RamanPeak(550, "Fe-O stretch", "weak"),
            ],
            notes="Possible Fe corrosion product in aqueous electrolyte.",
        ),

        # =====================================================================
        # CONTAMINANTS / OTHER
        # =====================================================================
        ReferenceSpecies(
            id="graphitic_carbon",
            name="Graphitic/Amorphous Carbon",
            formula="C",
            category="contaminant",
            peaks=[
                RamanPeak(1350, "D band (disorder)", "strong", width_cm1=60),
                RamanPeak(1580, "G band (graphitic)", "strong", width_cm1=40),
            ],
            notes="Carbon deposits from organic decomposition. D/G ratio indicates disorder. "
                  "Can form at high overpotentials — indicates catalyst degradation.",
        ),
        ReferenceSpecies(
            id="sulfate",
            name="Sulfate",
            formula="SO4²⁻",
            category="contaminant",
            peaks=[
                RamanPeak(981, "ν1 symmetric stretch", "strong"),
            ],
            notes="Trace contamination indicator. Very sharp, easy to identify.",
        ),
    ]
