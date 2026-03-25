"""
Hypothesis data structure for CO2 electroreduction experiments.

A hypothesis encodes:
  - The reaction network (species, edges, branch points)
  - Predictions per experimental condition (potential, pH)
  - Expected Raman signatures
  - Temporal expectations (which species when, time constants)

This is loaded from JSON and used by the MCP server to cross-reference
experimental observations.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class HypothesisSpecies:
    """A species in the hypothesized reaction network."""
    id: str
    name: str
    formula: str
    role: str  # "reactant", "intermediate", "product_C1", "product_C2", "adsorbed"
    raman_peaks_cm1: List[Dict[str, Any]]  # [{wavenumber, assignment, intensity_class}]
    detectable_by_raman: bool = True
    notes: str = ""


@dataclass
class HypothesisReaction:
    """A reaction edge in the network."""
    id: str
    reactants: List[str]  # species IDs
    products: List[str]  # species IDs
    mechanism: str  # brief description
    is_rate_limiting: bool = False
    time_constant_s: Optional[float] = None  # τ from kinetic model
    notes: str = ""


@dataclass
class BranchPoint:
    """A selectivity-controlling branch point."""
    species_id: str  # The branching intermediate
    branches: Dict[str, List[str]]  # branch_name -> [product species IDs]
    selectivity_factors: List[str]  # what controls selectivity here
    notes: str = ""


@dataclass
class ConditionPrediction:
    """Predicted behavior at a specific experimental condition."""
    condition_label: str  # e.g. "-0.56V vs RHE"
    potential_vs_rhe_V: float
    dominant_products: List[str]  # species IDs, ordered by expected abundance
    expected_intermediates: List[str]
    temporal_expectations: List[Dict[str, Any]]  # [{species, behavior, time_s}]
    selectivity_notes: str = ""


@dataclass
class Hypothesis:
    """Complete hypothesis for an electrochemical system."""
    hypothesis_id: str
    title: str
    system: str  # e.g. "Cu-Fe CO2 electroreduction"
    source: str  # provenance
    created_date: str
    species: List[HypothesisSpecies]
    reactions: List[HypothesisReaction]
    branch_points: List[BranchPoint]
    predictions: List[ConditionPrediction]
    network_notes: str = ""
    catalyst: Dict[str, Any] = field(default_factory=dict)

    @property
    def species_by_id(self) -> Dict[str, HypothesisSpecies]:
        return {s.id: s for s in self.species}


def load_hypothesis(path: str) -> Dict[str, Any]:
    """Load hypothesis from JSON file. Returns raw dict."""
    with open(path) as f:
        return json.load(f)


def save_hypothesis(data: Dict[str, Any], path: str):
    """Save hypothesis dict to JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
