"""
Report generator for CO2RR hypothesis testing.

Produces the 'reaction network + temporal evolution' style report
that was a hit with the chemists (sample.pdf style).

Generates:
  Page 1: Reaction network diagram with embedded time-series thumbnails
  Page 2: Temporal evolution of species + selectivity analysis
  Page 3: Hypothesis vs observation comparison / rate-limiting step analysis
  Page 4: Blind-spot guard summary

Output: HTML file (viewable in browser) + PDF via WeasyPrint.
"""

import base64
import io
import json
import os
import re
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects

from txtr_parser import parse_txtr, TxtrSpectrum
from peak_detection import analyze_spectrum, peak_analysis_to_dict
from hypothesis_schema import load_hypothesis
from reference_library import get_reference_library


# =============================================================================
# COLORS & STYLES
# =============================================================================

# Color scheme matching sample.pdf aesthetic
COLORS = {
    "reactant": "#2E86AB",       # Teal-blue for CO2
    "intermediate": "#A23B72",   # Magenta for intermediates
    "product_C1": "#F18F01",     # Orange for C1 products
    "product_C2": "#2D6A4F",     # Forest green for C2 products
    "adsorbed": "#7B2D8E",       # Purple for adsorbed species
    "target": "#C73E1D",         # Red-orange for target product
    "background": "#F8F9FA",
    "text_dark": "#1A1A2E",
    "text_mid": "#4A4A6A",
    "border": "#DEE2E6",
    "accent": "#E63946",
    "confirmed": "#2D6A4F",
    "missing": "#E63946",
    "unexpected": "#F18F01",
}

ROLE_COLORS = {
    "reactant": COLORS["reactant"],
    "intermediate": COLORS["intermediate"],
    "product_C1": COLORS["product_C1"],
    "product_C2": COLORS["product_C2"],
    "adsorbed": COLORS["adsorbed"],
    "target": COLORS["target"],
}


def _fig_to_base64(fig: plt.Figure, dpi: int = 150) -> str:
    """Convert matplotlib figure to base64-encoded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# =============================================================================
# PLOT: REACTION NETWORK DIAGRAM
# =============================================================================


def _plot_reaction_network(hypothesis: Dict) -> str:
    """Generate the reaction network diagram with species nodes and reaction arrows."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    species = {s["id"]: s for s in hypothesis["species"]}
    reactions = hypothesis["reactions"]

    # Layout positions for CO2RR network (hand-tuned for clarity)
    positions = {
        "CO2":       (1.0, 7.0),
        "COOH_ads":  (4.0, 7.0),
        "CO_ads":    (4.0, 4.0),
        "formate":   (8.0, 8.0),
        "methanol":  (1.0, 2.5),
        "ethylene":  (7.0, 2.5),
        "ethanol":   (10.0, 2.5),
        "acetate":   (10.0, 5.0),
    }

    # Draw reaction arrows first (behind nodes)
    arrow_map = {}
    for rxn in reactions:
        src_ids = rxn["reactants"]
        dst_ids = rxn["products"]
        src_id = src_ids[0]
        dst_id = dst_ids[0]
        if src_id in positions and dst_id in positions:
            sx, sy = positions[src_id]
            dx, dy = positions[dst_id]
            color = "#888888"
            lw = 2.0
            style = "-"
            if rxn.get("is_rate_limiting"):
                color = COLORS["accent"]
                lw = 3.0

            # Draw arrow
            ax.annotate(
                "", xy=(dx, dy), xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=lw,
                    connectionstyle="arc3,rad=0.1",
                    shrinkA=25,
                    shrinkB=25,
                ),
            )

            # Add τ label if available
            tc = rxn.get("time_constant_s")
            if tc is not None:
                mx, my = (sx + dx) / 2, (sy + dy) / 2
                label = f"τ = {tc}s"
                bbox_color = "#FFF3E0" if rxn.get("is_rate_limiting") else "#F0F0F0"
                ax.text(
                    mx, my, label,
                    fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_color,
                              edgecolor="#CCCCCC", linewidth=0.5),
                    fontfamily="sans-serif",
                )

    # Draw species nodes
    for sid, (x, y) in positions.items():
        if sid not in species:
            continue
        sp = species[sid]
        role = sp.get("role", "intermediate")
        color = ROLE_COLORS.get(role, COLORS["intermediate"])

        # Larger node for target
        radius = 0.65 if role == "target" else 0.55
        circle = plt.Circle((x, y), radius, facecolor=color, edgecolor="white",
                            linewidth=2.5, zorder=5)
        ax.add_patch(circle)

        # Species label
        formula = sp.get("formula", sp.get("name", sid))
        # Render subscripts
        display = _render_formula(formula)
        fontsize = 11 if role == "target" else 10
        ax.text(x, y, display, fontsize=fontsize, fontweight="bold",
                ha="center", va="center", color="white", zorder=6,
                fontfamily="sans-serif",
                path_effects=[
                    path_effects.withStroke(linewidth=1, foreground=color)
                ])

        # Role label below
        role_label = role.replace("_", " ").title()
        if sid == hypothesis.get("catalyst", {}).get("target_product", ""):
            role_label = "TARGET"
        ax.text(x, y - radius - 0.25, role_label, fontsize=7,
                ha="center", va="top", color=COLORS["text_mid"],
                fontfamily="sans-serif", style="italic")

    # Title
    ax.text(6.0, 8.7, "CO₂ Electroreduction Reaction Network",
            fontsize=16, fontweight="bold", ha="center",
            color=COLORS["text_dark"], fontfamily="sans-serif")

    system = hypothesis.get("system", "")
    ax.text(6.0, 8.3, system, fontsize=10, ha="center",
            color=COLORS["text_mid"], fontfamily="sans-serif")

    # Legend
    legend_items = [
        ("Reactant", COLORS["reactant"]),
        ("Intermediate", COLORS["intermediate"]),
        ("C1 Product", COLORS["product_C1"]),
        ("C2 Product", COLORS["product_C2"]),
        ("Rate-Limiting", COLORS["accent"]),
    ]
    for i, (label, color) in enumerate(legend_items):
        ax.add_patch(plt.Circle((11.5, 7.5 - i * 0.5), 0.15,
                                facecolor=color, edgecolor="none", zorder=5))
        ax.text(11.9, 7.5 - i * 0.5, label, fontsize=8, va="center",
                color=COLORS["text_dark"], fontfamily="sans-serif")

    # Branch point annotations
    for bp in hypothesis.get("branch_points", []):
        sid = bp["species_id"]
        if sid in positions:
            x, y = positions[sid]
            ax.text(x, y + 0.9, "⚡ Branch Point", fontsize=7,
                    ha="center", color=COLORS["accent"],
                    fontweight="bold", fontfamily="sans-serif")

    fig.tight_layout()
    return _fig_to_base64(fig)


def _render_formula(formula: str) -> str:
    """Simple formula rendering for matplotlib (no real subscripts in text)."""
    # Clean up for display - matplotlib can't do inline subscripts in Text
    formula = formula.replace("⁻", "⁻").replace("₂", "2").replace("₃", "3")
    return formula


# =============================================================================
# PLOT: TEMPORAL EVOLUTION
# =============================================================================


def _plot_temporal_evolution(
    condition_data: Dict[str, List[Dict]],
    hypothesis: Dict,
    condition_label: str,
) -> str:
    """Plot species intensity evolution across scans, styled like sample.pdf page 3."""
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1.5])

    species_map = {s["id"]: s for s in hypothesis["species"]}
    scan_numbers = None

    # Top: all species evolution
    for sid, entries in condition_data.items():
        if sid not in species_map:
            continue
        sp = species_map[sid]
        role = sp.get("role", "intermediate")
        color = ROLE_COLORS.get(role, COLORS["intermediate"])
        scans = [e["scan_number"] for e in entries]
        intensities = [e["total_intensity"] for e in entries]
        if scan_numbers is None:
            scan_numbers = scans

        marker = "o" if role in ("target", "reactant") else "s"
        lw = 2.5 if role in ("target", "reactant") else 1.5
        ax_top.plot(scans, intensities, color=color, marker=marker,
                    linewidth=lw, label=f"{sp['formula']} ({role})",
                    markersize=6)

    ax_top.set_ylabel("Peak Intensity (counts)", fontsize=11, fontfamily="sans-serif")
    ax_top.set_title(f"Species Evolution — {condition_label}",
                     fontsize=14, fontweight="bold", fontfamily="sans-serif")
    ax_top.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_top.grid(True, alpha=0.3)
    ax_top.set_facecolor("#FAFAFA")

    # Bottom: C1 vs C2 total
    c1_ids = [s["id"] for s in hypothesis["species"] if s.get("role") == "product_C1"]
    c2_ids = [s["id"] for s in hypothesis["species"] if s.get("role") == "product_C2"]

    if scan_numbers:
        c1_total = []
        c2_total = []
        for i in range(len(scan_numbers)):
            c1_sum = sum(
                condition_data.get(sid, [{}] * (i + 1))[i].get("total_intensity", 0)
                for sid in c1_ids if sid in condition_data
            )
            c2_sum = sum(
                condition_data.get(sid, [{}] * (i + 1))[i].get("total_intensity", 0)
                for sid in c2_ids if sid in condition_data
            )
            c1_total.append(c1_sum)
            c2_total.append(c2_sum)

        ax_bot.plot(scan_numbers, c1_total, color=COLORS["product_C1"],
                    marker="o", linewidth=2, label="Total C1")
        ax_bot.fill_between(scan_numbers, c1_total, alpha=0.15, color=COLORS["product_C1"])
        ax_bot.plot(scan_numbers, c2_total, color=COLORS["product_C2"],
                    marker="s", linewidth=2, label="Total C2")
        ax_bot.fill_between(scan_numbers, c2_total, alpha=0.15, color=COLORS["product_C2"])

    ax_bot.set_xlabel("Scan Number", fontsize=11, fontfamily="sans-serif")
    ax_bot.set_ylabel("Cumulative Intensity", fontsize=11, fontfamily="sans-serif")
    ax_bot.set_title("C1 vs C2 Product Evolution", fontsize=12,
                     fontweight="bold", fontfamily="sans-serif")
    ax_bot.legend(loc="upper left", fontsize=9)
    ax_bot.grid(True, alpha=0.3)
    ax_bot.set_facecolor("#FAFAFA")

    fig.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# PLOT: HYPOTHESIS COMPARISON
# =============================================================================


def _plot_hypothesis_comparison(
    comparison_results: List[Dict],
    hypothesis: Dict,
) -> str:
    """Visual comparison: predicted vs observed species across conditions."""
    fig, ax = plt.subplots(figsize=(12, 6))

    species_ids = sorted(set(
        sid for c in comparison_results
        for sid in c.get("predicted_species", []) + c.get("detected_hypothesis_species", [])
    ))
    conditions = [c.get("condition", f"Condition {i}") for i, c in enumerate(comparison_results)]

    n_species = len(species_ids)
    n_conds = len(conditions)

    if n_species == 0 or n_conds == 0:
        ax.text(0.5, 0.5, "No comparison data available",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        fig.tight_layout()
        return _fig_to_base64(fig)

    # Create grid
    cell_w = 0.8
    cell_h = 0.6

    for j, comp in enumerate(comparison_results):
        predicted = set(comp.get("predicted_species", []))
        detected = set(comp.get("detected_hypothesis_species", []))
        confirmed = predicted & detected
        missing = predicted - detected
        unexpected = detected - predicted

        for i, sid in enumerate(species_ids):
            x = j * (cell_w + 0.3) + 0.5
            y = (n_species - 1 - i) * (cell_h + 0.15)

            if sid in confirmed:
                color = COLORS["confirmed"]
                symbol = "✓"
            elif sid in missing:
                color = COLORS["missing"]
                symbol = "✗"
            elif sid in unexpected:
                color = COLORS["unexpected"]
                symbol = "!"
            else:
                color = "#E0E0E0"
                symbol = "—"

            rect = FancyBboxPatch(
                (x, y), cell_w, cell_h,
                boxstyle="round,pad=0.05",
                facecolor=color, alpha=0.3,
                edgecolor=color, linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(x + cell_w / 2, y + cell_h / 2, symbol,
                    ha="center", va="center", fontsize=16,
                    fontweight="bold", color=color)

    # Row labels (species)
    for i, sid in enumerate(species_ids):
        y = (n_species - 1 - i) * (cell_h + 0.15) + cell_h / 2
        ax.text(0.3, y, sid, ha="right", va="center", fontsize=9,
                fontfamily="sans-serif")

    # Column labels (conditions)
    for j, cond in enumerate(conditions):
        x = j * (cell_w + 0.3) + 0.5 + cell_w / 2
        y = n_species * (cell_h + 0.15) + 0.1
        ax.text(x, y, cond, ha="center", va="bottom", fontsize=9,
                fontweight="bold", fontfamily="sans-serif", rotation=15)

    ax.set_xlim(-0.5, n_conds * (cell_w + 0.3) + 1)
    ax.set_ylim(-0.5, n_species * (cell_h + 0.15) + 0.8)
    ax.axis("off")
    ax.set_title("Hypothesis vs Observation", fontsize=14,
                 fontweight="bold", fontfamily="sans-serif", pad=20)

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=COLORS["confirmed"], alpha=0.5, label="Confirmed (predicted & detected)"),
        mpatches.Patch(facecolor=COLORS["missing"], alpha=0.5, label="Missing (predicted but not detected)"),
        mpatches.Patch(facecolor=COLORS["unexpected"], alpha=0.5, label="Unexpected (detected but not predicted)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    fig.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# PLOT: SPECTRUM WITH PEAK LABELS
# =============================================================================


def _plot_annotated_spectrum(
    spectrum: TxtrSpectrum,
    analysis_result: Dict,
    title: str = "",
) -> str:
    """Plot a spectrum with peaks labeled by species assignment."""
    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.array(spectrum.raman_shift_cm1)
    y = np.array(spectrum.intensity)
    ax.plot(x, y, color="#333333", linewidth=0.6, alpha=0.8)
    ax.fill_between(x, y, alpha=0.05, color="#333333")

    # Mark assigned peaks
    for peak in analysis_result.get("assigned", []):
        wn = peak["wavenumber_cm1"]
        inten = peak["intensity"]
        sid = peak["species_id"]
        category = peak.get("category", "hypothesis")

        if category == "hypothesis":
            color = COLORS["confirmed"]
        elif category == "electrolyte":
            color = COLORS["reactant"]
        elif category == "electrode":
            color = COLORS["unexpected"]
        else:
            color = COLORS["text_mid"]

        ax.axvline(wn, color=color, alpha=0.3, linewidth=1, linestyle="--")
        ax.annotate(
            f"{sid}\n{wn:.0f} cm⁻¹",
            xy=(wn, inten), xytext=(0, 20),
            textcoords="offset points",
            fontsize=7, ha="center", color=color,
            fontweight="bold", fontfamily="sans-serif",
            arrowprops=dict(arrowstyle="-", color=color, alpha=0.5),
        )

    # Mark unassigned peaks
    for peak in analysis_result.get("unassigned", []):
        wn = peak["wavenumber_cm1"]
        inten = peak["intensity"]
        ax.axvline(wn, color=COLORS["accent"], alpha=0.4, linewidth=1, linestyle=":")
        label = f"? {wn:.0f} cm⁻¹"
        if peak.get("nearest_reference"):
            label += f"\n(near {peak['nearest_reference']})"
        ax.annotate(
            label, xy=(wn, inten), xytext=(0, -25),
            textcoords="offset points",
            fontsize=6, ha="center", color=COLORS["accent"],
            fontfamily="sans-serif",
            arrowprops=dict(arrowstyle="-", color=COLORS["accent"], alpha=0.5),
        )

    ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=11, fontfamily="sans-serif")
    ax.set_ylabel("Intensity (counts)", fontsize=11, fontfamily="sans-serif")
    ax.set_title(title or "Annotated Raman Spectrum", fontsize=13,
                 fontweight="bold", fontfamily="sans-serif")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.2)
    ax.set_facecolor("#FAFAFA")

    fig.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
@page {{
    size: A4 landscape;
    margin: 1.5cm;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    color: #1A1A2E;
    background: white;
    line-height: 1.5;
}}
.page {{
    page-break-after: always;
    padding: 20px 30px;
    min-height: 680px;
}}
.page:last-child {{ page-break-after: avoid; }}

/* Cover */
.cover {{
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    min-height: 680px;
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
    color: white;
    border-radius: 4px;
}}
.cover h1 {{
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 12px;
    letter-spacing: -0.5px;
}}
.cover .subtitle {{
    font-size: 16px;
    opacity: 0.85;
    margin-bottom: 30px;
}}
.cover .meta {{
    font-size: 12px;
    opacity: 0.6;
    margin-top: 40px;
}}
.cover .system-badge {{
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 6px 20px;
    font-size: 14px;
    margin-bottom: 20px;
}}

/* Section headers */
.section-header {{
    font-size: 20px;
    font-weight: 700;
    color: #1A1A2E;
    border-bottom: 3px solid #2E86AB;
    padding-bottom: 6px;
    margin-bottom: 16px;
}}
.section-subheader {{
    font-size: 14px;
    color: #4A4A6A;
    margin-bottom: 12px;
}}

/* Figures */
.figure-container {{
    text-align: center;
    margin: 10px 0;
}}
.figure-container img {{
    max-width: 100%;
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
.figure-caption {{
    font-size: 11px;
    color: #4A4A6A;
    margin-top: 6px;
    font-style: italic;
}}

/* Verdict boxes */
.verdict-box {{
    padding: 14px 18px;
    border-radius: 6px;
    margin: 12px 0;
    font-size: 13px;
}}
.verdict-consistent {{
    background: #E8F5E9;
    border-left: 4px solid #2D6A4F;
    color: #1B5E20;
}}
.verdict-partial {{
    background: #FFF8E1;
    border-left: 4px solid #F18F01;
    color: #E65100;
}}
.verdict-inconsistent {{
    background: #FFEBEE;
    border-left: 4px solid #E63946;
    color: #B71C1C;
}}

/* Blind-spot table */
.blindspot-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin: 10px 0;
}}
.blindspot-table th {{
    background: #2E86AB;
    color: white;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
}}
.blindspot-table td {{
    padding: 6px 12px;
    border-bottom: 1px solid #E0E0E0;
}}
.blindspot-table tr:nth-child(even) {{
    background: #F8F9FA;
}}
.llm-flag {{
    display: inline-block;
    background: #FFF3E0;
    color: #E65100;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 600;
}}

/* Prediction cards */
.pred-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 14px;
    margin: 12px 0;
}}
.pred-card {{
    border: 1px solid #DEE2E6;
    border-radius: 6px;
    padding: 14px;
    background: #FAFAFA;
}}
.pred-card h4 {{
    font-size: 13px;
    color: #2E86AB;
    margin-bottom: 6px;
}}
.pred-card .products {{
    font-size: 12px;
    color: #333;
    margin: 4px 0;
}}
.pred-card .notes {{
    font-size: 11px;
    color: #666;
    margin-top: 6px;
    line-height: 1.4;
}}

/* Footer */
.page-footer {{
    font-size: 9px;
    color: #999;
    text-align: right;
    margin-top: auto;
    padding-top: 10px;
}}
</style>
</head>
<body>
{content}
</body>
</html>"""


def generate_report(
    data_root: str,
    hypothesis_path: str,
    output_html: str,
    output_pdf: Optional[str] = None,
    condition_filter: Optional[List[str]] = None,
) -> str:
    """
    Generate the full hypothesis-testing report.

    Args:
        data_root: Experiment data directory (contains date/condition folders).
        hypothesis_path: Path to hypothesis JSON.
        output_html: Output HTML file path.
        output_pdf: Optional PDF output path (requires WeasyPrint).
        condition_filter: Only include these conditions (None = all).

    Returns:
        Path to generated HTML file.
    """
    hypothesis = load_hypothesis(hypothesis_path)

    # Find experiment data
    experiment_dirs = _find_condition_dirs(data_root)
    if condition_filter:
        experiment_dirs = {
            k: v for k, v in experiment_dirs.items()
            if any(cf.lower() in k.lower() for cf in condition_filter)
        }

    if not experiment_dirs:
        raise ValueError(f"No experiment conditions found under {data_root}")

    # === Build report pages ===
    pages = []

    # Page 1: Cover
    pages.append(_build_cover_page(hypothesis))

    # Page 2: Reaction Network
    network_img = _plot_reaction_network(hypothesis)
    pages.append(_build_figure_page(
        "Reaction Network",
        f"Hypothesized CO₂ electroreduction pathway on {hypothesis.get('catalyst', {}).get('name', 'catalyst')}",
        network_img,
        "Reaction network showing species, pathways, and time constants (τ). "
        "Rate-limiting step highlighted in red. Branch points marked with ⚡.",
    ))

    # Page 3: Predictions
    pages.append(_build_predictions_page(hypothesis))

    # Pages 4+: Per-condition analysis
    comparison_results = []
    for cond_name, cond_dir in sorted(experiment_dirs.items()):
        scan_files = _get_scan_files(cond_dir)
        if not scan_files:
            continue

        # Analyze first scan for annotated spectrum
        first_scan_num, first_scan_path = scan_files[0]
        spectrum = parse_txtr(first_scan_path)
        analysis = analyze_spectrum(spectrum.raman_shift_cm1, spectrum.intensity)
        analysis_dict = peak_analysis_to_dict(analysis)

        # Annotated spectrum
        spec_img = _plot_annotated_spectrum(
            spectrum, analysis_dict,
            title=f"Scan {first_scan_num} — {cond_name}",
        )
        pages.append(_build_spectrum_page(cond_name, first_scan_num, analysis_dict, spec_img))

        # Temporal evolution (all species from hypothesis)
        all_sids = [s["id"] for s in hypothesis["species"]]
        evolution_data = _compute_evolution(cond_dir, scan_files, all_sids)
        if evolution_data:
            evo_img = _plot_temporal_evolution(evolution_data, hypothesis, cond_name)
            pages.append(_build_figure_page(
                f"Temporal Evolution — {cond_name}",
                f"Species intensity tracking across {len(scan_files)} scans",
                evo_img,
                "Top: individual species evolution. Bottom: cumulative C1 vs C2 product intensities.",
            ))

        # Hypothesis comparison
        comp = _build_comparison_data(cond_name, analysis_dict, hypothesis)
        comparison_results.append(comp)

    # Hypothesis comparison matrix
    if comparison_results:
        comp_img = _plot_hypothesis_comparison(comparison_results, hypothesis)
        pages.append(_build_comparison_page(comparison_results, comp_img))

    # Blind-spot guard summary
    pages.append(_build_blindspot_page(experiment_dirs, hypothesis))

    # === Render HTML ===
    content = "\n".join(pages)
    html = _HTML_TEMPLATE.format(
        title=f"Hypothesis Test Report — {hypothesis.get('title', '')}",
        content=content,
    )

    with open(output_html, "w") as f:
        f.write(html)
    print(f"HTML report: {output_html}")

    # === Convert to PDF ===
    if output_pdf:
        try:
            from weasyprint import HTML
            HTML(filename=output_html).write_pdf(output_pdf)
            print(f"PDF report: {output_pdf}")
        except Exception as e:
            print(f"WARNING: PDF generation failed ({e}). HTML report is available.")

    return output_html


# =============================================================================
# PAGE BUILDERS
# =============================================================================


def _build_cover_page(hypothesis: Dict) -> str:
    title = hypothesis.get("title", "Hypothesis Test Report")
    system = hypothesis.get("system", "")
    date = hypothesis.get("created_date", "")
    catalyst = hypothesis.get("catalyst", {})
    n_species = len(hypothesis.get("species", []))
    n_reactions = len(hypothesis.get("reactions", []))
    n_predictions = len(hypothesis.get("predictions", []))

    return f"""
    <div class="page cover">
        <div class="system-badge">{system}</div>
        <h1>{title}</h1>
        <div class="subtitle">Hypothesis Testing Report</div>
        <div class="subtitle" style="font-size: 13px; opacity: 0.7;">
            Catalyst: {catalyst.get('name', 'N/A')} |
            {n_species} species | {n_reactions} reactions | {n_predictions} condition predictions
        </div>
        <div class="meta">
            Generated: {date}<br>
            Source: {hypothesis.get('source', 'N/A')}
        </div>
    </div>"""


def _build_figure_page(title: str, subtitle: str, img_base64: str, caption: str) -> str:
    return f"""
    <div class="page">
        <div class="section-header">{title}</div>
        <div class="section-subheader">{subtitle}</div>
        <div class="figure-container">
            <img src="data:image/png;base64,{img_base64}" />
            <div class="figure-caption">{caption}</div>
        </div>
    </div>"""


def _build_predictions_page(hypothesis: Dict) -> str:
    predictions = hypothesis.get("predictions", [])
    cards = ""
    for pred in predictions:
        products = ", ".join(pred.get("dominant_products", []))
        temporals = ""
        for t in pred.get("temporal_expectations", []):
            temporals += f"<br>• {t['species']}: {t['behavior']}"
        notes = pred.get("selectivity_notes", "")
        cards += f"""
        <div class="pred-card">
            <h4>{pred['condition_label']} ({pred['potential_vs_rhe_V']}V vs RHE)</h4>
            <div class="products"><strong>Dominant:</strong> {products}</div>
            <div class="notes"><strong>Temporal:</strong>{temporals}</div>
            <div class="notes" style="margin-top: 8px;">{notes}</div>
        </div>"""

    network_notes = hypothesis.get("network_notes", "")
    return f"""
    <div class="page">
        <div class="section-header">Hypothesis Predictions</div>
        <div class="section-subheader">Expected behavior at each experimental condition</div>
        <div class="pred-grid">{cards}</div>
        <div style="margin-top: 16px; font-size: 11px; color: #666; line-height: 1.5; padding: 10px; background: #F8F9FA; border-radius: 4px;">
            <strong>Network Notes:</strong> {network_notes}
        </div>
    </div>"""


def _build_spectrum_page(
    condition: str, scan_number: int, analysis: Dict, img_base64: str,
) -> str:
    n_assigned = analysis.get("n_assigned", 0)
    n_unassigned = analysis.get("n_unassigned", 0)
    hyp_species = ", ".join(analysis.get("hypothesis_species_detected", [])) or "None"
    guard_species = ", ".join(analysis.get("guard_species_detected", [])) or "None"
    residual = analysis.get("residual_energy_pct", 0)

    assigned_rows = ""
    for a in sorted(analysis.get("assigned", []), key=lambda x: -x.get("intensity", 0)):
        conf_color = {"high": "#2D6A4F", "medium": "#F18F01", "low": "#E63946"}.get(
            a.get("confidence", "low"), "#666"
        )
        assigned_rows += f"""
        <tr>
            <td>{a['wavenumber_cm1']:.1f}</td>
            <td>{a['intensity']:.0f}</td>
            <td>{a['snr']:.1f}</td>
            <td>{a['species_id']}</td>
            <td>{a['assignment']}</td>
            <td style="color: {conf_color}; font-weight: 600;">{a['confidence']}</td>
        </tr>"""

    return f"""
    <div class="page">
        <div class="section-header">Spectral Analysis — {condition}, Scan {scan_number}</div>
        <div class="section-subheader">
            {n_assigned} peaks assigned | {n_unassigned} unassigned |
            Hypothesis species: {hyp_species} | Guard species: {guard_species} |
            Residual energy: {residual:.1f}%
        </div>
        <div class="figure-container">
            <img src="data:image/png;base64,{img_base64}" />
        </div>
        <table class="blindspot-table" style="margin-top: 10px;">
            <tr><th>Wavenumber</th><th>Intensity</th><th>SNR</th><th>Species</th><th>Assignment</th><th>Confidence</th></tr>
            {assigned_rows}
        </table>
    </div>"""


def _build_comparison_page(comparisons: List[Dict], img_base64: str) -> str:
    verdicts = ""
    for comp in comparisons:
        verdict = comp.get("verdict", "")
        vclass = "verdict-consistent"
        if "INCONSISTENT" in verdict:
            vclass = "verdict-inconsistent"
        elif "PARTIAL" in verdict:
            vclass = "verdict-partial"
        verdicts += f"""
        <div class="verdict-box {vclass}">
            <strong>{comp['condition']}:</strong> {verdict}
        </div>"""

    return f"""
    <div class="page">
        <div class="section-header">Hypothesis vs Observation</div>
        <div class="section-subheader">Cross-referencing predictions against experimental data</div>
        <div class="figure-container">
            <img src="data:image/png;base64,{img_base64}" />
            <div class="figure-caption">
                ✓ = Predicted and detected | ✗ = Predicted but not detected | ! = Detected but not predicted
            </div>
        </div>
        {verdicts}
    </div>"""


def _build_blindspot_page(experiment_dirs: Dict[str, str], hypothesis: Dict) -> str:
    """Build blind-spot guard summary across all conditions."""
    all_unassigned = []

    for cond_name, cond_dir in sorted(experiment_dirs.items()):
        scan_files = _get_scan_files(cond_dir)
        for scan_num, scan_path in scan_files[:3]:  # First 3 scans per condition
            try:
                spectrum = parse_txtr(scan_path)
                analysis = analyze_spectrum(spectrum.raman_shift_cm1, spectrum.intensity)
                result = peak_analysis_to_dict(analysis)
                for u in result.get("unassigned", []):
                    all_unassigned.append({
                        "condition": cond_name,
                        "scan": scan_num,
                        **u,
                    })
            except Exception:
                continue

    rows = ""
    for u in sorted(all_unassigned, key=lambda x: -x.get("snr", 0)):
        llm_flag = '<span class="llm-flag">LLM Query</span>' if u.get("llm_query_suggested") else ""
        rows += f"""
        <tr>
            <td>{u['condition']}</td>
            <td>Scan {u['scan']}</td>
            <td>{u['wavenumber_cm1']:.1f}</td>
            <td>{u['intensity']:.0f}</td>
            <td>{u['snr']:.1f}</td>
            <td>{u.get('nearest_reference', '—')}</td>
            <td>{u.get('nearest_distance_cm1', 0):.0f}</td>
            <td>{llm_flag}</td>
        </tr>"""

    n_total = len(all_unassigned)
    n_llm = sum(1 for u in all_unassigned if u.get("llm_query_suggested"))

    return f"""
    <div class="page">
        <div class="section-header">Blind-Spot Guard</div>
        <div class="section-subheader">
            Unassigned peaks across all conditions — {n_total} peaks not matching any known species.
            {n_llm} flagged for LLM interpretation (SNR &gt; 4).
        </div>
        <table class="blindspot-table">
            <tr>
                <th>Condition</th><th>Scan</th><th>Wavenumber (cm⁻¹)</th>
                <th>Intensity</th><th>SNR</th><th>Nearest Species</th>
                <th>Distance (cm⁻¹)</th><th>Flag</th>
            </tr>
            {rows if rows else '<tr><td colspan="8" style="text-align:center; color:#999;">No unassigned peaks found — all detected features accounted for.</td></tr>'}
        </table>
        <div style="margin-top: 16px; padding: 12px; background: #FFF8E1; border-radius: 4px; font-size: 12px; border-left: 4px solid #F18F01;">
            <strong>Interpretation guide:</strong> Peaks flagged for LLM query have high SNR
            but don't match any species in the hypothesis or reference library.
            These may indicate unexpected reaction products, catalyst degradation,
            or species not considered in the original hypothesis.
            Consider querying with the wavenumber, experimental context (potential, electrolyte, electrode),
            and peak characteristics for identification.
        </div>
    </div>"""


# =============================================================================
# DATA HELPERS
# =============================================================================


def _find_condition_dirs(data_root: str) -> Dict[str, str]:
    """Find condition directories under data_root (handles date subdirectories)."""
    result = {}
    for entry in os.listdir(data_root):
        full = os.path.join(data_root, entry)
        if not os.path.isdir(full) or entry.startswith("."):
            continue
        # Check if it's a date directory
        if re.match(r"\d{4}-\d{2}-\d{2}", entry):
            for sub in os.listdir(full):
                sub_path = os.path.join(full, sub)
                if os.path.isdir(sub_path) and not sub.startswith("."):
                    txtr = [f for f in os.listdir(sub_path) if f.lower().endswith(".txtr")]
                    if txtr:
                        result[sub] = sub_path
        else:
            txtr = [f for f in os.listdir(full) if f.lower().endswith(".txtr")]
            if txtr:
                result[entry] = full
    return result


def _get_scan_files(cond_dir: str) -> List[Tuple[int, str]]:
    """Get sorted list of (scan_number, filepath) for electrolysis scans."""
    scans = []
    for f in os.listdir(cond_dir):
        if not f.lower().endswith(".txtr"):
            continue
        match = re.search(r"[Ss]can\s*(\d+)", f)
        if match:
            scans.append((int(match.group(1)), os.path.join(cond_dir, f)))
    return sorted(scans, key=lambda x: x[0])


def _compute_evolution(
    cond_dir: str,
    scan_files: List[Tuple[int, str]],
    species_ids: List[str],
) -> Dict[str, List[Dict]]:
    """Compute temporal evolution of species intensities."""
    evolution = {sid: [] for sid in species_ids}
    for scan_num, filepath in scan_files:
        try:
            spectrum = parse_txtr(filepath)
            analysis = analyze_spectrum(spectrum.raman_shift_cm1, spectrum.intensity)

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
        except Exception:
            for sid in species_ids:
                evolution[sid].append({
                    "scan_number": scan_num,
                    "total_intensity": 0.0,
                    "detected": False,
                })
    return evolution


def _build_comparison_data(
    condition: str, analysis: Dict, hypothesis: Dict,
) -> Dict:
    """Build hypothesis comparison data for one condition."""
    # Extract potential from condition name
    pot_match = re.search(r"(-?\d+\.?\d*)\s*V", condition)
    condition_potential = float(pot_match.group(1)) if pot_match else None

    predictions = hypothesis.get("predictions", [])
    matched_pred = None
    if condition_potential is not None and predictions:
        matched_pred = min(
            predictions,
            key=lambda p: abs(p["potential_vs_rhe_V"] - condition_potential),
        )

    detected_hyp = set(analysis.get("hypothesis_species_detected", []))
    predicted = set()
    if matched_pred:
        predicted = set(
            matched_pred.get("dominant_products", []) +
            matched_pred.get("expected_intermediates", [])
        )

    confirmed = detected_hyp & predicted
    missing = predicted - detected_hyp
    unexpected = detected_hyp - predicted

    verdict = ""
    if confirmed and not missing:
        verdict = "CONSISTENT"
    elif confirmed:
        verdict = f"PARTIAL: {len(missing)} missing ({', '.join(sorted(missing))})"
    elif predicted:
        verdict = f"INCONSISTENT: None of {', '.join(sorted(predicted))} detected"
    else:
        verdict = "No predictions available"

    return {
        "condition": condition,
        "predicted_species": sorted(predicted),
        "detected_hypothesis_species": sorted(detected_hyp),
        "confirmed": sorted(confirmed),
        "missing": sorted(missing),
        "unexpected": sorted(unexpected),
        "verdict": verdict,
    }


# =============================================================================
# CLI
# =============================================================================


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate CO2RR Hypothesis Test Report")
    parser.add_argument("data_root", help="Experiment data directory")
    parser.add_argument("hypothesis", help="Path to hypothesis JSON")
    parser.add_argument("--output-html", default="report.html")
    parser.add_argument("--output-pdf", default=None)
    parser.add_argument("--conditions", nargs="+", help="Filter to specific conditions")
    args = parser.parse_args()

    generate_report(
        data_root=args.data_root,
        hypothesis_path=args.hypothesis,
        output_html=args.output_html,
        output_pdf=args.output_pdf,
        condition_filter=args.conditions,
    )


if __name__ == "__main__":
    main()
