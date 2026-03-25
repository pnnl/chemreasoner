#!/usr/bin/env python3
"""
ChemReasoner MCP Connector — Setup

Validates the environment, checks dependencies, and initializes the
data directory structure.

Usage:
    python setup.py                    # Check everything
    python setup.py --init-demo        # Create demo directory structure
    python setup.py --check-only       # Just validate, don't create anything
"""

import argparse
import importlib
import os
import shutil
import sys


REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("weasyprint", "weasyprint"),
    ("mcp", "mcp"),
]

OPTIONAL_PACKAGES = [
    ("pyngrok", "pyngrok"),
]


def check_python():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print(f"  ✗ Python {v.major}.{v.minor}.{v.micro} — need 3.10+")
        return False
    print(f"  ✓ Python {v.major}.{v.minor}.{v.micro}")
    return True


def check_packages():
    ok = True
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
            print(f"  ✓ {pip_name}")
        except ImportError:
            print(f"  ✗ {pip_name} — pip install {pip_name}")
            ok = False

    for import_name, pip_name in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(import_name)
            print(f"  ✓ {pip_name} (optional)")
        except ImportError:
            print(f"  ○ {pip_name} (optional — needed for ngrok tunnel)")

    return ok


def check_data_root(data_root: str):
    if not os.path.isdir(data_root):
        print(f"  ○ Data root not found: {data_root}")
        print(f"    Run with --init-demo to create a sample structure")
        return False

    # Check for hypothesis
    hyp_dir = os.path.join(data_root, "hypothesis")
    if os.path.isdir(hyp_dir):
        jsons = [f for f in os.listdir(hyp_dir) if f.endswith(".json")]
        if jsons:
            print(f"  ✓ Hypothesis: {jsons[0]}")
        else:
            print(f"  ✗ hypothesis/ directory exists but no .json files")
            print(f"    Copy your hypothesis JSON into {hyp_dir}/")
    else:
        print(f"  ○ No hypothesis/ directory at {hyp_dir}")

    # Check for experiment data
    date_dirs = []
    for entry in os.listdir(data_root):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full) and len(entry) == 10 and entry[4] == "-":
            date_dirs.append(entry)

    if date_dirs:
        for dd in sorted(date_dirs):
            dd_path = os.path.join(data_root, dd)
            conditions = [
                c for c in os.listdir(dd_path)
                if os.path.isdir(os.path.join(dd_path, c)) and not c.startswith(".")
            ]
            n_txtr = 0
            for c in conditions:
                cpath = os.path.join(dd_path, c)
                n_txtr += len([f for f in os.listdir(cpath) if f.lower().endswith(".txtr")])
            print(f"  ✓ Experiment {dd}: {len(conditions)} conditions, {n_txtr} spectra")
    else:
        print(f"  ○ No experiment data found (expected date-named directories)")

    return True


def check_ngrok():
    token = os.environ.get("NGROK_AUTHTOKEN")
    if token:
        print(f"  ✓ NGROK_AUTHTOKEN is set")
        return True
    else:
        print(f"  ○ NGROK_AUTHTOKEN not set — server will run local-only")
        print(f"    Get a token at https://dashboard.ngrok.com/get-started/your-authtoken")
        return False


def init_demo(data_root: str):
    """Create the demo directory structure."""
    hyp_dir = os.path.join(data_root, "hypothesis")
    os.makedirs(hyp_dir, exist_ok=True)

    # Copy hypothesis if it exists in the hypotheses/ directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_hyp = os.path.join(script_dir, "hypotheses", "co2rr_cufe_acetate.json")
    if os.path.exists(repo_hyp):
        dest = os.path.join(hyp_dir, "co2rr_cufe_hypothesis.json")
        if not os.path.exists(dest):
            shutil.copy2(repo_hyp, dest)
            print(f"  Copied hypothesis to {dest}")
        else:
            print(f"  Hypothesis already exists at {dest}")
    else:
        print(f"  No hypothesis JSON found in repo root to copy")
        print(f"  Place your hypothesis JSON in {hyp_dir}/")

    # Create example date directory
    example_dir = os.path.join(data_root, "YYYY-MM-DD")
    example_cond = os.path.join(example_dir, "Sample_Condition")
    os.makedirs(example_cond, exist_ok=True)
    readme = os.path.join(example_cond, "PUT_YOUR_TXTR_FILES_HERE.txt")
    with open(readme, "w") as f:
        f.write("Place your .txtr Raman spectra files in condition folders.\n")
        f.write("Rename this directory to your experiment date (e.g., 2026-03-24).\n")
        f.write("Rename this folder to your condition (e.g., Cu-Fe 41_-0.56V).\n")

    print(f"  Created {data_root}/ with example structure")
    print(f"  Next: replace YYYY-MM-DD/ with your actual experiment data")


def main():
    parser = argparse.ArgumentParser(description="ChemReasoner MCP Connector Setup")
    parser.add_argument(
        "--data-root", default="../demo_data/co2_acetate",
        help="Path to data directory (default: ../demo_data/co2_acetate)",
    )
    parser.add_argument("--init-demo", action="store_true", help="Create demo directory structure")
    parser.add_argument("--check-only", action="store_true", help="Only validate, don't create")
    args = parser.parse_args()

    print("\nChemReasoner MCP Connector — Setup Check")
    print("=" * 45)

    print("\n1. Python")
    py_ok = check_python()

    print("\n2. Packages")
    pkg_ok = check_packages()

    print("\n3. Data Directory")
    if args.init_demo and not args.check_only:
        init_demo(args.data_root)
    data_ok = check_data_root(args.data_root)

    print("\n4. ngrok (for remote access)")
    ngrok_ok = check_ngrok()

    print("\n" + "=" * 45)
    if py_ok and pkg_ok:
        print("Ready to run. Start with:")
        print(f"  python run_server.py --data-root {args.data_root}")
        if not ngrok_ok:
            print(f"  (add --no-ngrok for local-only mode)")
    else:
        print("Fix the issues above before running.")
        missing = []
        for import_name, pip_name in REQUIRED_PACKAGES:
            try:
                importlib.import_module(import_name)
            except ImportError:
                missing.append(pip_name)
        if missing:
            print(f"\n  pip install {' '.join(missing)}")

    print()


if __name__ == "__main__":
    main()
