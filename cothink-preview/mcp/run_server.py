#!/usr/bin/env python3
"""
Start the ChemReasoner MCP server with an ngrok tunnel.

Usage:
    python run_server.py --data-root ../demo_data/co2_acetate --port 8765

Setup:
    pip install mcp[sse] pyngrok scipy numpy
    export NGROK_AUTHTOKEN=<your-token>

Directory structure expected:
    demo_data/co2_acetate/
        hypothesis/
            co2rr_cufe_hypothesis.json
        2026-03-24/
            Cu-Fe 41_-0.56V/
                CF41_-1.2V_Scan 1 .txtr
                ...
            Cu-Fe 41_-0.76V/
                ...
        analysis/  (created by MCP server)
"""

import argparse
import os
import signal
import sys
import subprocess
import time


def run_mcp_server(data_root: str, port: int):
    """Start the MCP server as a subprocess."""
    env = os.environ.copy()
    env["CO2RR_DATA_ROOT"] = data_root
    proc = subprocess.Popen(
        [
            sys.executable, "mcp_server.py",
            "--data-root", data_root,
            "--transport", "sse",
            "--port", str(port),
        ],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
    )
    return proc


def start_ngrok(port: int) -> str:
    """Start ngrok tunnel and return the SSE URL."""
    from pyngrok import ngrok

    token = os.environ.get("NGROK_AUTHTOKEN")
    if token:
        ngrok.set_auth_token(token)

    tunnel = ngrok.connect(port, "http")
    return f"{tunnel.public_url}/sse"


def validate_data_root(data_root: str):
    """Validate the data directory structure."""
    if not os.path.isdir(data_root):
        print(f"ERROR: Data root not found: {data_root}")
        sys.exit(1)

    # Check for hypothesis
    hyp_dir = os.path.join(data_root, "hypothesis")
    if os.path.isdir(hyp_dir):
        hyp_files = [f for f in os.listdir(hyp_dir) if f.endswith(".json")]
        if hyp_files:
            print(f"  Hypothesis: {hyp_files[0]}")
        else:
            print("  WARNING: hypothesis/ directory exists but no .json files found")
    else:
        print("  WARNING: No hypothesis/ directory. Create one with a hypothesis JSON.")

    # Check for experiment directories
    date_dirs = [
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
        and len(d) == 10 and d[4] == "-"
    ]
    if date_dirs:
        for dd in sorted(date_dirs):
            dd_path = os.path.join(data_root, dd)
            conds = [
                c for c in os.listdir(dd_path)
                if os.path.isdir(os.path.join(dd_path, c)) and not c.startswith(".")
            ]
            print(f"  Experiment {dd}: {len(conds)} conditions")
            for c in sorted(conds):
                cpath = os.path.join(dd_path, c)
                txtr = [f for f in os.listdir(cpath) if f.lower().endswith(".txtr")]
                print(f"    {c}: {len(txtr)} .txtr files")
    else:
        print("  WARNING: No date-named experiment directories found")


def main():
    parser = argparse.ArgumentParser(
        description="Start ChemReasoner MCP server"
    )
    parser.add_argument(
        "--data-root",
        default="../demo_data/co2_acetate",
        help="Data directory (contains hypothesis/ and date directories)",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-ngrok", action="store_true",
        help="Run local only, no ngrok tunnel",
    )
    args = parser.parse_args()

    print(f"Data root: {args.data_root}")
    validate_data_root(args.data_root)

    print(f"\nStarting MCP server on port {args.port}...")
    server_proc = run_mcp_server(args.data_root, args.port)
    time.sleep(2)

    if server_proc.poll() is not None:
        print("ERROR: MCP server failed to start")
        sys.exit(1)

    sse_url = f"http://localhost:{args.port}/sse"

    if not args.no_ngrok:
        print("Starting ngrok tunnel...")
        try:
            sse_url = start_ngrok(args.port)
        except Exception as e:
            print(f"WARNING: ngrok failed ({e}). Using local URL only.")

    print("\n" + "=" * 70)
    print("CHEMREASONER MCP SERVER RUNNING")
    print("=" * 70)
    print(f"\n  MCP SSE URL: {sse_url}")
    print(f"\n  Add as MCP connector in Claude.ai:")
    print(f"    Name: ChemReasoner")
    print(f"    URL:  {sse_url}")
    print(f"\n  Local: http://localhost:{args.port}/sse")
    print(f"\n  Press Ctrl+C to stop.\n")

    def shutdown(sig, frame):
        print("\nShutting down...")
        server_proc.terminate()
        try:
            from pyngrok import ngrok
            ngrok.kill()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    server_proc.wait()


if __name__ == "__main__":
    main()
