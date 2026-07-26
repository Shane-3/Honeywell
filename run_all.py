"""
One-Command End-to-End Orchestrator
==================================
Runs full pipeline from clean state:
1. Generates 160k synthetic log events (Phase 1)
2. Trains Baseline Profiler, Markov Sequence Model, and Anomaly Classifier (Phase 2-5)
3. Evaluates models and outputs figures & metrics summary (Phase 6)
4. Verifies Analyst Dashboard API endpoints (Phase 7)
5. Optionally launches live FastAPI web server on http://localhost:8000 (--server)
"""

import argparse
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_command(cmd, cwd=SCRIPT_DIR):
    print(f"\n[EXEC] Running: {cmd} (in {cwd})", flush=True)
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        print(f"[!] Command failed with exit code: {res.returncode}", flush=True)
        sys.exit(res.returncode)

def main():
    parser = argparse.ArgumentParser(description="Honeywell AI Anomaly Detection One-Command Pipeline")
    parser.add_argument("--server", action="store_true", help="Launch FastAPI web server after pipeline execution")
    parser.add_argument("--serve-only", action="store_true", help="Launch dashboard web server instantly using pre-trained data & models")
    parser.add_argument("--test-only", action="store_true", help="Run full pipeline & endpoint tests without launching server")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data generation")
    parser.add_argument("--port", type=int, default=8000, help="Port for dashboard web server")

    args = parser.parse_args()

    # If serve-only is requested, jump straight to launching the server
    if args.serve_only:
        print(f"\n>>> Launching Analyst Dashboard Web Server instantly on http://localhost:{args.port}...")
        import uvicorn
        uvicorn.run("dashboard.app:app", host="127.0.0.1", port=args.port, reload=True)
        return

    print("=" * 70)
    print("HONEYWELL AI CAMPUS CONNECT — END-TO-END PIPELINE")
    print("=" * 70)

    # Step 1: Generate Synthetic Data
    print("\n>>> STEP 1: Generating Synthetic Logs...", flush=True)
    run_command(f"python -u data/generate_data.py --seed {args.seed}")

    # Step 2: Train Models & Score Pipeline
    print("\n>>> STEP 2: Training Models & Scoring Events...", flush=True)
    run_command("python -u models/train.py")

    # Step 3: Evaluate System Performance
    print("\n>>> STEP 3: Evaluating Performance & Generating Reports...", flush=True)
    run_command("python -u evaluation/evaluate.py")

    # Step 4: Verify Dashboard Endpoints
    print("\n>>> STEP 4: Testing Analyst Dashboard API Endpoints...", flush=True)
    dashboard_dir = os.path.join(SCRIPT_DIR, "dashboard")
    run_command("python -u test_dashboard.py", cwd=dashboard_dir)

    print("\n" + "=" * 70)
    print("[OK] PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    # Step 5: Launch Dashboard Server if requested
    if args.server:
        print(f"\n>>> STEP 5: Launching Analyst Dashboard Web Server on http://localhost:{args.port}...")
        import uvicorn
        uvicorn.run("dashboard.app:app", host="127.0.0.1", port=args.port, reload=True)
    elif not args.test_only:
        print("\nTo launch the live analyst dashboard web server, run:")
        print(f"    python run_all.py --server")

if __name__ == "__main__":
    main()
