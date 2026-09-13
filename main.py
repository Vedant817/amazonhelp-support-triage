"""
Local entrypoint for the @AmazonHelp support-agent prototype.

Running:
    python main.py                     # Start the local server & UI (default)
    python main.py serve               # Start the local server & UI
    python main.py test                # Run complete unit & integration test suite
    python main.py benchmark           # Run the 200 Golden Evaluation benchmark
    python main.py stress              # Run the safety regression suite
    python main.py review --port 8001  # Start the independent label-review desk
"""

import sys
import os
import argparse
import subprocess
import unittest

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def run_serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    generation_mode: str | None = None,
):
    """Start the local FastAPI REST API and web dashboard."""
    import uvicorn
    if generation_mode is not None:
        os.environ["SUPPORT_GENERATION_MODE"] = generation_mode
    active_mode = os.getenv("SUPPORT_GENERATION_MODE", "offline")
    print("\n" + "=" * 80)
    print("      @AmazonHelp AI Support Agent - Local Prototype")
    print("=" * 80)
    print(f"  Dashboard:        http://{host}:{port}/")
    print(f"  OpenAPI:          http://{host}:{port}/docs")
    print(f"  Health:           http://{host}:{port}/api/v1/health")
    print(f"  Metrics:          http://{host}:{port}/metrics")
    print(f"  Generation mode:  {active_mode}")
    print("=" * 80 + "\n")
    uvicorn.run("src.api:app", host=host, port=port, reload=reload)


def run_test():
    """Run the discovered unit and integration tests."""
    print("\n" + "=" * 80)
    print("      Running Full Test Suite (Unit + Integration)")
    print("=" * 80 + "\n")
    suite = unittest.defaultTestLoader.discover(os.path.join(REPO_ROOT, "tests"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print("\n[ERROR] Test suite failed!")
        sys.exit(1)
    print("\n[SUCCESS] All tests passed cleanly!\n")


def run_review(host: str = "127.0.0.1", port: int = 8001):
    """Start the local-only independent annotation interface."""
    import uvicorn
    print(f"Review desk: http://{host}:{port}/")
    print("Predictions and gold labels are hidden from reviewers.")
    uvicorn.run("src.review_app:app", host=host, port=port, reload=False)


def run_stress():
    """Run the behavioral and adversarial safety suite."""
    print("\n" + "=" * 80)
    print("      Running Behavioral & Adversarial CheckList Stress Suite")
    print("=" * 80 + "\n")
    script = os.path.join(REPO_ROOT, "scripts", "run_stress_test.py")
    ret = subprocess.call([sys.executable, script])
    if ret != 0:
        print("\n[ERROR] Stress suite failed!")
        sys.exit(ret)


def run_benchmark(
    generation_mode: str = "offline",
    judge_local: bool = False,
    include_zero_shot_local: bool = False,
    contrastive_pairs: int = 0,
    escalation_knn: bool = False,
    reviewed_labels: str = "data/training/human_consensus_labels.csv",
):
    """Run the uncontaminated 200 Golden Evaluation Set benchmark."""
    print("\n" + "=" * 80)
    print("      Running 200 Golden Set Benchmark Reproduction")
    print("=" * 80 + "\n")
    script = os.path.join(REPO_ROOT, "scripts", "reproduce_headline.py")
    command = [sys.executable, script, "--generation-mode", generation_mode]
    command.extend(["--reviewed-labels", reviewed_labels])
    if judge_local:
        command.append("--judge-local")
    if include_zero_shot_local:
        command.append("--include-zero-shot-local")
    if contrastive_pairs:
        command.extend(["--contrastive-pairs", str(contrastive_pairs)])
    if escalation_knn:
        command.append("--escalation-knn")
    ret = subprocess.call(command)
    if ret != 0:
        print("\n[ERROR] Benchmark failed!")
        sys.exit(ret)


def main():
    parser = argparse.ArgumentParser(
        description="@AmazonHelp AI Support Agent - Local Prototype Entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  serve      Start the local FastAPI server & live web UI (default)
  test       Run the discovered unit & integration suite
  benchmark  Run the 200 Golden Evaluation benchmark
  stress     Run the behavioral and adversarial safety suite

Examples:
  python main.py                       # Starts local server on http://127.0.0.1:8000
  python main.py serve --port 8080     # Starts server on custom port
  python main.py serve --generation-mode local  # Uses pinned local Qwen, no API key
  python main.py test                  # Runs the discovered unit/integration tests
  python main.py benchmark             # Runs golden evaluation benchmark
"""
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["serve", "benchmark", "stress", "test", "review"],
        help="Command to execute (default: serve)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind for server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind for server (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--generation-mode", choices=["offline", "local"], default=None)
    parser.add_argument("--judge-local", action="store_true", help="Run the pinned local-model reply judge")
    parser.add_argument("--include-zero-shot-local", action="store_true", help="Add the local Qwen zero-shot baseline")
    parser.add_argument(
        "--contrastive-pairs", type=int, default=0,
        help="Experimental local contrastive intent-pair ablation",
    )
    parser.add_argument(
        "--escalation-knn", action="store_true",
        help="Run the experimental dense-neighbor escalation ablation",
    )
    parser.add_argument(
        "--reviewed-labels",
        default="data/training/human_consensus_labels.csv",
        help="Adjudicated independent-review labels used for training",
    )

    args = parser.parse_args()

    if args.command == "serve":
        run_serve(
            host=args.host,
            port=args.port,
            reload=args.reload,
            generation_mode=args.generation_mode,
        )
    elif args.command == "test":
        run_test()
    elif args.command == "stress":
        run_stress()
    elif args.command == "review":
        run_review(args.host, args.port)
    elif args.command == "benchmark":
        run_benchmark(
            args.generation_mode or "offline",
            args.judge_local,
            args.include_zero_shot_local,
            args.contrastive_pairs,
            args.escalation_knn,
            args.reviewed_labels,
        )


if __name__ == "__main__":
    main()
