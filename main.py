"""
Universal Production Entrypoint for @AmazonHelp AI Customer Support System.

Running:
    python main.py                     # Start the Real-Time Production Server & UI (default)
    python main.py serve               # Start the Real-Time Production Server & UI
    python main.py test                # Run complete unit & integration test suite (27 tests)
    python main.py benchmark           # Run the 200 Golden Evaluation benchmark
    python main.py stress              # Run the 12-case CheckList adversarial stress suite
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


def run_serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start the production FastAPI REST API and live web dashboard."""
    import uvicorn
    print("\n" + "=" * 80)
    print("      @AmazonHelp AI Support Agent - Production Server")
    print("=" * 80)
    print(f"  • Interactive Web Dashboard: http://{host}:{port}/")
    print(f"  • OpenAPI Specification:     http://{host}:{port}/docs")
    print(f"  • Health & Readiness Probe:  http://{host}:{port}/api/v1/health")
    print(f"  • Prometheus Metrics:        http://{host}:{port}/metrics")
    print("=" * 80 + "\n")
    uvicorn.run("src.api:app", host=host, port=port, reload=reload)


def run_test():
    """Run all 27 unit and integration tests."""
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


def run_stress():
    """Run the 12-case behavioral & adversarial stress suite."""
    print("\n" + "=" * 80)
    print("      Running Behavioral & Adversarial CheckList Stress Suite")
    print("=" * 80 + "\n")
    script = os.path.join(REPO_ROOT, "scripts", "run_stress_test.py")
    ret = subprocess.call([sys.executable, script])
    if ret != 0:
        print("\n[ERROR] Stress suite failed!")
        sys.exit(ret)


def run_benchmark():
    """Run the uncontaminated 200 Golden Evaluation Set benchmark."""
    print("\n" + "=" * 80)
    print("      Running 200 Golden Set Benchmark Reproduction")
    print("=" * 80 + "\n")
    script = os.path.join(REPO_ROOT, "scripts", "reproduce_headline.py")
    ret = subprocess.call([sys.executable, script])
    if ret != 0:
        print("\n[ERROR] Benchmark failed!")
        sys.exit(ret)


def main():
    parser = argparse.ArgumentParser(
        description="@AmazonHelp AI Support Agent - Production Entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  serve      Start the production FastAPI server & live web UI (default)
  test       Run the full 27-test unit & integration suite
  benchmark  Run the 200 Golden Evaluation benchmark
  stress     Run the 12-case CheckList adversarial stress suite

Examples:
  python main.py                       # Starts production server on http://127.0.0.1:8000
  python main.py serve --port 8080     # Starts server on custom port
  python main.py test                  # Runs 27 tests
  python main.py benchmark             # Runs golden evaluation benchmark
"""
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["serve", "benchmark", "stress", "test"],
        help="Command to execute (default: serve)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind for server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind for server (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    if args.command == "serve":
        run_serve(host=args.host, port=args.port, reload=args.reload)
    elif args.command == "test":
        run_test()
    elif args.command == "stress":
        run_stress()
    elif args.command == "benchmark":
        run_benchmark()


if __name__ == "__main__":
    main()
