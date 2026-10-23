"""
Local entrypoint for the @AmazonHelp support-agent prototype.

Running:
    python main.py                     # Start the local server & UI (default)
    python main.py serve               # Start the local server & UI
"""

import sys
import os
import argparse


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


def main():
    parser = argparse.ArgumentParser(
        description="@AmazonHelp AI Support Agent - Local Prototype Entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  serve      Start the local FastAPI server & live web UI (default)

Examples:
  python main.py                       # Starts local server on http://127.0.0.1:8000
  python main.py serve --port 8080     # Starts server on custom port
  python main.py serve --generation-mode local  # Uses pinned local Qwen, no API key
"""
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["serve"],
        help="Command to execute (default: serve)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind for server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind for server (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--generation-mode", choices=["offline", "local"], default=None)

    args = parser.parse_args()

    if args.command == "serve":
        run_serve(
            host=args.host,
            port=args.port,
            reload=args.reload,
            generation_mode=args.generation_mode,
        )


if __name__ == "__main__":
    main()