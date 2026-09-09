"""
Standalone Server Launcher for @AmazonHelp Support AI Dashboard & REST API.
Usage:
    python scripts/serve_dashboard.py [--port 8000] [--host 0.0.0.0]
"""

import argparse
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uvicorn
from src.api import app

def main():
    parser = argparse.ArgumentParser(description="Serve @AmazonHelp Support AI REST API & Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    args = parser.parse_args()

    print("=" * 80)
    print("      @AmazonHelp Support AI Agent: Enterprise REST API & Live UI")
    print("=" * 80)
    print(f"  • Interactive Web Dashboard: http://{args.host}:{args.port}/")
    print(f"  • Interactive OpenAPI Docs:  http://{args.host}:{args.port}/docs")
    print(f"  • Health & Readiness Probe:  http://{args.host}:{args.port}/api/v1/health")
    print(f"  • Prometheus Metrics:        http://{args.host}:{args.port}/metrics")
    print("=" * 80)

    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
