"""Build a processed pair file containing only causally prior conversation turns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline import add_prior_conversation_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/twcs.csv")
    parser.add_argument("--pairs", default="data/processed/amazonhelp_english_pairs.csv")
    parser.add_argument("--output", default="data/processed/amazonhelp_context_pairs.csv")
    parser.add_argument("--max-prior-turns", type=int, default=4)
    args = parser.parse_args()

    raw = ROOT / args.raw
    pairs_path = ROOT / args.pairs
    output = ROOT / args.output
    if not raw.exists():
        raise FileNotFoundError(f"Raw conversation file is required: {raw}")
    pairs = pd.read_csv(pairs_path)
    enriched = add_prior_conversation_context(
        pairs, str(raw), max_prior_turns=args.max_prior_turns
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output, index=False)
    coverage = float((enriched["prior_turn_count"] > 0).mean())
    print(
        f"Wrote {len(enriched)} rows to {output}; "
        f"prior-context coverage={coverage:.3f}, max_prior_turns={args.max_prior_turns}"
    )


if __name__ == "__main__":
    main()
