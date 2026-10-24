"""Human labelling workflow: sample, adjudicate, gate.

The intent ceiling is supervision, not model capacity: the targets are extracted
from historical agent replies by rule, so they encode what the agent did while
the model only ever sees what the customer wrote. Closing that gap needs human
labels, and this is the path that produces them.

    # 1. Draw a stratified sample and emit one sheet per annotator
    python scripts_label_workflow.py sample --annotators alice,bob --per-class 60

    # 2. Annotators fill in reviewed_intent / reviewed_escalate (blank = abstain)

    # 3. Combine, measure agreement, and write the file the pipeline reads
    python scripts_label_workflow.py adjudicate \\
        data/annotations/alice.csv data/annotations/bob.csv

Agreement is reported as Cohen's kappa per dimension and gated: a dimension that
fails the floor is left unlabelled rather than written into training, because
low-agreement labels are worse than no labels — they look like ground truth while
encoding disagreement.

`reviewed_intent` takes one of the six intents, `reviewed_escalate` takes
true/false, and either may be left blank to abstain. Every row carries the
`source_sha256` binding it to the exact historical pair that was shown, so a label
can never silently attach to a different conversation after the corpus changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.classifier import INTENT_TAXONOMY
from src.supervision import infer_reply_intent, training_review_source_hash

PAIRS_CSV = "data/processed/amazonhelp_context_pairs.csv"
ADJUDICATED_CSV = "data/training/labels_human_adjudicated.csv"
ANNOTATION_DIR = "data/annotations"

# Below this, annotators are not applying a shared definition and the labels
# should not be trusted as targets. 0.61 is the conventional lower bound for
# "substantial" agreement on the Landis & Koch scale.
KAPPA_FLOOR = 0.61


def cohen_kappa(first: Sequence[object], second: Sequence[object]) -> Tuple[float, float]:
    """Return (kappa, observed agreement) for two raters over paired labels.

    Kappa is used rather than raw agreement because raw agreement is inflated
    whenever one label dominates: with six intents where one covers ~69% of
    traffic, two annotators who both always answer ORDER_TRACKING_AND_DELAY would
    agree 69% of the time while carrying no information at all.
    """
    if len(first) != len(second):
        raise ValueError(f"Raters must label the same rows ({len(first)} vs {len(second)})")
    if not first:
        raise ValueError("At least one paired label is required")
    labels = sorted({str(value) for value in list(first) + list(second)})
    index = {label: position for position, label in enumerate(labels)}
    size = len(labels)
    confusion = [[0] * size for _ in range(size)]
    for left, right in zip(first, second):
        confusion[index[str(left)]][index[str(right)]] += 1
    total = len(first)
    observed = sum(confusion[i][i] for i in range(size)) / total
    row_totals = [sum(row) / total for row in confusion]
    column_totals = [sum(confusion[i][j] for i in range(size)) / total for j in range(size)]
    expected = sum(row_totals[i] * column_totals[i] for i in range(size))
    if expected >= 1.0:
        # Perfectly degenerate marginals: agreement is guaranteed, so kappa is
        # undefined. Report 1.0 when they in fact agreed, else 0.0.
        return (1.0 if observed >= 1.0 else 0.0), observed
    return (observed - expected) / (1.0 - expected), observed


def sample_for_annotation(
    per_class: int,
    seed: int,
    annotators: Sequence[str],
    output_dir: str = ANNOTATION_DIR,
) -> List[Path]:
    """Draw a stratified sample and emit one blank sheet per annotator."""
    pairs = pd.read_csv(PAIRS_CSV)
    pairs["reply_intent"] = pairs["reply_text"].fillna("").map(infer_reply_intent)
    labelled = pairs[pairs["reply_intent"].notna()].copy()
    parts = [
        group.sample(n=min(per_class, len(group)), random_state=seed)
        for _, group in labelled.groupby("reply_intent", sort=True)
    ]
    sample = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    sample["source_sha256"] = sample.apply(training_review_source_hash, axis=1)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for annotator in annotators:
        path = destination / f"{annotator}.csv"
        frame = pd.DataFrame({
            "inbound_id": sample["inbound_id"].astype(str),
            "source_sha256": sample["source_sha256"],
            # Read-only context for the annotator.
            "conversation_context": sample["conversation_context"].fillna(""),
            "customer_text": sample["customer_text"].fillna(""),
            "reply_text": sample["reply_text"].fillna(""),
            # To be filled in by hand. Blank means abstain.
            "reviewed_intent": "",
            "reviewed_escalate": "",
        })
        frame.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        written.append(path)
    return written


def _load_sheets(paths: Sequence[str]) -> Dict[str, pd.DataFrame]:
    sheets: Dict[str, pd.DataFrame] = {}
    for path in paths:
        frame = pd.read_csv(path, dtype={"inbound_id": str})
        missing = {"inbound_id", "source_sha256", "reviewed_intent", "reviewed_escalate"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame["inbound_id"] = frame["inbound_id"].astype(str)
        if frame["inbound_id"].duplicated().any():
            raise ValueError(f"{path} contains duplicate inbound_id values")
        sheets[Path(path).stem] = frame.set_index("inbound_id")
    return sheets


def _agreement_on_shared_rows(
    sheets: Dict[str, pd.DataFrame], column: str
) -> Tuple[List[str], List[object], List[object]]:
    """Paired labels for the first two annotators on rows both actually labelled."""
    names = list(sheets)
    if len(names) < 2:
        return [], [], []
    ids = [  # rows both annotators labelled on this dimension
        inbound_id for inbound_id in sheets[names[0]].index
        if inbound_id in sheets[names[1]].index
        and pd.notna(sheets[names[0]].at[inbound_id, column])
        and pd.notna(sheets[names[1]].at[inbound_id, column])
        and str(sheets[names[0]].at[inbound_id, column]).strip() != ""
        and str(sheets[names[1]].at[inbound_id, column]).strip() != ""
    ]
    left = [str(sheets[names[0]].at[i, column]).strip() for i in ids]
    right = [str(sheets[names[1]].at[i, column]).strip() for i in ids]
    return ids, left, right


def _majority(values: List[str]) -> Optional[str]:
    """Return the strict majority label, or None when there is no majority."""
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    best, wins = max(counts.items(), key=lambda item: item[1])
    return best if wins * 2 > len(values) else None


def adjudicate(
    paths: Sequence[str],
    output: str = ADJUDICATED_CSV,
    kappa_floor: float = KAPPA_FLOOR,
) -> Dict[str, object]:
    """Combine annotator sheets into the file the pipeline consumes."""
    sheets = _load_sheets(paths)
    if len(sheets) < 2:
        raise ValueError("Adjudication needs at least two annotator sheets")

    report: Dict[str, object] = {"annotators": list(sheets), "dimensions": {}}
    accepted: Dict[str, Dict[str, str]] = {}

    for column, transform, valid in (
        ("reviewed_intent", str, set(INTENT_TAXONOMY)),
        ("reviewed_escalate", lambda value: str(value).strip().lower(), {"true", "false"}),
    ):
        _ids, left, right = _agreement_on_shared_rows(sheets, column)
        if not _ids:
            report["dimensions"][column] = {"status": "no_overlap"}
            continue
        kappa, observed = cohen_kappa(left, right)
        entry = {
            "status": "accepted" if kappa >= kappa_floor else "rejected",
            "paired_rows": len(_ids),
            "cohen_kappa": round(kappa, 4),
            "observed_agreement": round(observed, 4),
            "kappa_floor": kappa_floor,
        }
        report["dimensions"][column] = entry
        if kappa < kappa_floor:
            # Deliberately no output for this dimension: writing low-agreement
            # labels into training would disguise disagreement as ground truth.
            continue

        agreed: Dict[str, str] = {}
        all_ids = sorted({i for sheet in sheets.values() for i in sheet.index})
        for inbound_id in all_ids:
            votes = [
                str(sheet.at[inbound_id, column]).strip()
                for sheet in sheets.values()
                if inbound_id in sheet.index and pd.notna(sheet.at[inbound_id, column])
                and str(sheet.at[inbound_id, column]).strip() != ""
            ]
            if not votes:
                continue
            winner = _majority(votes)
            if winner is None:
                continue
            # Normalise before validating. pandas reads a column of "true"/"false"
            # strings back as boolean dtype, so the raw vote for an escalation can
            # be the Python bool True rather than "true"; checking the raw value
            # against {"true","false"} silently rejected every row.
            normalised = transform(winner)
            if normalised not in valid:
                continue
            agreed[inbound_id] = normalised
        accepted[column] = agreed
        entry["labelled_rows"] = len(agreed)

    intent_labels = accepted.get("reviewed_intent", {})
    escalate_labels = accepted.get("reviewed_escalate", {})
    if not intent_labels and not escalate_labels:
        report["written"] = None
        report["note"] = (
            "No dimension met the agreement floor, so nothing was written. "
            "Re-brief annotators on the taxonomy and re-run."
        )
        return report

    hashes = {
        inbound_id: str(sheet.at[inbound_id, "source_sha256"])
        for sheet in sheets.values() for inbound_id in sheet.index
    }
    rows = [
        {
            "inbound_id": inbound_id,
            "source_sha256": hashes[inbound_id],
            "reviewed_intent": intent_labels.get(inbound_id, ""),
            "reviewed_escalate": escalate_labels.get(inbound_id, ""),
        }
        for inbound_id in sorted(set(intent_labels) | set(escalate_labels))
    ]
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False, quoting=csv.QUOTE_NONNUMERIC)
    report["written"] = str(destination)
    report["rows"] = len(rows)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sample_parser = sub.add_parser("sample", help="Draw a stratified sample for annotation")
    sample_parser.add_argument("--annotators", default="alice,bob",
                               help="Comma-separated annotator names, one sheet each")
    sample_parser.add_argument("--per-class", type=int, default=60)
    sample_parser.add_argument("--seed", type=int, default=42)
    sample_parser.add_argument("--output-dir", default=ANNOTATION_DIR)

    adjudicate_parser = sub.add_parser("adjudicate", help="Combine sheets and gate on agreement")
    adjudicate_parser.add_argument("sheets", nargs="+")
    adjudicate_parser.add_argument("--output", default=ADJUDICATED_CSV)
    adjudicate_parser.add_argument("--kappa-floor", type=float, default=KAPPA_FLOOR)
    adjudicate_parser.add_argument("--report", default=None,
                                   help="Optional path to write the JSON agreement report")

    args = parser.parse_args(argv)
    if args.command == "sample":
        names = [name.strip() for name in args.annotators.split(",") if name.strip()]
        if len(names) < 2:
            print("At least two annotators are required for agreement to be measurable",
                  file=sys.stderr)
            return 2
        for path in sample_for_annotation(args.per_class, args.seed, names, args.output_dir):
            print(f"wrote {path}")
        return 0

    report = adjudicate(args.sheets, output=args.output, kappa_floor=args.kappa_floor)
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    written = report.get("written")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
