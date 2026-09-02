from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import BenchmarkError, load_json, prediction_template, score_predictions, validate_predictions


def _write(value: object, output: str | None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if output:
        Path(output).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-only deterministic benchmark evaluator")
    parser.add_argument("command", nargs="?", choices=("score", "validate", "template"), default="score")
    parser.add_argument("--packets", default="visible_packets/packets.json")
    parser.add_argument("--predictions")
    parser.add_argument("--answers", default="hidden_answers/answers.json")
    parser.add_argument("--output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        packets = load_json(args.packets)
        if args.command == "template":
            _write(prediction_template(packets), args.output)
            return 0
        if not args.predictions:
            raise BenchmarkError("--predictions is required for validate/score")
        predictions = load_json(args.predictions)
        if args.command == "validate":
            valid = validate_predictions(packets, predictions)
            _write({"status": "PASS", "validated_predictions": len(valid)}, args.output)
            return 0
        answers = load_json(args.answers)
        _write(score_predictions(packets, predictions, answers), args.output)
        return 0
    except (BenchmarkError, OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
