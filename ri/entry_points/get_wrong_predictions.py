#!/usr/bin/env python3
import argparse
import json
import os
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Optional


def normalize_numeric(value: Any) -> Optional[Decimal]:
    """
    Strip punctuation and convert to Decimal.
    Returns None if the value is missing or not a number.
    """
    if value is None:
        return None

    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value))

    s = str(value).strip().replace(",", "").replace("$", "")
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def main() -> None:
    """Extract wrong predictions from a results JSON file."""
    parser = argparse.ArgumentParser(
        description="Extract wrong predictions from a GSM-8K JSON results file"
    )
    parser.add_argument("json_file", help="Path to the predictions JSON file")
    args = parser.parse_args()

    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    wrong = []
    for rec in data:
        gold = normalize_numeric(rec.get("Answer_num"))
        pred = normalize_numeric(rec.get("Generated Answer_num"))

        if gold is None or pred is None or gold != pred:
            wrong.append(rec)

    # Build output filename
    base = os.path.basename(args.json_file)
    m = re.search(r"seed(\d+)", base)
    seed = m.group(1) if m else ""
    out_name = f"wrong_predictions_seed{seed}.json" if seed else "wrong_predictions.json"

    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(wrong, f, indent=4, ensure_ascii=False)

    print(f"{len(wrong)} wrong predictions written to {out_name}")


if __name__ == "__main__":
    main()
