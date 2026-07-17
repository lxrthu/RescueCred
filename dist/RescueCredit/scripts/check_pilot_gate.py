#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--rescue", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/pilot/gate.json"))
    args = parser.parse_args()
    mask = json.loads(args.mask.read_text(encoding="utf-8"))
    rescue = json.loads(args.rescue.read_text(encoding="utf-8"))
    improvements = {
        "s_off": rescue["s_off"] - mask["s_off"],
        "first_pass": rescue["first_pass"] - mask["first_pass"],
    }
    passed = any(value > 0 for value in improvements.values())
    result = {"passed": passed, "rule": "expand only if RescueCredit improves S_off or First-pass in this pilot", "improvements": improvements}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

