#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("data/raw/DAMO-ConvAI"))
    parser.add_argument("--commit", default="483554eae102996f5ec1f4feab4e78ef29c2a394")
    args = parser.parse_args()
    if not args.destination.exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--sparse", "https://github.com/AlibabaResearch/DAMO-ConvAI.git", str(args.destination)],
            check=True,
        )
    subprocess.run(["git", "-C", str(args.destination), "sparse-checkout", "set", "api-bank"], check=True)
    subprocess.run(["git", "-C", str(args.destination), "fetch", "--depth", "1", "origin", args.commit], check=True)
    subprocess.run(["git", "-C", str(args.destination), "checkout", "--detach", args.commit], check=True)
    print(f"API-Bank ready at {args.destination / 'api-bank'} ({args.commit})")


if __name__ == "__main__":
    main()

