#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rescuecredit.azure_client import AzureOpenAIAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("azure", "deepseek"), required=True)
    args = parser.parse_args()
    adapter = AzureOpenAIAdapter(provider=args.provider)
    answer = adapter.complete(
        [
            {"role": "system", "content": "You are a connectivity checker."},
            {"role": "user", "content": "Reply only LLM_OK"},
        ],
        max_tokens=16,
        temperature=0.0,
    )
    print(f"LLM_OK provider={adapter.provider} model={adapter.deployment} answer={answer}")


if __name__ == "__main__":
    main()
