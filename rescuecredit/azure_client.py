from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_dotenv_if_present(path: str | Path = ".env") -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(path, override=False)


class AzureOpenAIAdapter:
    """Exact AzureOpenAI SDK shape requested by the project owner."""

    def __init__(self) -> None:
        load_dotenv_if_present()
        from openai import AzureOpenAI

        endpoint = os.getenv("ENDPOINT_URL", "https://scdall3.openai.azure.com/")
        key = os.getenv("AZURE_OPENAI_API_KEY")
        if not key or key == "REPLACE_WITH_ROTATED_KEY":
            raise RuntimeError("Set AZURE_OPENAI_API_KEY in .env before using Azure")
        self.deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        )

    def complete(self, messages: list[dict[str, Any]], max_tokens: int = 1024, temperature: float = 0.0) -> str:
        completion = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
        )
        return completion.choices[0].message.content or ""

