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
    """Chat-completion adapter with an Azure-compatible default.

    Existing AppWorld experiments keep using Azure unless a caller explicitly
    selects another provider.  ToolSandbox can therefore use DeepSeek without
    silently changing any frozen AppWorld protocol.
    """

    def __init__(self, provider: str | None = None) -> None:
        load_dotenv_if_present()
        self.provider = (provider or os.getenv("LLM_PROVIDER", "azure")).strip().lower()
        self._extra_body: dict[str, Any] | None = None
        if self.provider == "azure":
            from openai import AzureOpenAI

            endpoint = os.getenv("ENDPOINT_URL", "https://scdall3.openai.azure.com/")
            key = os.getenv("AZURE_OPENAI_API_KEY")
            if not key or key == "REPLACE_WITH_ROTATED_KEY":
                raise RuntimeError("Set AZURE_OPENAI_API_KEY in .env before using Azure")
            self.deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=key,
                api_version=os.getenv(
                    "AZURE_OPENAI_API_VERSION", "2025-01-01-preview"
                ),
            )
        elif self.provider == "deepseek":
            from openai import OpenAI

            key = os.getenv("DEEPSEEK_API_KEY")
            if not key or key == "REPLACE_WITH_ROTATED_KEY":
                raise RuntimeError("Set DEEPSEEK_API_KEY in .env before using DeepSeek")
            self.deployment = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
            self.client = OpenAI(
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                api_key=key,
            )
            thinking = os.getenv("DEEPSEEK_THINKING", "disabled").strip().lower()
            if thinking not in {"enabled", "disabled"}:
                raise RuntimeError("DEEPSEEK_THINKING must be enabled or disabled")
            self._extra_body = {"thinking": {"type": thinking}}
        else:
            raise RuntimeError(f"unsupported LLM provider: {self.provider}")

    def complete(self, messages: list[dict[str, Any]], max_tokens: int = 1024, temperature: float = 0.0) -> str:
        request: dict[str, Any] = dict(
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
        if self._extra_body is not None:
            request["extra_body"] = self._extra_body
        completion = self.client.chat.completions.create(**request)
        return completion.choices[0].message.content or ""
