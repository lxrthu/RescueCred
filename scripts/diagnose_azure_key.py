#!/usr/bin/env python3
"""Diagnose an Azure OpenAI key without printing the secret.

The default configuration source is the requested .env file, not the current
process environment.  This is intentional: an interactive shell may retain an
old exported key after .env has been edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


@dataclass(frozen=True)
class AzureConfig:
    endpoint: str
    key: str
    deployment: str
    api_version: str
    source: str


def _clean(value: str | None) -> str:
    return (value or "").strip().strip('"').strip("'")


def load_config(env_file: Path, use_process_environment: bool) -> AzureConfig:
    values: dict[str, Any]
    if use_process_environment:
        values = dict(os.environ)
        source = "process_environment"
    else:
        if not env_file.is_file():
            raise ValueError(f"env file does not exist: {env_file}")
        values = dict(dotenv_values(env_file))
        source = str(env_file.resolve())

    endpoint = _clean(values.get("ENDPOINT_URL"))
    key = _clean(values.get("AZURE_OPENAI_API_KEY"))
    deployment = _clean(values.get("DEPLOYMENT_NAME")) or "gpt-4o"
    api_version = _clean(values.get("AZURE_OPENAI_API_VERSION")) or "2025-01-01-preview"
    if not endpoint:
        raise ValueError(f"ENDPOINT_URL is missing from {source}")
    if not key or key == "REPLACE_WITH_ROTATED_KEY":
        raise ValueError(f"AZURE_OPENAI_API_KEY is missing from {source}")
    return AzureConfig(endpoint.rstrip("/") + "/", key, deployment, api_version, source)


def key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def classify_http_status(status: int) -> tuple[bool, str]:
    if 200 <= status < 300:
        return True, "KEY_ACCEPTED_AND_COMPLETION_SUCCEEDED"
    if status == 401:
        return False, "KEY_REJECTED_OR_ENDPOINT_MISMATCH"
    if status == 403:
        return False, "KEY_ACCEPTED_BUT_ACCESS_FORBIDDEN_OR_NETWORK_POLICY_BLOCKED"
    if status == 404:
        return False, "AUTH_LIKELY_ACCEPTED_BUT_DEPLOYMENT_OR_ROUTE_NOT_FOUND"
    if status == 429:
        return True, "KEY_ACCEPTED_BUT_RATE_LIMITED_OR_QUOTA_EXHAUSTED"
    if status == 400:
        return False, "AUTH_LIKELY_ACCEPTED_BUT_REQUEST_OR_API_VERSION_INVALID"
    return False, f"UNEXPECTED_HTTP_STATUS_{status}"


def _safe_error_body(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")[:1000]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    return json.dumps(payload, ensure_ascii=False)


def raw_rest_probe(config: AzureConfig, timeout: float) -> dict[str, Any]:
    deployment = urllib.parse.quote(config.deployment, safe="")
    version = urllib.parse.quote(config.api_version, safe="")
    url = (
        f"{config.endpoint}openai/deployments/{deployment}/chat/completions"
        f"?api-version={version}"
    )
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": "Reply only AZURE_OK"}],
            "max_tokens": 16,
            "temperature": 0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"api-key": config.key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            accepted, diagnosis = classify_http_status(response.status)
            payload = json.loads(raw.decode("utf-8"))
            answer = payload["choices"][0]["message"].get("content", "")
            return {
                "transport": "raw_rest",
                "http_status": response.status,
                "accepted": accepted,
                "diagnosis": diagnosis,
                "answer": answer,
            }
    except urllib.error.HTTPError as exc:
        accepted, diagnosis = classify_http_status(exc.code)
        return {
            "transport": "raw_rest",
            "http_status": exc.code,
            "accepted": accepted,
            "diagnosis": diagnosis,
            "error": _safe_error_body(exc.read()),
        }
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        return {
            "transport": "raw_rest",
            "http_status": None,
            "accepted": False,
            "diagnosis": "NETWORK_OR_TLS_FAILURE",
            "error": str(exc),
        }


def sdk_probe(config: AzureConfig, timeout: float) -> dict[str, Any]:
    try:
        from openai import APIConnectionError, APIStatusError, AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=config.endpoint,
            api_key=config.key,
            api_version=config.api_version,
            timeout=timeout,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=config.deployment,
            messages=[{"role": "user", "content": "Reply only AZURE_OK"}],
            max_tokens=16,
            temperature=0,
        )
        return {
            "transport": "openai_sdk",
            "http_status": 200,
            "accepted": True,
            "diagnosis": "KEY_ACCEPTED_AND_COMPLETION_SUCCEEDED",
            "answer": response.choices[0].message.content or "",
        }
    except APIStatusError as exc:
        accepted, diagnosis = classify_http_status(exc.status_code)
        return {
            "transport": "openai_sdk",
            "http_status": exc.status_code,
            "accepted": accepted,
            "diagnosis": diagnosis,
            "error": str(exc)[:1000],
        }
    except APIConnectionError as exc:
        return {
            "transport": "openai_sdk",
            "http_status": None,
            "accepted": False,
            "diagnosis": "NETWORK_OR_TLS_FAILURE",
            "error": str(exc)[:1000],
        }
    except Exception as exc:  # Keep this diagnostic useful across SDK versions.
        return {
            "transport": "openai_sdk",
            "http_status": None,
            "accepted": False,
            "diagnosis": "LOCAL_SDK_OR_CONFIGURATION_FAILURE",
            "error": f"{type(exc).__name__}: {exc}"[:1000],
        }


def alternate_endpoint(endpoint: str) -> str | None:
    marker = ".openai.azure.com"
    if marker not in endpoint:
        return None
    return endpoint.replace(marker, ".cognitiveservices.azure.com")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--use-process-environment",
        action="store_true",
        help="Use exported variables instead of the env file.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--probe-alternate-host",
        action="store_true",
        help="Also try the cognitiveservices.azure.com form of an OpenAI endpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.env_file, args.use_process_environment)
    except ValueError as exc:
        print(json.dumps({"status": "CONFIG_ERROR", "error": str(exc)}, indent=2))
        return 4

    endpoints = [config.endpoint]
    if args.probe_alternate_host:
        alternate = alternate_endpoint(config.endpoint)
        if alternate and alternate not in endpoints:
            endpoints.append(alternate)

    print(
        json.dumps(
            {
                "configuration_source": config.source,
                "endpoint": config.endpoint,
                "deployment": config.deployment,
                "api_version": config.api_version,
                "key_length": len(config.key),
                "key_sha256_prefix": key_fingerprint(config.key),
                "key_is_never_printed": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    results: list[dict[str, Any]] = []
    for endpoint in endpoints:
        endpoint_config = AzureConfig(
            endpoint, config.key, config.deployment, config.api_version, config.source
        )
        result = {
            "endpoint": endpoint,
            "raw_rest": raw_rest_probe(endpoint_config, args.timeout),
            "openai_sdk": sdk_probe(endpoint_config, args.timeout),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    accepted = any(
        probe["accepted"]
        for result in results
        for probe in (result["raw_rest"], result["openai_sdk"])
    )
    print("AZURE_KEY_USABLE" if accepted else "AZURE_KEY_NOT_USABLE_WITH_TESTED_ENDPOINTS")
    return 0 if accepted else 3


if __name__ == "__main__":
    sys.exit(main())
