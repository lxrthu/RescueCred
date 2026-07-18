from pathlib import Path

import pytest

from scripts.diagnose_azure_key import (
    alternate_endpoint,
    classify_http_status,
    key_fingerprint,
    load_config,
)


def test_load_config_reads_file_instead_of_stale_environment(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ENDPOINT_URL=https://example.openai.azure.com/\n"
        "AZURE_OPENAI_API_KEY=file-key-value\n"
        "DEPLOYMENT_NAME=gpt-4o-test\n"
        "AZURE_OPENAI_API_VERSION=2025-01-01-preview\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "stale-process-key")

    config = load_config(env_file, use_process_environment=False)

    assert config.key == "file-key-value"
    assert config.endpoint == "https://example.openai.azure.com/"
    assert config.deployment == "gpt-4o-test"


def test_load_config_rejects_missing_key(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("ENDPOINT_URL=https://example.openai.azure.com/\n", encoding="utf-8")

    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
        load_config(env_file, use_process_environment=False)


@pytest.mark.parametrize(
    ("status", "accepted", "diagnosis"),
    [
        (200, True, "KEY_ACCEPTED_AND_COMPLETION_SUCCEEDED"),
        (401, False, "KEY_REJECTED_OR_ENDPOINT_MISMATCH"),
        (404, False, "AUTH_LIKELY_ACCEPTED_BUT_DEPLOYMENT_OR_ROUTE_NOT_FOUND"),
        (429, True, "KEY_ACCEPTED_BUT_RATE_LIMITED_OR_QUOTA_EXHAUSTED"),
    ],
)
def test_classify_http_status(status: int, accepted: bool, diagnosis: str):
    assert classify_http_status(status) == (accepted, diagnosis)


def test_fingerprint_does_not_contain_key():
    key = "super-secret-key"
    fingerprint = key_fingerprint(key)
    assert len(fingerprint) == 12
    assert key not in fingerprint


def test_alternate_endpoint():
    assert alternate_endpoint("https://x.openai.azure.com/") == (
        "https://x.cognitiveservices.azure.com/"
    )
    assert alternate_endpoint("https://example.com/") is None
