from types import SimpleNamespace

import openai
import pytest

from rescuecredit.azure_client import AzureOpenAIAdapter


class _FakeCompletions:
    def __init__(self):
        self.request = None

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="LLM_OK"))]
        )


class _FakeClient:
    def __init__(self, **config):
        self.config = config
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_deepseek_provider_uses_openai_base_url_and_nonthinking(monkeypatch):
    created = []

    def fake_openai(**config):
        client = _FakeClient(**config)
        created.append(client)
        return client

    monkeypatch.setattr(openai, "OpenAI", fake_openai)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_THINKING", "disabled")

    adapter = AzureOpenAIAdapter(provider="deepseek")
    result = adapter.complete([{"role": "user", "content": "test"}])

    assert result == "LLM_OK"
    assert created[0].config["base_url"] == "https://api.deepseek.com"
    assert created[0].chat.completions.request["model"] == "deepseek-v4-pro"
    assert created[0].chat.completions.request["extra_body"] == {
        "thinking": {"type": "disabled"}
    }


def test_deepseek_provider_requires_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        AzureOpenAIAdapter(provider="deepseek")


def test_deepseek_thinking_mode_is_validated(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_THINKING", "sometimes")
    with pytest.raises(RuntimeError, match="DEEPSEEK_THINKING"):
        AzureOpenAIAdapter(provider="deepseek")


def test_unknown_provider_is_rejected():
    with pytest.raises(RuntimeError, match="unsupported LLM provider"):
        AzureOpenAIAdapter(provider="unknown")
