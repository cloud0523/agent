from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from rag_agent.generation.llm import LLMGenerator


class DummyAnthropicClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.calls = []

    class messages:
        @staticmethod
        def create(**kwargs):
            return SimpleNamespace(content=[SimpleNamespace(text="claude answer")])


class DummyOpenAIClient:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.calls = []

    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="openai answer"))]
                )


class DummyOpenAIStreamClient:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url

    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                return iter([
                    SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))]),
                    SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=" world"))]),
                ])


class DummyOpenAIWithMessagesClient:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.calls = []

    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="openai message answer"))]
                )


class DummyOllamaClient:
    def __init__(self, base_url=None):
        self.base_url = base_url

    @staticmethod
    def chat(**kwargs):
        return {"message": {"content": "ollama answer"}}


class DummyOllamaStreamClient:
    def __init__(self, base_url=None):
        self.base_url = base_url

    @staticmethod
    def chat(**kwargs):
        return iter([
            {"message": {"content": "Hello"}},
            {"message": {"content": " world"}},
        ])


def test_generate_claude(monkeypatch):
    module = types.ModuleType("anthropic")
    module.Anthropic = DummyAnthropicClient
    monkeypatch.setitem(__import__("sys").modules, "anthropic", module)

    generator = LLMGenerator(provider="claude", model="claude-3", api_key="abc")
    result = generator.generate("hello", system_prompt="You are helpful")

    assert result == "claude answer"


def test_generate_stream_openai(monkeypatch):
    module = types.ModuleType("openai")
    module.OpenAI = DummyOpenAIStreamClient
    monkeypatch.setitem(__import__("sys").modules, "openai", module)

    generator = LLMGenerator(provider="openai", model="gpt-4o")
    result = list(generator.generate_stream("hello", system_prompt="You are helpful"))

    assert result == ["Hello", " world"]


def test_generate_with_messages_openai(monkeypatch):
    module = types.ModuleType("openai")
    module.OpenAI = DummyOpenAIWithMessagesClient
    monkeypatch.setitem(sys.modules, "openai", module)

    generator = LLMGenerator(provider="openai", model="gpt-4o")
    result = generator.generate_with_messages(
        [{"role": "user", "content": "hello"}],
        system_prompt="You are helpful",
    )

    assert result == "openai message answer"


def test_generate_ollama(monkeypatch):
    module = types.ModuleType("ollama")
    module.chat = DummyOllamaClient.chat
    monkeypatch.setitem(__import__("sys").modules, "ollama", module)

    generator = LLMGenerator(provider="ollama", model="llama3", base_url="http://localhost:11434")
    result = generator.generate("hello", system_prompt="You are helpful")

    assert result == "ollama answer"
