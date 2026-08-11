from __future__ import annotations

from collections.abc import Iterator
from typing import Any

RAG_SYSTEM_PROMPT = """你是一个文档问答助手。请根据提供的文档内容回答问题。

规则：
- 只使用文档中提供的资料来回答
- 如果文档中没有相关信息，请如实告知用户
- 回答时引用具体来源（标明出自哪份文档）
- 回答要简洁、准确、有条理"""

RAG_USER_PROMPT = """参考文档：

{context}

问题：{question}

回答："""


class LLMGenerator:
    def __init__(self, provider: str, model: str, api_key: str | None = None, base_url: str | None = None):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """非流式，返回完整回答。"""
        if self.provider == "claude":
            return self._generate_claude(prompt, system_prompt)
        elif self.provider == "openai":
            return self._generate_openai(prompt, system_prompt)
        elif self.provider == "ollama":
            return self._generate_ollama(prompt, system_prompt)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_with_messages(self, messages: list[dict], system_prompt: str | None = None) -> str:
        """非流式，按消息列表生成完整回答。"""
        if self.provider == "claude":
            return self._generate_with_messages_claude(messages, system_prompt)
        elif self.provider == "openai":
            return self._generate_with_messages_openai(messages, system_prompt)
        elif self.provider == "ollama":
            return self._generate_with_messages_ollama(messages, system_prompt)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_stream(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
        """流式，逐个 token yield。"""
        if self.provider == "claude":
            yield from self._generate_stream_claude(prompt, system_prompt)
        elif self.provider == "openai":
            yield from self._generate_stream_openai(prompt, system_prompt)
        elif self.provider == "ollama":
            yield from self._generate_stream_ollama(prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_stream_with_messages(self, messages: list[dict], system_prompt: str | None = None) -> Iterator[str]:
        """流式，按消息列表逐个 token yield。"""
        if self.provider == "claude":
            yield from self._generate_stream_with_messages_claude(messages, system_prompt)
        elif self.provider == "openai":
            yield from self._generate_stream_with_messages_openai(messages, system_prompt)
        elif self.provider == "ollama":
            yield from self._generate_stream_with_messages_ollama(messages, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_claude(self, prompt: str, system_prompt: str | None = None) -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt or RAG_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _generate_openai(self, prompt: str, system_prompt: str | None = None) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt or RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    
    def _generate_ollama(self, prompt: str, system_prompt: str | None = None) -> str:
        import ollama

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt or RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            host=self.base_url,
            stream=False,
        )
        return response["message"]["content"]

    def _generate_with_messages_claude(self, messages: list[dict], system_prompt: str | None = None) -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system_prompt or RAG_SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text

    def _generate_with_messages_openai(self, messages: list[dict], system_prompt: str | None = None) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt or RAG_SYSTEM_PROMPT},
                *messages,
            ],
        )
        return response.choices[0].message.content

    def _generate_with_messages_ollama(self, messages: list[dict], system_prompt: str | None = None) -> str:
        import ollama

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt or RAG_SYSTEM_PROMPT},
                *messages,
            ],
            host=self.base_url,
            stream=False,
        )
        return response["message"]["content"]

    def _generate_stream_claude(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)
        with client.messages.stream(
            model=self.model,
            max_tokens=2048,
            system=system_prompt or RAG_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for chunk in stream:
                text = getattr(getattr(chunk, "delta", None), "text", None)
                if text:
                    yield text

    def _generate_stream_openai(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        stream = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt or RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def _generate_stream_with_messages_claude(self, messages: list[dict], system_prompt: str | None = None) -> Iterator[str]:
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)
        with client.messages.stream(
            model=self.model,
            max_tokens=2048,
            system=system_prompt or RAG_SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            for chunk in stream:
                text = getattr(getattr(chunk, "delta", None), "text", None)
                if text:
                    yield text

    def _generate_stream_with_messages_openai(self, messages: list[dict], system_prompt: str | None = None) -> Iterator[str]:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        stream = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt or RAG_SYSTEM_PROMPT},
                *messages,
            ],
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def _generate_stream_with_messages_ollama(self, messages: list[dict], system_prompt: str | None = None) -> Iterator[str]:
        import ollama

        stream = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt or RAG_SYSTEM_PROMPT},
                *messages,
            ],
            host=self.base_url,
            stream=True,
        )
        for chunk in stream:
            content = chunk["message"]["content"]
            if content:
                yield content

    def _generate_stream_ollama(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
        import ollama

        stream = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt or RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            host=self.base_url,
            stream=True,
        )
        for chunk in stream:
            content = chunk["message"]["content"]
            if content:
                yield content
