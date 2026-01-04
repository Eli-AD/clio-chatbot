"""Ollama API client for local LLM inference."""

import httpx
from typing import AsyncIterator, Optional


class OllamaClient:
    """Async client for Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)

    async def generate(
        self,
        prompt: str,
        model: str = "qwen2.5:7b",
        system: str = None,
        stream: bool = True
    ) -> AsyncIterator[str]:
        """Generate a response from Ollama, streaming tokens."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }

        if system:
            payload["system"] = system

        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done"):
                        break

    async def generate_full(
        self,
        prompt: str,
        model: str = "qwen2.5:7b",
        system: str = None
    ) -> str:
        """Generate a complete response (non-streaming)."""
        chunks = []
        async for chunk in self.generate(prompt, model, system, stream=True):
            chunks.append(chunk)
        return "".join(chunks)

    async def chat(
        self,
        messages: list,
        model: str = "qwen2.5:7b",
        stream: bool = True
    ) -> AsyncIterator[str]:
        """Chat completion with message history."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
                    if data.get("done"):
                        break

    async def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False

    async def list_models(self) -> list:
        """List available models."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except:
            pass
        return []

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
