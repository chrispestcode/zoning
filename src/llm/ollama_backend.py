import json
from typing import Any

import requests
from loguru import logger

from llm.base import LLMBackend


class OllamaBackend(LLMBackend):
    """LLM backend strategy for a locally-hosted Ollama instance.

    Communicates with Ollama's /api/generate endpoint and reassembles
    the NDJSON stream into a single string.

    Example:
        backend = OllamaBackend(host="http://127.0.0.1:11435/api/generate", model="qwen3:8b")
        text = await backend.generate(system="You are helpful.", prompt="Hello")
    """

    def __init__(self, host: str, model: str) -> None:
        """Initialise the Ollama backend.

        Args:
            host: Full URL to the Ollama /api/generate endpoint.
            model: Model tag to use (e.g. "qwen3:8b").
        """
        self._host = host
        self._model = model

    async def generate(self, system: str, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to Ollama and return the assembled response text.

        Args:
            system: System prompt passed to the model.
            prompt: User prompt content.
            **kwargs: Extra Ollama parameters (e.g. temperature, format, top_p).

        Returns:
            The full model response as a single string.

        Raises:
            ValueError: If the response stream contains malformed JSON.
            RuntimeError: If the HTTP request fails.
        """
        params: dict[str, Any] = {
            "model": self._model,
            "system": system,
            "prompt": prompt,
            "format": "json",
            "do_sample": True,
            **kwargs,
        }
        try:
            response = requests.post(url=self._host, json=params)
            response.raise_for_status()
            return self._combine_streams(response)
        except requests.RequestException as exc:
            logger.error(f"Ollama request failed: {exc}")
            raise RuntimeError("Ollama backend request failed.") from exc

    def _combine_streams(self, response: requests.Response) -> str:
        """Reassemble an Ollama NDJSON streaming response into a single string.

        Args:
            response: The raw streaming HTTP response from Ollama.

        Returns:
            Concatenated response text.

        Raises:
            ValueError: If any stream line contains malformed JSON.
        """
        parts: list[str] = []
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error(f"Failed to decode JSON line: {line}. Error: {exc}")
                raise ValueError("Malformed JSON in Ollama stream.") from exc

            parts.append(data.get("response", ""))
            if data.get("done", False):
                break

        return "".join(parts)
