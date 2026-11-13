from typing import Any

import anthropic
from loguru import logger

from llm.base import LLMBackend
from dotenv import load_dotenv
import os

class ClaudeBackend(LLMBackend):
    """LLM backend strategy for Anthropic's Claude API.

    Uses ``AsyncAnthropic`` for non-blocking I/O.

    Example:
        backend = ClaudeBackend(model="claude-sonnet-4-6")
        text = await backend.generate(system="You are helpful.", prompt="Hello")
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        """Initialise the Claude backend.

        Args:
            model: Anthropic model ID to use.
        """
        load_dotenv()
        self._model = model
        self._api_key = os.getenv("ANTHROPIC_API_KEY")
        self._client = anthropic.AsyncAnthropic(api_key=self._api_key)

    async def generate(self, system: str, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to Claude and return the response text.

        Args:
            system: System prompt passed to the model.
            prompt: User prompt content.
            **kwargs: Optional Claude parameters.

        Returns:
            The model's text response.

        Raises:
            RuntimeError: If the Anthropic API call fails.
        """
        max_tokens: int = kwargs.pop("max_tokens", 1024)

        try:
            message = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system = system,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return message.content[0].text
        except anthropic.APIError as exc:
            logger.error(f"Claude API error: {exc}")
            raise RuntimeError("Claude backend request failed.") from exc
