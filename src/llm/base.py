from abc import ABC, abstractmethod
from typing import Any


class LLMBackend(ABC):
    """Abstract base class for LLM backend strategies.

    Each concrete backend implements the transport and response
    normalisation for a specific LLM provider.

    Example:
        class MyBackend(LLMBackend):
            async def generate(self, system: str, prompt: str, **kwargs: Any) -> str:
                ...
    """

    @abstractmethod
    async def generate(self, system: str, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the LLM and return the raw text response.

        Args:
            system: System / instruction prompt.
            prompt: User-facing prompt content.
            **kwargs: Provider-specific generation parameters
                      (e.g. temperature, top_p, format).

        Returns:
            The model's raw text output.

        Raises:
            RuntimeError: If the backend fails to produce a response.
        """
