from llm.base import LLMBackend
from llm.claude_backend import ClaudeBackend
from llm.ollama_backend import OllamaBackend
from utils.load_config import Config


class LLMFactory:
    """Factory for instantiating LLM backend strategies from config.

    Reads the ``models.provider`` key from the application config dict
    and returns the appropriate :class:`LLMBackend` concrete class.

    Example:
        config = load_config()
        backend = LLMFactory.create(config)
    """

    @staticmethod
    def create(config: Config) -> LLMBackend:
        """Instantiate the configured LLM backend.

        Args:
            config: Application config dict as returned by ``load_config()``.
                    Must contain ``models.provider`` and the provider-specific
                    keys (e.g. ``urls.ollama-host``, ``models.ollama``).

        Returns:
            A concrete :class:`LLMBackend` instance.

        Raises:
            ValueError: If the configured provider is not supported.
        """
        provider: str = config.get_config_value("models","provider", "ollama")

        match provider:
            case "ollama":
                return OllamaBackend(
                    host=config.get_config_value("urls","ollama-host"),
                    model=config.get_config_value("models","ollama")
                )
            case "claude":
                return ClaudeBackend(model=config["models"]["claude"])
            case _:
                raise ValueError(f"Unsupported LLM provider: '{provider}'")
