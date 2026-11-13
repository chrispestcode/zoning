import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, List

from loguru import logger
from pydantic import ValidationError

from llm.base import LLMBackend
from llm.factory import LLMFactory
from prompts.system_prompts import DERIVATIVE_PROMPT, QA_PROMPTS, RESPONSE_PROMPT
from utils.load_config import load_config, get_config_value

class ContentGenerator:
    """Facade for LLM-based content generation tasks.

    Delegates all model I/O to an :class:`~llm.base.LLMBackend` strategy,
    which is resolved at construction time via :class:`~llm.factory.LLMFactory`.

    Example:
        cg = ContentGenerator()
        key_points = await cg.get_derivatives_chunks("Some passage text...")
    """

    def __init__(self, backend: LLMBackend | None = None) -> None:
        """Initialise ContentGenerator.

        Args:
            backend: Optional pre-built :class:`~llm.base.LLMBackend`. When
                     omitted the factory resolves the backend from config.
        """
        config = load_config()
        self._backend: LLMBackend = backend or LLMFactory.create(config)

    async def get_derivatives_chunks(self, chunk: str) -> list[Any]:
        """Extract key points from a text chunk.

        Args:
            chunk: Raw passage text to process.

        Returns:
            List of key-point strings, or an empty list if none were generated.
        """
        try:
            raw = await self._backend.generate(
                system=DERIVATIVE_PROMPT,
                prompt=chunk,
                temperature=0.85,
            )
            key_points: list[str] =  self.format_llm_json_response(raw).get("key-points", [])
            if not key_points:
                logger.warning("Did not generate derivative chunks for given chunk")
            return key_points
        except Exception as exc:
            logger.error(f"get_derivatives_chunks failed: {exc}")
            return []

    async def get_qa_pairs(self, text: str) -> list[Any]:
        """Generate question-and-answer pairs from input text.

        Args:
            text: Source text from which QA pairs are derived.

        Returns:
            List of QA pair dicts, or an empty list if none were generated.
        """
        try:
            raw = await self._backend.generate(
                system=QA_PROMPTS,
                prompt=text,
                temperature=0.34,
            )
            qa_pairs: list[Any] = self.format_llm_json_response(raw).get("questions_and_answers", [])

            if not qa_pairs:
                logger.warning("Did not generate QA pairs for given text")
            return qa_pairs
        except Exception as exc:
            logger.error(f"get_qa_pairs failed: {exc}")
            return []

    async def run_llm_as_a_judge(self, system:str, context: dict[str, Any]) -> dict[str, str]:
        """Answer a user query using the supplied context chunks per metric.

        Args:
            query: The user's question.
            chunks: Retrieved document chunks to use as evidence.

        Returns:
            Dict with key ``"text_response"`` containing the answer string.
            :param context: Required context to inject into metric prompt
            :param system: System prompt
        """
        prompt = ""
        raw = ""
        for key,value in context.items():
                prompt += f'''
                    <{key.upper()}> <{value}> </{key.upper()}>
                '''
        try:
            raw = await self._backend.generate(
                system=system,
                prompt=prompt,
            )
        except Exception as exc:
            logger.error(f"{exc}")

        return self.format_llm_json_response(raw)

    async def get_chunk_response(self, query: str, chunks: List[str]) -> dict[str, str]:
        """Answer a user query using the supplied context chunks.

        Args:
            query: The user's question.
            chunks: Retrieved document chunks to use as evidence.

        Returns:
            Dict with key ``"text_response"`` containing the answer string.
        """
        input_parts = [f"<question>\t{query}\n</question>"]
        for chunk in chunks:
            input_parts.append(f"<chunk>\n\t{chunk}\n</chunk>")

        try:
            raw = await self._backend.generate(
                system=RESPONSE_PROMPT,
                prompt="\n".join(input_parts),
            )

            answer = self.format_llm_json_response(raw).get('answer', None)
            if not answer:
                logger.warning("Did not generate response for given query")
            return {"text_response": answer}
        except Exception as exc:
            logger.error(f"get_chunk_response failed: {exc}")
            return {"text_response": "An error occurred while generating the response."}
        
    def format_llm_json_response(self, raw: str) -> dict[str, str]:
        cleaned = raw.strip(raw.strip().removeprefix("```json").removesuffix("```"))
        return json.loads(cleaned)