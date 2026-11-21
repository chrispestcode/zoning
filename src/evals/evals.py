"""Retrieval evaluation for the zoning RAG pipeline.

Run from the command line (with ``src`` on the import path):

    python evals/evals.py generate            # build a QA eval set (JSONL) from the corpus
    python evals/evals.py evaluate [FILE]     # score Qdrant retrieval against an eval set

``evaluate`` uses the newest ``qa_pairs_*.jsonl`` in the configured evals folder
when no file is given.

Naming convention used by every metric below:
    actual    -- document IDs the retrieval system returned, in rank order
    predicted -- ground-truth document IDs the answer should come from

TODO: precision@k / recall@k, DCG / NDCG, MMR.
"""
import argparse
import asyncio
from typing import Union, Any
import json
import random
from datetime import datetime
from pathlib import Path

import pandas
from loguru import logger
from pydantic import ValidationError
from tqdm import tqdm

from content_generator import ContentGenerator
from models.contracts import DerivativeCreate, QAPairCreate, Status, QAPairModel
from document_database import DocumentDatabase
from qdrant import QdrantManager
from utils.load_config import get_config


# ============================================================================
# Metrics
# ============================================================================
config = get_config()

def calculate_recall(actual: list[int], predicted: list[int]) -> float:
    """Fraction of ground-truth documents that were retrieved."""
    if not predicted:
        return 0.0
    return len(set(actual) & set(predicted)) / len(set(predicted))


def calculate_precision(actual: list[int], predicted: list[int]) -> float:
    """Fraction of retrieved results that belong to a ground-truth document."""
    if not actual:
        return 0.0
    relevant = set(predicted)
    return sum(1 for a in actual if a in relevant) / len(actual)


def calculate_f1(actual: list[int], predicted: list[int]) -> float:
    """Harmonic mean of precision and recall; 0.0 when both are 0."""
    recall = calculate_recall(actual, predicted)
    precision = calculate_precision(actual, predicted)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_reciprocal_rank(actual: list[int], predicted: list[int]) -> float:
    """1 / rank of the first retrieved result that is a ground-truth document."""
    relevant = set(predicted)
    for rank, a in enumerate(actual, start=1):
        if a in relevant:
            return 1 / rank
    return 0.0


def calculate_average_precision(actual: list[int], predicted: list[int]) -> float:
    """Mean of precision@rank taken at each rank where a relevant result appears."""
    relevant = set(predicted)
    hits = 0
    total = 0.0
    for rank, a in enumerate(actual, start=1):
        if a in relevant:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0


def calculate_mean_reciprocal_rank(queries: list[tuple[list[int], list[int]]]) -> float:
    """MRR over (actual, predicted) pairs."""
    if not queries:
        return 0.0
    return sum(calculate_reciprocal_rank(a, p) for a, p in queries) / len(queries)


def calculate_mean_average_precision(queries: list[tuple[list[int], list[int]]]) -> float:
    """MAP over (actual, predicted) pairs."""
    if not queries:
        return 0.0
    return sum(calculate_average_precision(a, p) for a, p in queries) / len(queries)


def calculate_discounted_cumulative_gain(actual: list[int], predicted: list[int]) -> float:
    raise NotImplementedError


def calculate_normalized_discounted_cumulative_gain(actual: list[int], predicted: list[int]) -> float:
    raise NotImplementedError


def calculate_max_marginal_relevance(actual: list[int], predicted: list[int]) -> float:
    """
    Iteratively pick the candidate maximizing relevance to the query minus
    similarity to what is already selected:
        max(lambda * sim(cand, query) - (1 - lambda) * sim(cand, prev_candidate)),
        with 0.5 < lambda < 0.8
    """
    raise NotImplementedError

async def run_evaluation_metrics(cg: Union[ContentGenerator|None] , question, context, answer, expected_answer) \
        -> dict[Any, Any]:
    if cg is None:
        cg = ContentGenerator()
    FAITHFULNESS_RUBRIC = ''' 
    Given the CONTEXT, label the RESPONSE either : 1) ACCEPTABLE if all parts of the RESPONSE
     is found in the CONTEXT, 2) ACCEPTABLE if RESPONSE references information not found in CONTEXT.
     INVALID should be used for any malformations. Provide 1-2 sentence rationale for UNACCEPTABLE OR INVALID".
     Return the answer as a json.
    
    <input>
    <CONTEXT>
    </CONTEXT>
    <RESPONSE>
    </RESPONSE>
    </input>
    <output>
    {"label": "ACCEPTABLE" / "UNACCEPTABLE" / "INVALID",
    "rationale": str("Provide 1-2 sentence rationale for ACCEPTABLE OR INVALID"}
    </output>
    '''

    ANSWER_RELEVANCE_RUBRIC = '''Given the RESPONSE, label the RESPONSE either : 1) ACCEPTABLE if all parts of QUERY
     is aligned with RESPONSE, 2) UNACCEPTABLE if RESPONSE does not align with the intent of QUERY. INVALID should be 
     used for any malformations. Provide 1-2 sentence rationale for UNACCEPTABLE OR INVALID".
     Return the answer as a json.
    
    <input>
    <QUERY>
    </QUERY>
    <RESPONSE>
    </RESPONSE>
    </input>
    <output>
    {"label": "ACCEPTABLE" / "UNACCEPTABLE" / "INVALID",
    "rationale": str("rationale"}
    </output> '''
    evals = {"faithfulness": FAITHFULNESS_RUBRIC, "answer_relevance": ANSWER_RELEVANCE_RUBRIC}

    evals_results = {}
    for metric_name, metric_prompt in evals.items():
        if metric_name == "faithfulness":
            content = {
                'CONTEXT': context,
                'RESPONSE': answer
            }
        elif metric_name == "answer_relevance":
            content = {
                'QUERY': question,
                'RESPONSE': answer
            }
        else:
            continue
        try:
            result = await cg.run_llm_as_a_judge(metric_prompt, content)
            evals_results[metric_name] = result
        except Exception as e:
            logger.error(f"Metric could not be logged: {metric_name} for {question}")

    return evals_results

# ============================================================================
# Eval set generation
# ============================================================================

def _evals_dir() -> Path:
    return config.get_config_path(get_config(), 'paths', 'evals')

def _evals_log_dir() -> Path:
    return config.get_config_path(get_config(), 'paths', 'logs', 'evals')

async def generate_eval_dataset(num_documents: int = 5, num_derivatives: int = 20) -> Path | None:
    """
    Create derivatives for up to ``num_documents`` ungenerated documents, then
    generate QA pairs from ``num_derivatives`` random derivatives and write them
    to a new timestamped JSONL file. Returns the file path, or None if there
    were no derivatives to work from.
    """
    db = DocumentDatabase()
    cg = ContentGenerator()

    documents = db.get_documents(Status.PROCESSING_COMPLETED)
    all_ids = [doc.DocumentId for doc in documents]
    doc_ids = random.sample(all_ids, k=min(num_documents, len(all_ids)))
    for doc_id in tqdm(doc_ids, desc="Generating derivatives", leave=False):
        if db.is_document_generated(doc_id):
            continue
        for c in db.get_chunks_by_document_id(doc_id, Status.PROCESSING_COMPLETED):
            key_points = await cg.get_derivatives_chunks(c.ChunkText)
            derivative_rows: list[DerivativeCreate] = []
            for kp in key_points:
                try:
                    derivative_rows.append(
                        DerivativeCreate(DocumentId=c.DocumentId, ChunkId=c.ChunkId, DerivativeText=kp)
                    )
                except ValidationError as exc:
                    logger.warning(f"Skipping invalid derivative for chunk {c.ChunkId}: {exc}")
            db.insert_derivatives(derivative_rows)

    derivatives = db.get_random_derivatives(num_derivatives)
    if not derivatives:
        logger.warning("No derivatives available; skipping QA generation")
        return None
    source_ids = sorted({d.DocumentId for d in derivatives})
    derivative_ids = [d.DerivativeId for d in derivatives]

    qa_pairs = await cg.get_qa_pairs("\n".join(d.DerivativeText for d in derivatives))
    qa_rows: list[QAPairCreate] = []
    for pair in qa_pairs:
        try:
            qa_rows.append(
                QAPairCreate(Question=pair["question"], Answer=pair["answer"], DocumentIds=source_ids)
            )
        except (KeyError, TypeError, ValidationError) as exc:
            logger.warning(f"Skipping malformed QA pair {pair!r}: {exc}")

    output_dir = _evals_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"qa_pairs_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for qa in qa_rows:
            f.write(json.dumps({
                "question": qa.Question,
                "answer": qa.Answer,
                "doc_ids": qa.DocumentIds,
                "derivative_ids": derivative_ids,
            }, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(qa_rows)} QA pairs to {output_path}")
    return output_path


# ============================================================================
# Evaluation
# ============================================================================

def latest_eval_file() -> Path:
    """Newest qa_pairs_*.jsonl in the evals folder (timestamped names sort chronologically)."""
    files = sorted(_evals_dir().glob("qa_pairs_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No qa_pairs_*.jsonl files in {_evals_dir()}")
    return files[-1]


def load_eval_set(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(eval_file: Path, top_k: int = 10) -> dict[str, float]:
    """Query Qdrant with each eval question, log per-query metrics, and return MAP and MRR."""
    eval_set = load_eval_set(eval_file)
    logger.info(f"Evaluating {len(eval_set)} questions from {eval_file}")
    qdrant_manager = QdrantManager(initialize=False)  # Don't initialize collections on startup

    query_pairs: list[tuple[list[int], list[int]]] = []
    individual_results = []
    for pair in eval_set:
        question = pair["question"]
        predicted = pair["doc_ids"]
        results = qdrant_manager.query_qdrant(question, top_k=top_k)
        if isinstance(results, dict):  # query_qdrant returns {"results": []} on failure
            results = results.get("results", [])
        actual = [item.payload.get("document_id") for item in results]
        result = {
            "question": {question},
            "recall": calculate_recall(actual, predicted),
            "precision": calculate_precision(actual, predicted),
            "f1_score": calculate_f1(actual, predicted),
            "reciprocal_rank": calculate_reciprocal_rank(actual, predicted),
            "average_precision": calculate_average_precision(actual, predicted)
        }
        individual_results.append(result)
        query_pairs.append((actual, predicted))

    summary = {
        "MAP": calculate_mean_average_precision(query_pairs),
        "MRR": calculate_mean_reciprocal_rank(query_pairs),
        "individual_results": individual_results
    }
    for name, value in summary.items():
        logger.info(f"{name}: {value}")
    return summary

async def evaluate_generation(eval_file: Path):
    eval_set = load_eval_set(eval_file)
    logger.info(f"Evaluating {len(eval_set)} questions from {eval_file}")
    cg = ContentGenerator()
    db = DocumentDatabase()
    output_dir = _evals_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {}
    for idx, item in tqdm(enumerate(eval_set), desc="Running generation metrics..."):
        # TODO: JSONL pairs aren't stored in the QAPair table, so the line index stands in for QAPairId
        eval: QAPairModel = None

        # 1. Prepare Eval Dataset
        try:
            eval = QAPairModel(
                QAPairId=idx,
                Question=item["question"],
                Answer=item["answer"],
                DocumentIds=item.get("doc_ids", []),
                DerivativeIds=item.get("derivative_ids", []),
            )
        except (KeyError, ValidationError) as exc:
            logger.warning(f"Eval #{idx}: skipping malformed line in {eval_file}: {exc}")
            continue
        chunk_ids = [d.ChunkId for d in db.get_derivatives(eval.DerivativeIds)]
        chunks = [c.ChunkText for c in db.get_chunks_by_ids(chunk_ids)]
        if not chunks:
            logger.warning(f"Eval #{idx}: No derivative chunks found in {eval_file}")

        # 2. Generate response for query
        response = await cg.get_chunk_response(eval.Question, chunks) # chunk response should contain answer as well as citations in bulk.
        if not response:
            logger.error("Eval #{idx}: No response could be generated")
            continue
       # Run evaluation metrics
        results = await run_evaluation_metrics(cg, question=eval.Question, context="|".join(chunks),
                                               answer=response["text_response"], expected_answer=eval.Answer)
        results_dict[idx] =  {'question': eval.Question, **results}

    output_path = output_dir / f"evals_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df = pandas.DataFrame.from_dict(results_dict, orient="index")
    df.to_csv(output_path, index=False)







# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate a QA eval set")
    gen.add_argument("--documents", type=int, default=5, help="Documents to sample for derivatives")
    gen.add_argument("--derivatives", type=int, default=20, help="Derivatives to generate QA pairs from")

    ev = sub.add_parser("evaluate", help="Score retrieval against an eval set")
    ev.add_argument("file", nargs="?", type=Path, help="Eval set JSONL (default: newest in evals folder)")
    ev.add_argument("--top-k", type=int, default=10, help="Results to retrieve per question")
    args = parser.parse_args()
    if args.command == "generate":
        asyncio.run(generate_eval_dataset(args.documents, args.derivatives))
    else:
        eval_file = _evals_dir() / "qa_pairs_20261031_204618.jsonl"  # hardcoded for now
        retrieval_metrics = evaluate(eval_file, top_k=10)
        generation_metrics = asyncio.run(evaluate_generation(eval_file))



if __name__ == "__main__":
    main()
