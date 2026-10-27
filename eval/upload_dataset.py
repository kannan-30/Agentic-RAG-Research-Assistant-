"""
eval/upload_dataset.py — Upload/update the evaluation dataset to LangSmith.

Run once to create the dataset; re-running is idempotent (compares existing
example IDs so no duplicates are created).

Usage:
    python -m eval.upload_dataset
    python -m eval.upload_dataset --dataset-name "my-custom-name"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
DEFAULT_DATASET_NAME = "agentic-rag-eval-v1"


def upload_dataset(dataset_name: str = DEFAULT_DATASET_NAME) -> None:
    """
    Create or update the LangSmith evaluation dataset idempotently.

    Compares existing examples by question text; skips any that already exist.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key or api_key == "your_langsmith_api_key_here":
        print("ERROR: LANGSMITH_API_KEY not set. Add it to .env")
        return

    try:
        from langsmith import Client  # noqa: PLC0415
    except ImportError:
        print("ERROR: langsmith not installed. pip install langsmith")
        return

    client = Client()

    # Load the local dataset
    examples = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples from {DATASET_PATH}")

    # Create dataset if it doesn't exist
    existing_datasets = {d.name: d for d in client.list_datasets()}
    if dataset_name in existing_datasets:
        dataset = existing_datasets[dataset_name]
        print(f"Dataset '{dataset_name}' already exists (id={dataset.id})")
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=(
                "Evaluation dataset for the Agentic RAG chatbot. "
                "35 questions across 6 categories: factual, comparison, multi_hop, "
                "followup, out_of_scope, adversarial."
            ),
        )
        print(f"Created dataset '{dataset_name}' (id={dataset.id})")

    # Get existing example questions to avoid duplicates
    existing = list(client.list_examples(dataset_id=dataset.id))
    existing_questions = {
        ex.inputs.get("question", "") for ex in existing if ex.inputs
    }
    print(f"Dataset has {len(existing)} existing examples.")

    # Upload new examples
    added = 0
    skipped = 0
    for ex in examples:
        question = ex["question"]
        if question in existing_questions:
            skipped += 1
            continue

        client.create_example(
            inputs={
                "question": question,
                "expected_sources": ex.get("expected_sources", []),
                "category": ex.get("category", "factual"),
            },
            outputs={
                "reference_answer": ex.get("reference_answer", ""),
            },
            dataset_id=dataset.id,
        )
        added += 1

    print(f"Added {added} new examples, skipped {skipped} duplicates.")
    print(f"\nDataset URL: https://smith.langchain.com/datasets/{dataset.id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload evaluation dataset to LangSmith.")
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help=f"Dataset name in LangSmith (default: {DEFAULT_DATASET_NAME})",
    )
    args = parser.parse_args()
    upload_dataset(args.dataset_name)
