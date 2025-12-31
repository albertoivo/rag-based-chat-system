from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional, Set
import json
import os
import argparse
import numpy as np
import rag_client
import llm_client

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

# ============================================================================
# AVAILABLE METRICS REGISTRY
# ============================================================================
# This registry documents all supported evaluation metrics.
# Core metrics (response_relevancy, faithfulness) are always computed.
# Additional metrics can be enabled via the 'additional_metrics' parameter
# or the --metrics CLI flag.
#
# Available metrics:
#   - response_relevancy: Measures how relevant the response is to the question
#   - faithfulness: Measures if the response is grounded in the context
#   - bleu: BLEU score for text similarity (requires reference)
#   - rouge: ROUGE score for text overlap (requires reference)
#   - precision: Context precision without LLM (requires reference)
#
# Usage examples:
#   CLI: python ragas_evaluator.py --metrics bleu rouge
#   API: evaluate_response_quality(q, a, ctx, additional_metrics=["bleu", "rouge"])
# ============================================================================

# Core metrics that are always computed
CORE_METRICS = {"response_relevancy", "faithfulness"}

# Additional metrics that can be enabled
ADDITIONAL_METRICS = {"bleu", "rouge", "precision"}

# All available metrics
ALL_METRICS = CORE_METRICS | ADDITIONAL_METRICS


def get_metric_instances(metric_names: Set[str], evaluator_llm, evaluator_embeddings) -> List:
    """
    Create metric instances based on requested metric names.
    
    Args:
        metric_names: Set of metric names to instantiate
        evaluator_llm: LLM wrapper for metrics that need it
        evaluator_embeddings: Embeddings wrapper for metrics that need it
    
    Returns:
        List of metric instances
    """
    metrics = []
    
    # Core metrics (always included)
    if "response_relevancy" in metric_names:
        metrics.append(ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings))
    
    if "faithfulness" in metric_names:
        metrics.append(Faithfulness(llm=evaluator_llm))
    
    # Additional metrics (enabled on request)
    if "bleu" in metric_names:
        metrics.append(BleuScore())
    
    if "rouge" in metric_names:
        metrics.append(RougeScore())
    
    if "precision" in metric_names:
        metrics.append(NonLLMContextPrecisionWithReference())
    
    return metrics


def evaluate_response_quality(
    question: str, 
    answer: str, 
    contexts: List[str],
    additional_metrics: Optional[List[str]] = None,
    reference: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate response quality using RAGAS metrics.
    
    Core metrics (ResponseRelevancy, Faithfulness) are always computed.
    Additional metrics can be enabled via the additional_metrics parameter.
    
    Args:
        question: The user's question
        answer: The generated answer
        contexts: List of retrieved context strings
        additional_metrics: Optional list of additional metrics to compute.
                           Supported values: "bleu", "rouge", "precision"
        reference: Optional reference answer (required for bleu, rouge, precision)
    
    Returns:
        Dictionary mapping metric names to scores
    
    Example:
        scores = evaluate_response_quality(
            question="What was Apollo 11?",
            answer="Apollo 11 was the first moon landing mission.",
            contexts=["Apollo 11 landed on the moon in 1969..."],
            additional_metrics=["bleu", "rouge"]
        )
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    if not question or not answer or not contexts:
        return {"error": "Invalid input: Question, answer, and contexts are required for evaluation."}
    
    # Determine which metrics to compute
    metrics_to_compute = CORE_METRICS.copy()
    
    if additional_metrics:
        for metric in additional_metrics:
            metric_lower = metric.lower()
            if metric_lower in ADDITIONAL_METRICS:
                metrics_to_compute.add(metric_lower)
            else:
                print(f"Warning: Unknown metric '{metric}'. Available: {ADDITIONAL_METRICS}")
    
    # Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
    
    # Create evaluator_embeddings with model text-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    
    # Get metric instances
    metrics = get_metric_instances(metrics_to_compute, evaluator_llm, evaluator_embeddings)
    
    # Create sample for evaluation
    sample_kwargs = {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": contexts
    }
    
    # Add reference if provided (needed for BLEU, ROUGE, Precision)
    if reference:
        sample_kwargs["reference"] = reference
    
    sample = SingleTurnSample(**sample_kwargs)
    
    # Evaluate the response using the metrics
    try:
        results = {}
        for metric in metrics:
            try:
                score = metric.single_turn_score(sample)
                results[metric.__class__.__name__] = score
            except Exception as metric_error:
                # Some metrics may fail if reference is missing
                results[metric.__class__.__name__] = f"Error: {str(metric_error)[:50]}"
        
        # Return the evaluation results
        return results
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}


def print_available_metrics():
    """Print documentation of available metrics."""
    print("\n=== AVAILABLE METRICS ===")
    print("\nCore metrics (always computed):")
    print("  - response_relevancy: Measures how relevant the response is to the question")
    print("  - faithfulness: Measures if the response is grounded in the retrieved context")
    print("\nAdditional metrics (enable with --metrics flag):")
    print("  - bleu: BLEU score for text similarity (may require reference answer)")
    print("  - rouge: ROUGE score for text overlap (may require reference answer)")
    print("  - precision: Context precision without LLM (may require reference answer)")
    print("\nExample usage:")
    print("  python ragas_evaluator.py --metrics bleu rouge")
    print()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="RAGAS Evaluation for RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available metrics:
  Core (always computed):
    - response_relevancy: Measures response relevance to the question
    - faithfulness: Measures if response is grounded in context
  
  Additional (enable with --metrics):
    - bleu: BLEU score for text similarity
    - rouge: ROUGE score for text overlap
    - precision: Context precision without LLM

Examples:
  python ragas_evaluator.py
  python ragas_evaluator.py --metrics bleu rouge
  python ragas_evaluator.py --list-metrics
        """
    )
    parser.add_argument(
        "--metrics", 
        nargs="+", 
        choices=list(ADDITIONAL_METRICS),
        help="Additional metrics to compute beyond core metrics"
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List all available metrics and exit"
    )
    parser.add_argument(
        "--chroma-dir",
        default="./chroma_db_openai",
        help="ChromaDB directory path"
    )
    parser.add_argument(
        "--collection",
        default="nasa_space_missions_text",
        help="ChromaDB collection name"
    )
    parser.add_argument(
        "--questions-file",
        default="test_questions.json",
        help="Path to test questions JSON file"
    )
    
    args = parser.parse_args()
    
    # Handle --list-metrics flag
    if args.list_metrics:
        print_available_metrics()
        exit(0)
    
    # Load test questions
    try:
        with open(args.questions_file, 'r') as f:
            questions = json.load(f)
    except FileNotFoundError:
        print(f"{args.questions_file} not found. Using sample questions.")
        questions = ["What was the primary mission of Apollo 11?", "Who were the astronauts on Apollo 13?"]

    # Check for OpenAI API key first (needed for embedding function)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("OPENAI_API_KEY not set.")
        exit(1)
    
    # Set CHROMA_OPENAI_API_KEY for the embedding function
    os.environ["CHROMA_OPENAI_API_KEY"] = openai_key

    # Initialize RAG
    collection, success, error = rag_client.initialize_rag_system(args.chroma_dir, args.collection)
    
    if not success:
        print(f"Failed to initialize RAG system: {error}")
        exit(1)

    metrics_data = {}
    
    # Display which metrics will be computed
    additional_metrics = args.metrics or []
    print("\n=== STARTING EVALUATION ===")
    print(f"Core metrics: {', '.join(CORE_METRICS)}")
    if additional_metrics:
        print(f"Additional metrics: {', '.join(additional_metrics)}")
    print()

    for q in questions:
        print(f"Question: {q}")
        # Retrieve
        retrieval = rag_client.retrieve_documents(collection, q, n_results=3)
        if not retrieval or not retrieval['documents']:
            print("No documents retrieved.")
            continue
            
        contexts = retrieval['documents'][0]
        metadatas = retrieval['metadatas'][0]
        context_str = rag_client.format_context(contexts, metadatas)
        
        # Generate
        answer = llm_client.generate_response(openai_key, q, context_str, [])
        print(f"Answer: {answer}")
        
        # Evaluate with additional metrics if specified
        scores = evaluate_response_quality(q, answer, contexts, additional_metrics=additional_metrics)
        print(f"Scores: {scores}\n")
        
        for k, v in scores.items():
            if isinstance(v, (int, float)):
                if k not in metrics_data:
                    metrics_data[k] = []
                metrics_data[k].append(v)

    print("\n=== AGGREGATE METRICS ===")
    for k, v in metrics_data.items():
        print(f"{k}: Mean = {np.mean(v):.4f}")

