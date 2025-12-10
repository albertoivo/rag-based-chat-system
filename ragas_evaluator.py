from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
import json
import os
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

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    if not question or not answer or not contexts:
        return {"error": "Invalid input: Question, answer, and contexts are required for evaluation."}
    
    # Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
    
    # Create evaluator_embeddings with model text-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    
    # Define an instance for each metric to evaluate
    metrics = [
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        Faithfulness(llm=evaluator_llm)
    ]
    
    # Create sample for evaluation
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts
    )
    
    # Evaluate the response using the metrics
    try:
        results = {}
        for metric in metrics:
            score = metric.single_turn_score(sample)
            results[metric.__class__.__name__] = score
        
        # Return the evaluation results
        return results
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}

if __name__ == "__main__":
    # Load test questions
    try:
        with open('test_questions.json', 'r') as f:
            questions = json.load(f)
    except FileNotFoundError:
        print("test_questions.json not found. Using sample questions.")
        questions = ["What was the primary mission of Apollo 11?", "Who were the astronauts on Apollo 13?"]

    # Initialize RAG
    chroma_dir = "./chroma_db_openai"
    collection_name = "nasa_space_missions_text"
    collection, success, _ = rag_client.initialize_rag_system(chroma_dir, collection_name)
    
    if not success:
        print("Failed to initialize RAG system.")
        exit(1)

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("OPENAI_API_KEY not set.")
        exit(1)

    metrics_data = {}

    print("\n=== STARTING EVALUATION ===\n")

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
        
        # Evaluate
        scores = evaluate_response_quality(q, answer, contexts)
        print(f"Scores: {scores}\n")
        
        for k, v in scores.items():
            if isinstance(v, (int, float)):
                if k not in metrics_data:
                    metrics_data[k] = []
                metrics_data[k].append(v)

    print("\n=== AGGREGATE METRICS ===")
    for k, v in metrics_data.items():
        print(f"{k}: Mean = {np.mean(v):.4f}")
