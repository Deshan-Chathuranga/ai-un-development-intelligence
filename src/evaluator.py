import json
import re
import time
import ollama
from .llm_client import query_llm, extract_key_results, summarize_chapters, extract_strengths_challenges, extract_numerical_indicators, extract_demographic_trends

def evaluate_extracted_data(query, context_text, model_answer, judge_model="llama3.2"):
    """
    Implements the LLM-as-a-Judge pattern to critique model output.
    Returns a dictionary with faithfulness score, relevance score, and justification.
    """
    judge_prompt = f"""You are an expert AI Quality Auditor. Your role is to critique an LLM's answer against the verified Ground Truth Context.
Evaluate based on two specific metrics:
1. Faithfulness (Score 1-5): Is the answer free of hallucinations and strictly backed by the context?
2. Relevance (Score 1-5): Does the answer directly address the user's question without fluff?

You must return your evaluation in strict JSON format matching the schema below. Do not write any conversational intro or outro.

Ground Truth Context:
{context_text}

User Question:
{query}

LLM Answer Under Evaluation:
{model_answer}

Expected JSON Response Schema:
{{
    "faithfulness_score": int,
    "relevance_score": int,
    "justification": "string"
}}
"""
    
    try:
        response = ollama.chat(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            format="json",
            options={"temperature": 0.0}
        )
        content = response.message.content
        parsed = json.loads(content)
        return parsed
    except Exception as e:
        print(f"Judge evaluation failed: {e}")
        # Clean potential markdown wrappers
        try:
            clean_content = re.sub(r'```json\s*|\s*```', '', response.message.content).strip()
            return json.loads(clean_content)
        except Exception:
            return {
                "faithfulness_score": 3,
                "relevance_score": 3,
                "justification": "Fallback evaluation: Judge failed to generate valid JSON."
            }

def run_model_benchmark(model_name, vector_store, chapter_texts, judge_model="llama3.2"):
    """
    Runs the entire pipeline for a given model, measuring latency, verbosity, and quality.
    """
    print(f"\n=========================================\nBenchmarking model: {model_name}...")
    start_time = time.time()
    
    # 1. Key Results
    print("Running key results extraction...")
    key_res, kr_lat = extract_key_results(model_name, vector_store)
    
    # 2. Chapter summaries (limit to first 3 chapters to save time on slow local machines, or run all if needed)
    # We will do all if small, but let's do all chapters to satisfy the assignment
    print("Running chapter summarisation...")
    ch_sums, ch_lat = summarize_chapters(model_name, chapter_texts)
    
    # 3. Strengths and challenges
    print("Running strengths and challenges extraction...")
    str_chal, sc_lat = extract_strengths_challenges(model_name, vector_store)
    
    # 4. Numerical indicators
    print("Running core numerical indicators extraction...")
    indicators, ind_lat = extract_numerical_indicators(model_name, vector_store)
    
    # 5. Demographic trends
    print("Running demographic trends extraction...")
    trends, tr_lat = extract_demographic_trends(model_name, vector_store)
    
    # Compile performance stats
    verbosity = 0
    if key_res:
        verbosity += len(key_res.split())
    for s in ch_sums.values():
        verbosity += len(s.split())
    verbosity += len(str(str_chal).split())
    verbosity += len(str(indicators).split())
    verbosity += len(str(trends).split())
    
    # Run evaluation on Key Results as a proxy for the model's performance
    print("Running LLM-as-a-Judge evaluation on Key Results...")
    query = "What are the most important key results, main findings, development achievements, and policy goals in this Human Development Report?"
    matched_chunks = vector_store.retrieve(query, top_k=5)
    context_text = "\n\n".join([chunk[1] for chunk in matched_chunks])
    
    evaluation = evaluate_extracted_data(query, context_text, key_res, judge_model=judge_model)
    
    total_time = time.time() - start_time
    
    benchmark_results = {
        "model_name": model_name,
        "key_results": key_res,
        "chapter_summaries": ch_sums,
        "strengths_challenges": str_chal,
        "numerical_indicators": indicators,
        "demographic_trends": trends,
        "metrics": {
            "key_results_latency": kr_lat,
            "chapter_summaries_latency": ch_lat,
            "strengths_challenges_latency": sc_lat,
            "numerical_indicators_latency": ind_lat,
            "demographic_trends_latency": tr_lat,
            "total_latency": total_time,
            "verbosity_word_count": verbosity
        },
        "evaluation": evaluation
    }
    
    print(f"Benchmark completed in {total_time:.2f} seconds.")
    return benchmark_results
