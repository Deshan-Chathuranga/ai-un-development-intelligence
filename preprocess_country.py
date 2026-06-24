import json
import os
import time
from src.pdf_processor import extract_raw_text, clean_text, create_chunks, extract_chapter_texts
from src.vector_store import LocalVectorStore
from src.llm_client import count_thematic_frequencies
from src.evaluator import run_model_benchmark

def main():
    pdf_path = "bosniaandhercegovina2007en.pdf"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    print("=== STARTING PRE-PROCESSING FOR COUNTRY REPORT ===")
    start_time = time.time()
    
    # 1. Extract and Clean Text
    print("Step 1: Extracting raw text from PDF...")
    raw_text = extract_raw_text(pdf_path)
    print(f"Extracted raw text. Length: {len(raw_text)} characters.")
    
    print("Step 2: Cleaning text...")
    cleaned_text = clean_text(raw_text)
    print(f"Cleaned text. Length: {len(cleaned_text)} characters.")
    
    # 2. Extract Chapters
    print("Step 3: Extracting chapter texts...")
    chapter_texts = extract_chapter_texts(pdf_path)
    print(f"Chapters extracted: {list(chapter_texts.keys())}")
    
    # 3. Create Chunks and Index Vector Database
    print("Step 4: Creating sliding chunks...")
    chunks = create_chunks(cleaned_text, chunk_size=800, overlap=150)
    print(f"Created {len(chunks)} chunks.")
    
    vector_db_path = os.path.join(results_dir, "vector_db.json")
    vstore = LocalVectorStore(model_name="nomic-embed-text")
    
    if os.path.exists(vector_db_path):
        print("Vector database cache found. Loading...")
        vstore.load(vector_db_path)
    else:
        print("Vector database cache not found. Building and saving...")
        vstore.add_chunks(chunks)
        vstore.save(vector_db_path)
        
    # 4. Count Theme Frequencies
    print("Step 5: Counting thematic frequencies...")
    theme_counts = count_thematic_frequencies(cleaned_text)
    print(f"Thematic Counts: {theme_counts}")
    
    # 5. Run benchmarks for the three models
    models = ["llama3.2", "qwen2.5:3b", "phi3:mini"]
    benchmarks = {}
    
    for model in models:
        model_safe_name = model.replace(":", "_").replace(".", "_")
        model_result_path = os.path.join(results_dir, f"{model_safe_name}_results.json")
        
        if os.path.exists(model_result_path):
            print(f"Benchmark cache for {model} found. Loading...")
            with open(model_result_path, 'r', encoding='utf-8') as f:
                benchmarks[model] = json.load(f)
        else:
            print(f"Running benchmark for {model}...")
            # We pass Chapter 1, 2, 3, 5, 6, 7, 8, 9, 10
            # To speed up the pre-processing during runtime, let's limit chapter summarization to first 5 chapters
            # But wait, let's do all chapters if possible, or limit to the first 4 for faster run.
            # Actually, to make it completely complete, let's run all chapters. 
            # If the chapter text is very long, it is already truncated to 12000 chars in llm_client.py.
            try:
                results = run_model_benchmark(model, vstore, chapter_texts, judge_model="llama3.2")
                benchmarks[model] = results
                with open(model_result_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"Error benching {model}: {e}")
                
    # 6. Save Global Summary
    summary = {
        "country": "Bosnia and Herzegovina",
        "year": 2007,
        "report_title": "Social Inclusion in Bosnia and Herzegovina",
        "theme_counts": theme_counts,
        "models_tested": models,
        "benchmark_summary": {
            model: {
                "faithfulness": benchmarks[model]["evaluation"]["faithfulness_score"] if model in benchmarks and "evaluation" in benchmarks[model] else None,
                "relevance": benchmarks[model]["evaluation"]["relevance_score"] if model in benchmarks and "evaluation" in benchmarks[model] else None,
                "justification": benchmarks[model]["evaluation"]["justification"] if model in benchmarks and "evaluation" in benchmarks[model] else "",
                "total_latency": benchmarks[model]["metrics"]["total_latency"] if model in benchmarks and "metrics" in benchmarks[model] else None,
                "verbosity": benchmarks[model]["metrics"]["verbosity_word_count"] if model in benchmarks and "metrics" in benchmarks[model] else None,
                "indicators": benchmarks[model]["numerical_indicators"] if model in benchmarks else {}
            } for model in models
        }
    }
    
    with open(os.path.join(results_dir, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
        
    print(f"=== PRE-PROCESSING COMPLETED IN {time.time() - start_time:.2f} SECONDS ===")

if __name__ == "__main__":
    main()
