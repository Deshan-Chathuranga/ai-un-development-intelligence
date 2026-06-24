import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.pdf_processor import extract_raw_text, clean_text, create_chunks, extract_chapter_texts
    from src.vector_store import LocalVectorStore, cosine_similarity
    from src.llm_client import query_llm, count_thematic_frequencies
    from src.evaluator import run_model_benchmark, evaluate_extracted_data
    print("Verification SUCCESS: All backend package modules compiled and imported successfully!")
except Exception as e:
    print(f"Verification FAILURE: {e}")
    sys.exit(1)
