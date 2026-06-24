import os
import sys
import json

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.pdf_processor import scan_themes_by_page

def test_page_scanner():
    print("--- Testing scan_themes_by_page on Bosnia Report ---")
    pdf_path = "bosniaandhercegovina2007en.pdf"
    if not os.path.exists(pdf_path):
        print(f"[Error] {pdf_path} not found in workspace root.")
        sys.exit(1)
        
    results = scan_themes_by_page(pdf_path)
    print(f"Scanner successfully processed the PDF and returned {len(results)} matches.")
    
    if len(results) > 0:
        print("Sample extracted page-theme matches:")
        for match in results[:5]:
            print(f"  Page {match['page']}: theme '{match['theme']}' is present.")
            
        # Verify schema
        assert "page" in results[0]
        assert "theme" in results[0]
        assert "presence" in results[0]
        
    print("[Success] Page scanner validation complete.")

def test_cache_structure():
    print("\n--- Testing cache structure in results/summary.json ---")
    cache_path = "results/summary.json"
    if not os.path.exists(cache_path):
        print(f"[Warning] Cache {cache_path} not found. Skipped.")
        return
        
    with open(cache_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
        
    # Check theme_counts
    assert "theme_counts" in summary, "Missing theme_counts in cache"
    print("Theme counts in cache:")
    print(summary["theme_counts"])
    
    print("[Success] Cache structure verification complete.")

if __name__ == "__main__":
    try:
        test_page_scanner()
        test_cache_structure()
        print("\nAll country plot checks passed successfully!")
    except AssertionError as e:
        print(f"\n[Failure] Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Failure] Unexpected error: {e}")
        sys.exit(1)
