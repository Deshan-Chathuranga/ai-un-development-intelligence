import json

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.2"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    # Cell 1: Intro
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Assignment 1: Local-LLM Country Development Intelligence & Evaluation\n",
            "\n",
            "This notebook implements the complete backend data pipeline for processing UN Human Development Reports, extracting qualitative/quantitative indicators using Retrieval-Augmented Generation (RAG), and benchmarking performance across three local LLM architectures (`llama3.2`, `qwen2.5:3b`, and `phi3:mini`) using the **LLM-as-a-Judge** framework.\n",
            "\n",
            "## Objectives\n",
            "- **PDF Processing & Chapter Segmentation:** Dynamic extraction and text cleaning of country reports.\n",
            "- **Local Embedding & Vector Indexing:** semantic chunk search using `nomic-embed-text` and local caching.\n",
            "- **Prompt Engineering & Structured JSON:** Restricting outputs to strict JSON schemas for data pipelines.\n",
            "- **LLM-as-a-Judge Evaluation:** Programmatically scoring Faithfulness and Relevance.\n",
            "- **Cross-LLM Benchmark:** Analyzing speed, word count, and accuracy tradeoffs.\n",
            "- **Policy Case Study:** Applying the pipeline to legislative briefings (`uk_net_zero_briefing.pdf`)."
        ]
    })
    
    # Cell 2: Imports
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Import modular classes and libraries\n",
            "import os\n",
            "import json\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import pandas as pd\n",
            "\n",
            "from src.pdf_processor import extract_raw_text, clean_text, create_chunks, extract_chapter_texts\n",
            "from src.vector_store import LocalVectorStore, cosine_similarity\n",
            "from src.llm_client import (\n",
            "    query_llm, \n",
            "    count_thematic_frequencies, \n",
            "    extract_key_results, \n",
            "    summarize_chapters,\n",
            "    extract_strengths_challenges,\n",
            "    extract_numerical_indicators,\n",
            "    extract_demographic_trends\n",
            ")\n",
            "from src.evaluator import run_model_benchmark, evaluate_extracted_data\n",
            "\n",
            "print(\"All backend modules successfully loaded!\")"
        ]
    })
    
    # Cell 3: PDF text parsing
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 1: PDF text Ingestion & Chapter Segmentation\n",
            "We load the PDF and clean the text of common parsing relics like line-break hyphens. We also locate the page boundaries of each chapter to perform isolated summarization tasks."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "pdf_path = \"bosniaandhercegovina2007en.pdf\"\n",
            "\n",
            "# Extract raw and clean texts\n",
            "print(\"Extracting text...\")\n",
            "raw_text = extract_raw_text(pdf_path)\n",
            "cleaned_text = clean_text(raw_text)\n",
            "print(f\"Extracted {len(raw_text)} characters. Cleaned length: {len(cleaned_text)} characters.\\n\")\n",
            "\n",
            "# Extract chapter boundaries\n",
            "print(\"Segmenting chapters...\")\n",
            "chapter_texts = extract_chapter_texts(pdf_path)\n",
            "for ch_num, text in sorted(chapter_texts.items()):\n",
            "    print(f\"Chapter {ch_num}: {len(text)} characters (preview: '{text[:80].strip()}...')\")"
        ]
    })
    
    # Cell 4: Chunking
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 2: Sliding Window Chunking\n",
            "To prevent the LLM from losing context in long texts, we implement sliding window chunking. This creates segments of 800 characters with an overlap of 150 characters."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "chunk_size = 800\n",
            "overlap = 150\n",
            "\n",
            "chunks = create_chunks(cleaned_text, chunk_size, overlap)\n",
            "print(f\"Created {len(chunks)} text chunks.\")\n",
            "print(f\"Preview of Chunk 0:\\n{'-'*30}\\n{chunks[0]}\\n{'-'*30}\")"
        ]
    })
    
    # Cell 5: Embeddings
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 3: Local Embedding & Vector Store Indexing\n",
            "We generate embeddings using the local model `nomic-embed-text`. We save the generated vectors on disk to allow instantaneous loads in subsequent runs."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "vstore = LocalVectorStore(model_name=\"nomic-embed-text\")\n",
            "cache_path = \"results/vector_db.json\"\n",
            "\n",
            "if os.path.exists(cache_path):\n",
            "    print(\"Loading cached vector index...\")\n",
            "    vstore.load(cache_path)\n",
            "else:\n",
            "    print(\"Vector index cache not found. Generating embeddings...\")\n",
            "    vstore.add_chunks(chunks)\n",
            "    vstore.save(cache_path)"
        ]
    })
    
    # Cell 6: Cosine Similarity
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 4: Semantic Search (Cosine Similarity)\n",
            "When a query is received, we embed it using the same embedding model and calculate the dot product over the norms to determine similarity scores."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "query = \"What is the main theme of the report regarding social exclusion?\"\n",
            "retrieved = vstore.retrieve(query, top_k=3)\n",
            "\n",
            "for idx, (score, text) in enumerate(retrieved):\n",
            "    print(f\"Match {idx+1} (Cosine Similarity: {score:.4f}):\")\n",
            "    print(f\"Text: '{text[:200].strip()}...'\\n\")"
        ]
    })
    
    # Cell 7: Grounded Inference
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 5: Grounded RAG Query Loop\n",
            "By setting strict system boundaries (\"rely ONLY on the verified text below\"), we force the local generator model (`llama3.2`) to answer based solely on the document contents."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "user_query = \"What is the primary definition of social exclusion and what groups are most vulnerable in BiH?\"\n",
            "matched = vstore.retrieve(user_query, top_k=4)\n",
            "context = \"\\n\\n\".join([chunk[1] for chunk in matched])\n",
            "\n",
            "system_prompt = (\n",
            "    \"You are a strict verification assistant. Answer the user question relying ONLY on the verified context block below. \"\n",
            "    \"Do not extrapolate or use external facts. If the answer cannot be found, say 'Information not found'.\"\n",
            ")\n",
            "user_prompt = f\"Context:\\n{context}\\n\\nQuestion: {user_query}\\nAnswer:\"\n",
            "\n",
            "answer, latency = query_llm(\"llama3.2\", system_prompt, user_prompt)\n",
            "print(f\"Answer (Latency: {latency:.2f}s):\\n{answer}\")"
        ]
    })
    
    # Cell 8: Thematic analysis
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 6: Thematic Counts & Sentiment Analysis\n",
            "We analyze the distribution of the 7 core development themes in the country report, visualising the results using matplotlib and seaborn."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "theme_counts = count_thematic_frequencies(cleaned_text)\n",
            "df_themes = pd.DataFrame(list(theme_counts.items()), columns=[\"Theme\", \"Counts\"]).sort_values(by=\"Counts\", ascending=False)\n",
            "\n",
            "# Visualise thematic frequency\n",
            "plt.figure(figsize=(10, 5))\n",
            "sns.set_theme(style=\"whitegrid\")\n",
            "sns.barplot(x=\"Counts\", y=\"Theme\", data=df_themes, palette=\"Blues_r\")\n",
            "plt.title(\"Thematic Frequency Count - Bosnia and Herzegovina Report 2007\")\n",
            "plt.xlabel(\"Count of Theme Word Stems\")\n",
            "plt.ylabel(\"Theme\")\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    # Cell 9: Structured extraction
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 7: Structured Extraction: Indicators, Strengths, Challenges\n",
            "We parse specific numerical statistics and qualitative strengths/challenges using schema-constrained JSON outputs."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Extracting core development indicators...\")\n",
            "indicators, _ = extract_numerical_indicators(\"llama3.2\", vstore)\n",
            "print(f\"Numerical Indicators:\\n{json.dumps(indicators, indent=4)}\\n\")\n",
            "\n",
            "print(\"Extracting qualitative strengths and challenges...\")\n",
            "strengths_challenges, _ = extract_strengths_challenges(\"llama3.2\", vstore)\n",
            "print(f\"Strengths and Challenges:\\n{json.dumps(strengths_challenges, indent=4)}\")"
        ]
    })
    
    # Cell 10: Judge
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 8: Automated Evaluation (LLM-as-a-Judge)\n",
            "We implement a self-checking evaluation loop where a separate LLM auditor grades model responses for **Faithfulness** (no hallucinations) and **Relevance** on a 1-5 scale."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "query = \"What are the key developmental achievements of the country in the last decade?\"\n",
            "matched_chunks = vstore.retrieve(query, top_k=3)\n",
            "context_text = \"\\n\\n\".join([chunk[1] for chunk in matched_chunks])\n",
            "\n",
            "# Generate an answer using llama3.2\n",
            "model_answer, _ = query_llm(\"llama3.2\", \"Answer based ONLY on context.\", f\"Context:\\n{context_text}\\n\\nQuestion: {query}\")\n",
            "\n",
            "# Critique using judge\n",
            "evaluation = evaluate_extracted_data(query, context_text, model_answer, judge_model=\"llama3.2\")\n",
            "print(f\"Model Answer:\\n{model_answer}\\n\")\n",
            "print(f\"Judge Evaluation:\\n{json.dumps(evaluation, indent=4)}\")"
        ]
    })
    
    # Cell 11: Benchmark display
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 9: Multi-Model Benchmark Comparison\n",
            "We compare performance statistics (speed, verbosity, quality) across `llama3.2`, `qwen2.5:3b`, and `phi3:mini` to evaluate trade-offs in local deployment."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load cached pre-processing benchmark summaries\n",
            "summary_path = \"results/summary.json\"\n",
            "if os.path.exists(summary_path):\n",
            "    with open(summary_path, 'r', encoding='utf-8') as f:\n",
            "        summary = json.load(f)\n",
            "    \n",
            "    bench_summary = summary.get(\"benchmark_summary\", {})\n",
            "    df_bench = pd.DataFrame.from_dict(bench_summary, orient='index')\n",
            "    print(\"Cross-LLM Performance Benchmark:\")\n",
            "    print(df_bench[[\"faithfulness\", \"relevance\", \"total_latency\", \"verbosity\"]])\n",
            "    \n",
            "    # Plot quality vs. speed\n",
            "    df_plot = df_bench.reset_index().rename(columns={'index': 'Model'})\n",
            "    plt.figure(figsize=(10, 5))\n",
            "    sns.scatterplot(x=\"total_latency\", y=\"faithfulness\", hue=\"Model\", size=\"verbosity\", sizes=(100, 500), data=df_plot)\n",
            "    plt.title(\"Inference Trade-offs: Faithfulness vs. Latency (Size represents Verbosity)\")\n",
            "    plt.xlabel(\"Total Latency (Seconds)\")\n",
            "    plt.ylabel(\"Faithfulness Score (1-5)\")\n",
            "    plt.ylim(0, 6)\n",
            "    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "else:\n",
            "    print(\"Pre-processed cache summary.json not found. Run preprocess_country.py first.\")"
        ]
    })
    
    # Cell 12: Net zero
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 10: Capstone Case Study — Legislative Policy Net-Zero Analysis\n",
            "We ingest `uk_net_zero_briefing.pdf` and run our RAG client to extract answers to specific statutory timelines and legislative questions."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "answers_path = \"scratch/net_zero_answers.json\"\n",
            "if not os.path.exists(answers_path):\n",
            "    # Run the get_net_zero_answers.py script inline to get findings\n",
            "    print(\"Executing Net Zero analysis RAG pipeline...\")\n",
            "    import subprocess\n",
            "    subprocess.run([\"python3\", \"get_net_zero_answers.py\"])\n",
            "\n",
            "with open(answers_path, 'r', encoding='utf-8') as f:\n",
            "    answers = json.load(f)\n",
            "    \n",
            "print(\"=== CAPSTONE RAG ANALYSIS RESULTS ===\\n\")\n",
            "for key, ans in sorted(answers.items()):\n",
            "    print(f\"Q: {key.replace('_', ' ').title()}:\")\n",
            "    print(f\"A: {ans}\\n{'-'*60}\\n\")"
        ]
    })
    
    # Cell 13: Export Figures for Report
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 11: Export Figures for Report\n",
            "We import and run the automated plot generator to save all analysis figures (thematic counts, narrative page timeline, judge quality metrics, performance trade-offs, indicator extraction comparisons, and the development radar profile) to the `results/plots/` folder."
        ]
    })
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import generate_plots\n",
            "generate_plots.main()"
        ]
    })
    
    with open("run_pipeline.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4)
    print("Successfully generated run_pipeline.ipynb!")

if __name__ == "__main__":
    create_notebook()
