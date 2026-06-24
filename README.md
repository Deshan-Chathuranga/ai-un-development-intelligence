# AI-Powered UN Development Intelligence

A production-grade, local LLM-powered Retrieval-Augmented Generation (RAG) pipeline and interactive analytics dashboard to ingest, process, and analyze UN Human Development Reports. Built using modular Python components, local vector stores, and an automated evaluation framework.

---

## 🚀 Key Features

- **Modular Software Architecture:** Clean python package design (`src/`) separating PDF processing, vector search, LLM inference, and evaluation logic.
- **Persistent Local Embeddings:** Utilizes `nomic-embed-text` with a custom in-memory vector database that persists index states to disk (`vector_db.json`), avoiding redundant computation.
- **Schema-Constrained Extraction:** Forces models to output strict JSON schemas for core quantitative and qualitative indicators, ensuring high data pipeline reliability.
- **LLM-as-a-Judge Evaluation:** Implements an automated auditor loop that grades generated answers against source text snippets for **Faithfulness** (hallucination checks) and **Relevance**.
- **Multi-Model Benchmarking:** Compares performance metrics (Inference Latency, Word Count, and Evaluation Scores) across three local architectures (`llama3.2`, `qwen2.5:3b`, and `phi3:mini`).
- **Interactive Visual Dashboard:** A polished Streamlit interface with dynamic Plotly charts (bar, pie, polar/radar profiles) that loads precomputed caches instantly for easy cloud hosting.
- **Document Narrative Mapping:** Dynamically scans document themes page-by-page using regular expressions and plots thematic timelines to visualize page coordinates where topics are emphasized.

---

## 🛠️ Project Structure

```
ai-un-development-intelligence/
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py   # PDF text extraction, cleaning, chapter boundaries detection, page scanner
│   ├── vector_store.py    # Local embedding generator and Cosine Similarity search engine
│   ├── llm_client.py      # Native Ollama API caller, custom system prompts, and JSON schemas
│   └── evaluator.py       # LLM-as-a-Judge evaluator and multi-model benchmarking suite
├── results/               # Cached extraction results for instantaneous dashboard loading
│   ├── summary.json       # Holds page narrative mapping and theme frequencies
│   ├── llama3_2_results.json
│   ├── qwen2_5_3b_results.json
│   └── phi3_mini_results.json
├── app.py                 # Streamlit static analysis dashboard application
├── run_pipeline.ipynb     # Demonstration notebook walking through RAG execution
├── requirements.txt       # Python package dependencies
└── README.md              # Project documentation and portfolio highlights
```

---

## ⚙️ Setup and Installation

### 1. Prerequisites

- **Ollama:** Download and install from [ollama.com](https://ollama.com).
- **Download Models:** Pull the required local embedding and generator models:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.2
  ollama pull qwen2.5:3b
  ollama pull phi3:mini
  ```

### 2. Python Environment Setup

Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

---

## 📈 How to Run

### 1. Ingestion & Pre-processing (Cache Build)
To process the default country report (`bosniaandhercegovina2007en.pdf`) and cache the multi-model extraction results (saving minutes of live execution time):
```bash
python3 preprocess_country.py
```

### 2. Launch the Streamlit Dashboard
To run the interactive visual dashboard:
```bash
streamlit run app.py
```
*Note: The dashboard is configured to load results directly from the `results/` cache, making it extremely fast, lightweight, and perfect for free cloud hosting (e.g., Streamlit Community Cloud).*

### 3. Run the Demonstration Notebook
To step through the RAG code and evaluation logic:
```bash
jupyter notebook run_pipeline.ipynb
```

---

## 💼 Portfolio Highlights (Skills Demonstrated)

- **Local GenAI & LLM Orchestration:** Practical experience running and comparing multiple local LLMs (llama, qwen, phi) offline, optimizing resource-constrained deployment.
- **RAG Architecture Design:** Built document parsing, token-aware overlap chunking, vector indexing, and cosine similarity lookup from scratch.
- **Evaluation & Quality Assurance:** Implemented the *LLM-as-a-Judge* pattern using structured JSON schemas to audit and monitor model hallucinations in data-critical pipelines.
- **Visual Analytics:** Designed interactive data dashboards using Streamlit and Plotly (including polar/radar development profiles, narrative timelines, and cross-model variance comparisons).
