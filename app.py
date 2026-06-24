import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
import os
import time
from pypdf import PdfReader

# Import backend modules
from src.pdf_processor import extract_raw_text, clean_text, create_chunks, extract_chapter_texts, scan_themes_by_page
from src.vector_store import LocalVectorStore
from src.llm_client import (
    count_thematic_frequencies, 
    extract_key_results, 
    summarize_chapters, 
    extract_strengths_challenges, 
    extract_numerical_indicators, 
    extract_demographic_trends
)
from src.evaluator import run_model_benchmark, evaluate_extracted_data


# Page config
st.set_page_config(
    page_title="UN HDR Intelligence & Evaluation Dashboard",
    page_icon="🇺🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling (UN Blue palette)
st.markdown("""
<style>
    .reportview-container {
        background: #f8f9fa;
    }
    .main-title {
        color: #1d70b8;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eef0f2;
        margin-bottom: 1rem;
    }
    .strength-item {
        background-color: #e6f4ea;
        color: #137333;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    .challenge-item {
        background-color: #fce8e6;
        color: #c5221f;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Helper to save uploaded file
def save_uploaded_file(uploaded_file):
    temp_dir = "scratch"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, "temp_uploaded_report.pdf")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Load preprocessed cached results
@st.cache_data
def load_cached_results():
    results_dir = "results"
    summary_path = os.path.join(results_dir, "summary.json")
    
    if not os.path.exists(summary_path):
        return None
        
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
        
    models = summary.get("models_tested", [])
    benchmarks = {}
    
    for model in models:
        model_safe_name = model.replace(":", "_").replace(".", "_")
        model_result_path = os.path.join(results_dir, f"{model_safe_name}_results.json")
        if os.path.exists(model_result_path):
            with open(model_result_path, 'r', encoding='utf-8') as f:
                benchmarks[model] = json.load(f)
                
    return {"summary": summary, "benchmarks": benchmarks}

# Load preprocessed cached results directly
data = load_cached_results()

# App Layout
st.markdown('<h1 class="main-title">AI-Powered UN Development Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Static cognitive analysis dashboard & local LLM thematic insights</p>', unsafe_allow_html=True)

if data is not None:
    summary = data["summary"]
    benchmarks = data["benchmarks"]
    
    # Sleek horizontal context banner at the top
    st.markdown(f"""
    <div style="background-color: #f8f9fa; border-left: 4px solid #1d70b8; padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 1.5rem; font-size: 1rem; color: #333;">
      🌐 <b>Analysed Report:</b> {summary.get('country')} ({summary.get('year')}) — "{summary.get('report_title')}"
    </div>
    """, unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/2/2f/Flag_of_the_United_Nations.svg", width=150)

# Pipeline hyperparameters
st.sidebar.markdown("### RAG Hyperparameters")
chunk_size = st.sidebar.slider("Chunk Size (characters)", 400, 1500, 800)
overlap = st.sidebar.slider("Overlap (characters)", 50, 300, 150)
top_k = st.sidebar.slider("Retrieval Depth (K)", 1, 10, 3)

models_list = ["llama3.2", "qwen2.5:3b", "phi3:mini"]
selected_models = st.sidebar.multiselect(
    "Models to Benchmark:",
    models_list,
    default=models_list
)

if data is not None:
    st.sidebar.success("Successfully loaded preprocessed country report!")
    
    # ------------------ MAIN TABS ------------------
    tab_summary, tab_themes, tab_indicators, tab_benchmarks = st.tabs([
        "📋 Summary & Chapters", 
        "📊 Thematic & Sentiment Analysis", 
        "📈 Development Indicators", 
        "⚖️ Model Benchmarking (Judge)"
    ])
    
    # --- TAB 1: EXECUTIVE SUMMARY & CHAPTERS ---
    with tab_summary:
        st.subheader("Executive Development Summary")
        
        # Display the output from the primary model (default to llama3.2 if available, else first)
        primary_model = "llama3.2" if "llama3.2" in benchmarks else list(benchmarks.keys())[0]
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"#### Key Results (Model: `{primary_model}`)")
            key_res_text = benchmarks[primary_model].get("key_results", "No summaries available.")
            st.write(key_res_text)
            
        with col2:
            st.markdown("#### Document Metadata")
            st.markdown(f"""
            - **Report Name:** `{summary.get('report_title')}`
            - **Target Country:** `{summary.get('country')}`
            - **Publication / Study Year:** `{summary.get('year')}`
            - **Local Embeddings Model:** `nomic-embed-text`
            - **Inference Models Tested:** `{', '.join(summary.get('models_tested', []))}`
            """)
            
        st.markdown("---")
        st.subheader("Chapter-by-Chapter Summaries (<100 Words Each)")
        
        chapter_sums = benchmarks[primary_model].get("chapter_summaries", {})
        if chapter_sums:
            cols = st.columns(2)
            for idx, (ch_num, text_sum) in enumerate(sorted(chapter_sums.items(), key=lambda x: int(x[0]))):
                col_idx = idx % 2
                with cols[col_idx]:
                    with st.expander(f"📖 Chapter {ch_num} Summary", expanded=(idx < 2)):
                        st.write(text_sum)
        else:
            st.write("No chapter summaries extracted.")

    # --- TAB 2: THEMATIC & SENTIMENT ANALYSIS ---
    with tab_themes:
        st.subheader("Thematic Frequency & Sentiment Distribution")
        
        col_theme1, col_theme2 = st.columns([1, 1])
        
        with col_theme1:
            st.markdown("#### Theme Occurrence Counts")
            counts = summary.get("theme_counts", {})
            df_counts = pd.DataFrame(list(counts.items()), columns=["Theme", "Occurrences"]).sort_values(by="Occurrences", ascending=False)
            df_counts['Theme'] = df_counts['Theme'].str.capitalize()
            
            fig = px.bar(
                df_counts, 
                x="Occurrences", 
                y="Theme", 
                orientation="h",
                color="Occurrences",
                color_continuous_scale=px.colors.sequential.Blues,
                title="Thematic Word Mentions in Text"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col_theme2:
            st.markdown("#### Theme Distribution %")
            fig_pie = px.pie(
                df_counts, 
                values="Occurrences", 
                names="Theme", 
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="Theme Percentage Breakdown"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        st.markdown("---")
        st.subheader("Qualitative Development Assessment")
        
        # Qualitative card comparisons (Strengths vs Challenges)
        qual_data = benchmarks[primary_model].get("strengths_challenges", {"strengths": [], "challenges": []})
        
        col_s, col_c = st.columns(2)
        with col_s:
            st.markdown("### 🌟 Key Strengths & Assets")
            for item in qual_data.get("strengths", []):
                st.markdown(f'<div class="strength-item">✓ {item}</div>', unsafe_allow_html=True)
                
        with col_c:
            st.markdown("### ⚠️ Primary Challenges & Risks")
            for item in qual_data.get("challenges", []):
                st.markdown(f'<div class="challenge-item">⚠ {item}</div>', unsafe_allow_html=True)
                
        # Add Qualitative Extraction Depth chart
        st.markdown("---")
        st.markdown("#### Qualitative Extraction Depth Analysis")
        qual_counts = []
        for model in summary.get("models_tested", []):
            m_qual = benchmarks[model].get("strengths_challenges", {"strengths": [], "challenges": []})
            qual_counts.append({
                "Model": model,
                "Strengths Extracted": len(m_qual.get("strengths", [])),
                "Challenges Extracted": len(m_qual.get("challenges", []))
            })
        df_qual_counts = pd.DataFrame(qual_counts)
        fig_qual_depth = go.Figure(data=[
            go.Bar(name='Strengths Extracted', x=df_qual_counts['Model'], y=df_qual_counts['Strengths Extracted'], marker_color='#e6f4ea', marker_line_color='#137333', marker_line_width=1.5),
            go.Bar(name='Challenges Extracted', x=df_qual_counts['Model'], y=df_qual_counts['Challenges Extracted'], marker_color='#fce8e6', marker_line_color='#c5221f', marker_line_width=1.5)
        ])
        fig_qual_depth.update_layout(
            barmode='group',
            title="Qualitative Extraction Depth: Unique Strengths vs. Challenges Extracted",
            xaxis_title="Local Inference Architecture",
            yaxis_title="Unique Insights Count"
        )
        st.plotly_chart(fig_qual_depth, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📖 Document Thematic Narrative Mapping")
        st.markdown("*Maps the distribution of core themes across PDF page coordinates to trace structural narrative focus.*")
        
        theme_by_page = summary.get("theme_by_page", [])
        if theme_by_page:
            df_trends = pd.DataFrame(theme_by_page)
            theme_page_counts = df_trends.groupby("theme").size().reset_index(name='Counts').sort_values(by="Counts", ascending=False)
            theme_page_counts['Theme'] = theme_page_counts['theme'].str.capitalize()
            
            col_map1, col_map2 = st.columns([1, 2])
            with col_map1:
                st.markdown("#### Cumulative Page Prevalence")
                fig_map_a = px.bar(
                    theme_page_counts,
                    x="Counts",
                    y="Theme",
                    orientation="h",
                    color="Theme",
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    title="Number of Pages Flagged by Theme"
                )
                fig_map_a.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_map_a, use_container_width=True)
                
            with col_map2:
                st.markdown("#### Thematic Timeline Scatter Location Map")
                df_present = df_trends.copy()
                df_present['Theme'] = df_present['theme'].str.capitalize()
                df_present = df_present.sort_values(by="Theme")
                
                fig_map_b = px.scatter(
                    df_present,
                    x="page",
                    y="Theme",
                    color="Theme",
                    symbol="Theme",
                    hover_data=["page"],
                    title="Narrative Timelines (X = PDF Page, Y = Theme)"
                )
                fig_map_b.update_traces(marker=dict(size=12, opacity=0.85))
                fig_map_b.update_layout(
                    xaxis_title="PDF Page Position",
                    yaxis_title="",
                    showlegend=False
                )
                st.plotly_chart(fig_map_b, use_container_width=True)
        else:
            st.info("Page-by-page thematic scanning data is not available. Please run live PDF ingestion to see narrative timelines.")

    # --- TAB 3: DEVELOPMENT INDICATORS & TRENDS ---
    with tab_indicators:
        st.subheader("Core Development Indicators Extract")
        
        # Compile side-by-side indicator comparison
        ind_compare = []
        for model in summary.get("models_tested", []):
            ind_data = summary["benchmark_summary"][model].get("indicators", {})
            ind_compare.append({
                "Model": model,
                "HDI Value": ind_data.get("hdi_value"),
                "HDI Rank": ind_data.get("hdi_rank"),
                "Life Expectancy (yrs)": ind_data.get("life_expectancy"),
                "Expected Schooling (yrs)": ind_data.get("expected_schooling"),
                "Mean Schooling (yrs)": ind_data.get("mean_schooling"),
                "GNI per capita ($)": ind_data.get("gni_per_capita"),
                "Population (M)": ind_data.get("population")
            })
            
        df_ind = pd.DataFrame(ind_compare)
        st.dataframe(df_ind, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        col_plot1, col_plot2 = st.columns([1, 1])
        
        with col_plot1:
            st.markdown("#### Advanced Visualisation: normalized Radar Chart")
            # Normalize indicators for the radar chart comparison
            prim_ind = summary["benchmark_summary"][primary_model].get("indicators", {})
            
            hdi_val = prim_ind.get("hdi_value") or 0.7
            life_exp = (prim_ind.get("life_expectancy") or 70) / 95.0
            exp_sch = (prim_ind.get("expected_schooling") or 12) / 20.0
            mean_sch = (prim_ind.get("mean_schooling") or 8) / 15.0
            gni_val = min((prim_ind.get("gni_per_capita") or 8000) / 50000.0, 1.0)
            
            categories = ['HDI Index', 'Life Expectancy', 'Expected Schooling', 'Mean Schooling', 'GNI per Capita']
            values = [hdi_val, life_exp, exp_sch, mean_sch, gni_val]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=summary.get("country"),
                line_color="#1d70b8"
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                title=f"Normalized Development Profile: {summary.get('country')}"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col_plot2:
            st.markdown("#### Demographic Trends Over Time")
            trend_data = benchmarks[primary_model].get("demographic_trends", {}).get("trends", [])
            if trend_data:
                df_trend = pd.DataFrame(trend_data)
                fig_trend = px.line(
                    df_trend, 
                    x="year", 
                    y="value", 
                    color="indicator_name", 
                    markers=True,
                    title="Extracted Longitudinal Indicators"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.write("No historical timeline tables detected in this report.")
                
        st.markdown("---")
        st.subheader("📊 Cross-Model Indicator Variance Analysis")
        st.markdown("*Compare the numeric values extracted by different models to audit LLM extraction consistency.*")
        
        # Indicator selectbox for comparison
        indicators_to_compare = [
            "HDI Value", "Life Expectancy (yrs)", 
            "Expected Schooling (yrs)", "Mean Schooling (yrs)", 
            "GNI per capita ($)", "Population (M)"
        ]
        selected_ind = st.selectbox(
            "Select Development Indicator to Compare across Models:",
            indicators_to_compare
        )
        
        # Filter dataframe for selected indicator
        df_compare = df_ind[["Model", selected_ind]].copy()
        # Force selected indicator to numeric for plotting (handling None values)
        df_compare[selected_ind] = pd.to_numeric(df_compare[selected_ind], errors='coerce')
        
        fig_variance = px.bar(
            df_compare,
            x="Model",
            y=selected_ind,
            color="Model",
            color_discrete_sequence=px.colors.qualitative.Plotly,
            title=f"Cross-Model Comparison: {selected_ind}"
        )
        fig_variance.update_layout(yaxis_title=selected_ind, showlegend=False)
        st.plotly_chart(fig_variance, use_container_width=True)

    # --- TAB 4: MODEL BENCHMARKING (JUDGE) ---
    with tab_benchmarks:
        st.subheader("Cross-LLM Behavior Analysis")
        
        bench_summary = summary.get("benchmark_summary", {})
        col_j1, col_j2 = st.columns(2)
        
        with col_j1:
            st.markdown("#### LLM-as-a-Judge Quality Scores")
            eval_data = []
            for model, metrics in bench_summary.items():
                eval_data.append({
                    "Model": model,
                    "Faithfulness (1-5)": metrics.get("faithfulness"),
                    "Relevance (1-5)": metrics.get("relevance")
                })
            df_eval = pd.DataFrame(eval_data)
            
            fig_eval = go.Figure(data=[
                go.Bar(name='Faithfulness', x=df_eval['Model'], y=df_eval['Faithfulness (1-5)'], marker_color='#137333'),
                go.Bar(name='Relevance', x=df_eval['Model'], y=df_eval['Relevance (1-5)'], marker_color='#1d70b8')
            ])
            fig_eval.update_layout(barmode='group', yaxis_range=[0,5], title="Quality Ratings (Higher is Better)")
            st.plotly_chart(fig_eval, use_container_width=True)
            
        with col_j2:
            st.markdown("#### Performance Efficiency Trade-offs")
            perf_data = []
            for model in summary.get("models_tested", []):
                m_data = benchmarks[model]
                perf_data.append({
                    "Model": model,
                    "Inference Latency (sec)": m_data["metrics"].get("total_latency"),
                    "Response Verbosity (words)": m_data["metrics"].get("verbosity_word_count")
                })
            df_perf = pd.DataFrame(perf_data)
            
            fig_perf = px.scatter(
                df_perf,
                x="Inference Latency (sec)",
                y="Response Verbosity (words)",
                color="Model",
                size=[30, 30, 30][:len(df_perf)],
                title="Latency vs. Verbosity Trade-off"
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            
        st.markdown("---")
        col_j3, col_j4 = st.columns(2)
        
        with col_j3:
            st.markdown("#### Execution Latency Breakdown by Extraction Task")
            latency_data = []
            sub_tasks = [
                ("Key Results", "key_results_latency"),
                ("Chapter Summaries", "chapter_summaries_latency"),
                ("Strengths & Challenges", "strengths_challenges_latency"),
                ("Numerical Indicators", "numerical_indicators_latency"),
                ("Demographic Trends", "demographic_trends_latency")
            ]
            for model in summary.get("models_tested", []):
                m_metrics = benchmarks[model].get("metrics", {})
                for task_label, task_key in sub_tasks:
                    latency_data.append({
                        "Model": model,
                        "Extraction Task": task_label,
                        "Latency (sec)": m_metrics.get(task_key) or 0
                    })
            df_latency_breakdown = pd.DataFrame(latency_data)
            
            fig_lat_breakdown = px.bar(
                df_latency_breakdown,
                x="Model",
                y="Latency (sec)",
                color="Extraction Task",
                title="Cumulative Execution Time by Modular Task (lower is better)",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_lat_breakdown.update_layout(xaxis_title="Local Inference Architecture", yaxis_title="Cumulative Time (Seconds)")
            st.plotly_chart(fig_lat_breakdown, use_container_width=True)
            
        with col_j4:
            st.markdown("#### Response Verbosity Comparison")
            verb_data = []
            for model in summary.get("models_tested", []):
                m_metrics = benchmarks[model].get("metrics", {})
                verb_data.append({
                    "Model": model,
                    "Word Count": m_metrics.get("verbosity_word_count") or 0
                })
            df_verb = pd.DataFrame(verb_data)
            
            fig_verb = px.bar(
                df_verb,
                x="Model",
                y="Word Count",
                color="Model",
                title="Combined Output Length (Total Verbosity in Words)",
                color_discrete_sequence=['#c5221f', '#1d70b8', '#137333']
            )
            fig_verb.update_layout(xaxis_title="Local Inference Architecture", yaxis_title="Combined Output Word Count", showlegend=False)
            st.plotly_chart(fig_verb, use_container_width=True)
            
        st.markdown("---")
        st.subheader("Judge Detailed Critiques & Justifications")
        for model in summary.get("models_tested", []):
            m_eval = bench_summary[model]
            judge_model_name = "qwen2.5:3b" if model == "llama3.2" else "llama3.2"
            with st.expander(f"⚖️ Model: {model} — Faithfulness: {m_eval.get('faithfulness')}/5 | Relevance: {m_eval.get('relevance')}/5"):
                st.markdown(f"**Evaluator Judge Model:** `{judge_model_name}`")
                st.markdown(f"**Judge Justification:**")
                st.write(m_eval.get("justification"))
else:
    st.info("Configure ingestion options in the sidebar and trigger 'Run Pipeline Ingestion' to start the analysis.")
