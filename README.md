**NovaInsight: Autonomous Data Analysis Agent \- Project Blueprint**

Version: 7.0 (Final Architecture)  
Status: Ready for Development

## **1\. Overview**

NovaInsight is a standalone, command-line agent designed to perform advanced exploratory data analysis (EDA) and insight generation on tabular datasets. It operates as a semi-autonomous analyst: it ingests raw datasets, detects structure, surfaces insights, visualizes relationships, and generates comprehensive, human-readable reports enriched by a Large Language Model (LLM). Its architecture is modular, configurable, and includes a persistent caching layer to support an efficient, iterative workflow.

### **1.1. Core Philosophy**

NovaInsight operates on two fundamental principles:

* **Analyst, Not Cleaner:** The agent's role is that of a skilled diagnostician. It identifies and quantifies data quality issues (nulls, outliers, skewness) but *never alters the original data*. It provides actionable recommendations for cleaning, but the final decision is always left to the user, who has the necessary domain context.  
* **The Two Brains:** The pipeline is architecturally separated into two distinct parts:  
  * **The Quantitative Brain (Modules 1-5):** Uses deterministic, mathematical libraries (Pandas, Scikit-learn) to calculate objective facts about the data.  
  * **The Language Brain (Modules 6-7):** Uses LLMs to take the factual output from the quantitative brain and translate it into a human-readable narrative and actionable advice.

## **2\. Core Capabilities & Features**

### **Analysis & Insight Generation**

* **Dual Analysis Modes:** Supports both *supervised* (target-based) and *unsupervised* (clustering and segmentation) analysis.  
* **Automated Data Profiling:** Ingests CSV and XLSX files and generates a detailed profile including data types, missing values, duplicate rows, skewness, and memory usage.  
* **Target Variable Detection:** In supervised mode, it automatically identifies or accepts a user-specified target variable. The current scope assumes a *single target variable*.  
* **Statistical Insights:** Performs outlier detection, class imbalance analysis, correlation analysis, and multicollinearity detection.  
* **Structure Discovery:** Uses dimensionality reduction (PCA, UMAP, t-SNE) and clustering to find hidden patterns.  
* **Automated Visualization:** Generates a gallery of plots, intelligently colored by the target variable or discovered cluster labels.  
* **LLM-Powered Summaries:** Synthesizes findings into a human-readable narrative.  
* **Actionable Recommendations:** Provides prioritized next steps for preprocessing, feature engineering, and modeling.

### **Key Architectural Features**

* **Selective Pipeline Execution:** The user can run specific parts of the pipeline for a faster, more focused analysis.  
* **Persistent Caching & Resumption:** Caches results to allow the user to resume a pipeline, running only the uncompleted steps.  
* **Configurable Behavior:** A central config.yaml file controls the agent's behavior.  
* **Graceful Degradation:** Resilient to non-critical failures, generating a partial report with warnings instead of crashing.

### **2.1. "Fast" Mode: The Structural Spot-Check**

A key feature of NovaInsight is the \--mode fast flag, which prioritizes speed for large datasets. It's crucial to understand its behavior:

* **"First N Rows" Sample:** In fast mode, the agent analyzes only the first N rows of the dataset (e.g., 5,000), as configured in config.yaml.  
* **Metadata Pre-Scan:** To provide accurate context, the agent first performs a quick, memory-efficient "pre-scan" of the full file to determine the *original total row count* without loading it into memory.  
* **A Spot-Check, Not an Estimate:** Because it analyzes the *first* rows and not a random sample, the results are a *structural spot-check*, not a statistically representative estimate. The final report will always include a prominent disclaimer explaining this limitation.  
* **Context-Aware Warnings:** The DataProfiler is "mode-aware" for specific warnings. For example, the "High-Dimensionality" warning correctly uses the original row count from the pre-scan to avoid false positives.

## **3\. Command-Line Interface (CLI)**

### **Main Command**

novainsight analyze \[OPTIONS\] FILE\_PATH

### **Options & Flags**

* \--task \<supervised|unsupervised\>: Specifies the analysis task type. **Default:** supervised.  
* \--output-dir \<path\>: Specifies the directory to save all output files. **Default:** Creates a new folder named \[filename\]\_novainsight\_report.  
* \--modules \<module1,...\>: A comma-separated list of the specific analysis modules to run. **Default:** Runs the entire pipeline.  
* \--target \<column\_name\>: Manually specifies the target variable (in supervised mode).  
* \--title "\<Your Title\>": Provides a custom title for the generated reports.  
* \--mode \<fast|full\>: Specifies the analysis mode. **Default:** full.  
* \--force-rerun: Ignores any cached results and runs a fresh analysis.

## **4\. Dynamic Pipeline Behavior**

The \--task flag fundamentally alters the behavior of the AnalysisPipeline.

| Module | Supervised Mode (Default) | Unsupervised Mode (--task unsupervised) |
| :---- | :---- | :---- |
| target\_detector | Runs full heuristics to find/validate a target. | Is completely skipped. |
| statistical\_analyzer | Includes class imbalance analysis. | Skips class imbalance analysis. |
| visualizer | Colors plots by the target variable's labels. | Performs clustering (e.g., K-Means) and colors plots by *discovered cluster labels*. |
| llm\_summarizer | Narrative focuses on "features that predict the target." | Narrative focuses on "describing the distinct groups found in the data." |
| recommender | Recommends classification/regression models. | Recommends clustering algorithms (e.g., K-Means, DBSCAN). |

## **5\. Module Responsibilities**

| Module File (novainsight/core/) | Conceptual Name | Primary Responsibility |
| :---- | :---- | :---- |
| orchestrator.py | Analysis Pipeline | The "brain" of the agent. Manages the dynamic pipeline, dependencies, and caching. |
| data\_profiler.py | Data Profiler | Loads, validates, and generates a detailed profile of the dataset. |
| target\_detector.py | Target Detector | Infers or validates the ML target variable. |
| statistical\_analyzer.py | Statistical Analyzer | Runs all quantitative tests (outliers, correlation, etc.). |
| dimensionality\_reducer.py | Structure Discoverer | Runs PCA, UMAP, and t-SNE to find and visualize latent data structures. |
| visualizer.py | Visualizer | Creates a consistent gallery of all plots and charts. |
| llm\_summarizer.py | LLM Summarizer | Uses an LLM to generate narrative explanations and summaries. |
| recommender.py | Recommender | Generates actionable next steps and warnings. |
| report\_generator.py | Report Generator | Creates the user-facing PDF, Jupyter Notebook, and JSON files. |
| cache\_manager.py | Cache Manager | Handles file hashing, saving/loading state, and managing the cache directory. |
| validators.py | Path Validators | Provides reusable functions for validating input/output paths. |

## **6\. Error Handling Strategy**

* **Fatal Errors (Halt Immediately):** Invalid inputs or critical failures will raise an exception and halt the pipeline with a clear error message.  
* **Module-Level Failures (Warn & Continue):** Non-critical failures (e.g., LLM API timeout) will be caught gracefully. The pipeline will continue, and the failure will be logged as a warning in the final report.  
* **Pre-flight Checks:** The orchestrator validates critical configurations before starting the analysis, failing fast if the setup is incorrect.

## **7\. LOC & Timeframe Estimation**

* **Total Estimated LOC:** 4,500 – 7,000 (including \~2,000 lines of comprehensive tests).  
* **Estimated Timeframe:** 6 – 9 Weeks.

## **8\. Technology Stack**

* **Core:** Python 3.10+  
* **Configuration:** Pydantic, PyYAML, python-dotenv  
* **Data Science:** Pandas, NumPy, Scikit-learn, Statsmodels  
* **Visualization:** Matplotlib, Seaborn  
* **CLI:** Click  
* **LLM:** Google Generative AI  
* **Reporting:** Jinja2, Nbformat, WeasyPrint  
* **Packaging & Testing:** pyproject.toml (with Setuptools), Pytest

## **9\. Future Upgrades & Vision**

This initial version of NovaInsight serves as a powerful foundation. Future versions could expand its capabilities to become a more comprehensive AI platform:

* **Deep Latent Structure Discovery:** Incorporate autoencoders and contrastive learning for more advanced feature embeddings.  
* **Fine-Tuned LLMs:** Utilize specialized tabular-to-text models for richer, more context-aware report generation.  
* **RAG-Enhanced EDA:** Augment reports with external knowledge by using Retrieval-Augmented Generation to provide context on domain-specific features.  
* **Vision Integration:** Add OCR modules to interpret user-uploaded plots and incorporate them into the analysis.  
* **Web Dashboard:** Create a web interface (using Streamlit or Gradio) for a better user experience.  
* **Plug-in System:** Develop a modular plug-in architecture that allows users to easily add their own custom analysis modules.
