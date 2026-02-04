# **NovaInsight: Autonomous Data Analysis Agent**

**NovaInsight** is a sophisticated, standalone CLI agent that performs end-to-end **Exploratory Data Analysis (EDA)**. It acts as a semi-autonomous data scientist, ingesting raw tabular data, performing rigorous statistical diagnostics, and generating rich, narrative-driven HTML reports powered by Large Language Models (LLM).

---

## **The "Dual Brain" Architecture**

NovaInsight is built on a unique architectural separation of concerns:

1. **The Quantitative Brain (Deterministic):**  
   * Uses robust libraries (Pandas, Scikit-learn, SciPy, Statsmodels) to calculate objective facts.  
   * Performs advanced tests: Modified Z-Score outlier detection, Multicollinearity (VIF), and polymorphic correlations (Spearman, Cramér's V, Correlation Ratio).  
   * **Zero Hallucination:** All statistics are mathematically calculated, not guessed.

2. **The Language Brain (Probabilistic):**  
   * Uses Google's Gemini to synthesize the quantitative findings.  
   * Translates complex stats into executive summaries, prioritized recommendations, and warnings.  
   * **Context Aware:** Understands if the task is supervised or unsupervised and tailors the narrative accordingly.

---

## **Key Features**

* **Comprehensive Profiling:** Automated type inference, missing value analysis, and memory usage profiling.  
* **Autonomous Target Detection:** Automatically identifies the prediction target and task type (Classification vs. Regression) using heuristic scoring.  
* **Deep Statistical Insights:**  
  * **Data Leakage Detection:** Identifies features that are "too good to be true."  
  * **Manifold Learning:** Visualizes high-dimensional data using **PCA**, **t-SNE**, and **UMAP**.  
  * **Correlations:** Generates Num-Num, Cat-Cat, and Cat-Num heatmaps.  
* **Fast Mode:** Intelligent pre-scanning and sampling for massive datasets to get structural insights in seconds.  
* **Single-File Reports:** Generates a self-contained, professional HTML report with embedded interactive plots and base64 images—easy to share via email.  
* **Smart Caching:** Resumes interrupted pipelines without re-calculating expensive steps.

---

## **🚀 Getting Started**

### **Prerequisites**

* Python 3.10+  
* A Google Cloud API Key (for Gemini LLM features)

---

### **Installation**

1. **Clone the repository:**
```bash
git clone https://github.com/EyadMostafa/NovaInsight.git
cd NovaInsight
```

2. **Set up the environment:**

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -e .
```

4. **Configure API Key:**  
Create a `.env` file in the root directory:
```env
NOVA_INSIGHT_LLM_API_KEY=your_google_gemini_api_key_here
```

---
## 🖥️ Command-Line Reference

The `analyze` command accepts a dataset file path and several options that control the pipeline's behavior.

---

### **Usage**

```bash
novainsight analyze FILE_PATH [OPTIONS]
```

---

### **Argument**

* **FILE_PATH** — Path to a CSV or XLSX dataset.

---

### **Options (Flags)**

#### `--task <supervised|unsupervised>`

Controls the overall task type.

* **supervised (default):** Attempts to detect a target variable and structures analysis around prediction.  
* **unsupervised:** Skips target detection; focuses on exploration, structure discovery, and clustering.

---

#### `--modules <module1,module2,...>`

Comma-separated list of specific modules to run (e.g., `profiler,stats`).  
If omitted, NovaInsight attempts to run the full pipeline.

**Available Modules:**
```
profiler, target, stats, dim_reduction, viz, llm, recommendations
```

---

#### `--target <column_name>`

Manually specifies the supervised target variable. If omitted, NovaInsight attempts automatic detection.

---

#### `--mode <fast|full>`

Controls dataset size processed.

* **full (default):** Analyzes the entire dataset.  
* **fast:** Analyzes only the first N rows (N configured in `config.yaml`).

---

#### `--output-dir <path>`

Outputs all results to a `novainsight_reports` folder inside the specified directory.  
Default: current working directory.

---

#### `--title "<Your Report Title>"`

Sets a custom title for generated reports.

---

#### `--force-rerun`

Ignores cached results and forces a full recomputation.



## **🖥️ Usage**



### **Basic Run (Supervised)**

Attempts to detect a target and runs the full pipeline.

```bash
novainsight analyze titanic.csv
```

---

### **Unsupervised Analysis**

Focuses on clustering and structure discovery instead of prediction.

```bash
novainsight analyze customer_segments.csv --task unsupervised
```

---

### **Fast Mode (Structural Spot-Check)**

Analyzes a sample (default 5,000 rows) of a large file to check data quality quickly.

```bash
novainsight analyze huge_dataset.csv --mode fast
```

---

### **Specific Target & Title**

```bash
novainsight analyze housing.csv --target SalePrice --title "Housing Market Q3 Analysis"
```

---

## **🛠️ Configuration**

NovaInsight is highly configurable via `config/config.yaml`. You can tune:

* **Statistical Thresholds:** Correlation strength, Outlier sensitivity (Z-Score), VIF limits.  
* **Dimensionality Reduction:** Perplexity for t-SNE, Neighbors for UMAP.  
* **Visualization:** Color palettes (Seaborn), DPI, Theme.  
* **System:** Cache directories, Logging levels.

*See `config/config.yaml` for the full list of options.*

---

## **📂 Output Artifacts**

For every run, NovaInsight creates a dedicated folder in `novainsight_reports/`.

* **report_title.html** — The main artifact. A polished, interactive dashboard.  
* **report.json** — The raw machine-readable data of the entire analysis.  
* **plots/** — High-resolution PNGs of all generated charts.  
* **correlations/** — CSVs of the correlation matrices.  
* **embeddings/** — CSVs of the PCA/t-SNE coordinates.

[View Sample Outputs](https://www.google.com/search?q=docs/EXAMPLES.md)

---

## **🗺️ Roadmap**

* [x] Quantitative Pipeline (Profiling, Stats, Target Detection)  
* [x] Visualization Engine  
* [x] LLM Integration (Gemini)  
* [x] HTML Report Generation  
* [ ] Comparison Mode (Compare two datasets)  
* [ ] Interactive Streamlit Dashboard

---

## **📄 License**

Distributed under the MIT License. See LICENSE for more information.

---

