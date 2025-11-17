# **NovaInsight: Autonomous Data Analysis Agent**

NovaInsight is a standalone command-line agent designed to perform **end‑to‑end Exploratory Data Analysis (EDA)** on tabular datasets. It acts as a semi‑autonomous analyst, ingesting raw data, generating statistical insights, creating visualizations, and assembling rich narrative reports powered by a Large Language Model (LLM).

NovaInsight is built around a core architectural principle — **the "Two Brains" model:**

1. **The Quantitative Brain:** Performs all deterministic statistical analysis using robust scientific libraries.
2. **The Language Brain:** Converts quantitative results into human‑readable summaries and actionable recommendations.

---

## 🚀 **Project Status: In Active Development**

The **Quantitative Brain** is largely functional. Current development focuses on visualization, natural‑language summarization, and report generation.

---

## ✅ **Current Capabilities (Quantitative Brain)**

NovaInsight already performs deep statistical and structural analysis through a modular pipeline:

### **1. Data Profiling**

Profiles each column with:

* Inferred data types
* Missing value statistics
* Distribution skewness
* Memory usage
* Duplicate row detection

### **2. Target Detection (Supervised Mode)**

Automatically identifies the most likely target variable using heuristics based on datatype, entropy, cardinality, and naming.

### **3. Statistical Analysis**

Includes:

* **Outlier Detection:** Modified Z‑Score (MAD)
* **Multicollinearity:** Variance Inflation Factor (VIF)
* **Correlation Analysis:**

  * Numerical–Numerical → Spearman’s ρ
  * Categorical–Categorical → Cramér’s V
  * Categorical–Numerical → Correlation Ratio (η)

### **4. Dimensionality Reduction**

Uncovers latent structure using:

* **PCA**
* **UMAP**
* **t‑SNE**

All methods automatically handle preprocessing (scaling, imputation).

---

## 🧠 **Coming Next (Language Brain & Visualization Layer)**

Upcoming modules include:

### **Visualizer**

Generates:

* Histograms
* Correlation heatmaps
* Scatter plots
* Dimensionality‑reduction embeddings

### **LLM Summarizer**

Produces:

* Narrative insights
* Explanations of trends and anomalies
* Highlighted risks and data gaps

### **Recommender**

Provides actionable next steps:

* Cleaning recommendations
* Feature engineering suggestions
* Model suitability assessments

### **Report Generator**

Outputs:

* JSON summaries
* PDF reports
* Jupyter Notebooks

---

## ⚡ **Getting Started**

### **1. Clone the repository**

```
git clone https://github.com/EyadMostafa/NovaInsight.git
cd NovaInsight
```

### **2. Create & activate a virtual environment**

**macOS/Linux:**

```
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**

```
python -m venv .venv
.venv/Scripts/activate
```

### **3. Install dependencies**

```
pip install -e .
```

Installs the project in editable mode and enables the `novainsight` CLI.

### **4. Run the pipeline**

Run only the available modules:

**Supervised example:**

```
novainsight analyze ./data/titanic.csv --modules profiler,target,stats,dim_reduction
```

**Unsupervised example:**

```
novainsight analyze ./data/titanic.csv --task unsupervised --modules profiler,stats,dim_reduction
```

---

## ⚙️ **Configuration System**

All settings are controlled via `config/config.yaml`.

### **Paths**

* Output directory
* Cache directory

### **Thresholds**

* Missing‑value limits
* VIF threshold
* Outlier sensitivity

### **Algorithm Hyperparameters**

* t‑SNE perplexity
* UMAP neighbors & metrics
* PCA component limits

Configuration loads automatically at runtime—no code changes required.

---

## 🖥️ Command-Line Reference

The `analyze` command accepts a dataset file path and several options that control the pipeline's behavior.

### **Usage**

```
novainsight analyze FILE_PATH [OPTIONS]
```

### **Argument**

* **FILE_PATH** — Path to a CSV or XLSX dataset.

### **Options (Flags)**

#### `--task <supervised|unsupervised>`

Controls the overall task type.

* **supervised (default):** Attempts to detect a target variable and structures analysis around prediction.
* **unsupervised:** Skips target detection; focuses on exploration, structure discovery, and clustering.

#### `--modules <module1,module2,...>`

Comma-separated list of specific modules to run (e.g., `profiler,stats`).
If omitted, NovaInsight attempts to run the full pipeline.

**Available Modules (including in-development modules):**
`profiler`, `target`, `stats`, `dim_reduction`, `viz`, `llm`, `recommendations`, `report`

#### `--target <column_name>`

Manually specifies the supervised target variable. If omitted, NovaInsight attempts automatic detection.

#### `--mode <fast|full>`

Controls dataset size processed.

* **full (default):** Analyzes the entire dataset.
* **fast:** Analyzes only the first N rows (N configured in `config.yaml`).

#### `--output-dir <path>`

Outputs all results to a `novainsight_reports` folder inside the specified directory.
Default: current working directory.

#### `--title "<Your Report Title>"`

Sets a custom title for generated reports.

#### `--force-rerun`

Ignores cached results and forces a full recomputation.

---

