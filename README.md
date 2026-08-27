<div align="center">

# 🔍 Sleuth

**Autonomous Exploratory Data Analysis — from raw file to narrative report in one command.**

[![Python](https://img.shields.io/badge/python-3.10%2B-4B8BBE?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](./LICENSE)
[![CI](https://github.com/EyadMostafa/sleuth/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/EyadMostafa/sleuth/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-81%25-22c55e)](./coverage.xml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](./Dockerfile)

[**View Sample Output →**](https://eyadmostafa.github.io/sleuth/)

</div>

---

Sleuth runs a deterministic statistical pipeline over any tabular dataset, then feeds the computed numbers to an LLM to write a plain-English narrative — executive summary, key findings, and prioritised recommendations. The LLM never sees raw data; it only interprets pre-computed statistics.

> **Analyst, not cleaner.** Sleuth diagnoses your data. It never mutates it.

---

## How it works

```
CSV / XLSX / Parquet / Feather
          │
          ▼
┌─────────────────────────────────────────────┐
│           Quantitative Brain                │  deterministic
│                                             │
│  Profiler → Target → Stats → DimReduce → Viz│
│  (pandas · scipy · sklearn · umap · seaborn)│
└────────────────────┬────────────────────────┘
                     │  structured JSON findings
                     ▼
┌─────────────────────────────────────────────┐
│             Language Brain                  │  probabilistic
│                                             │
│   Gemini · Claude · OpenAI · Ollama (local) │
└────────────────────┬────────────────────────┘
                     │
                     ▼
            Self-contained HTML report
```

---

## Features

| Module | What it does |
|---|---|
| **Data Profiler** | Type inference, missing values, cardinality, skewness, memory usage per column |
| **Target Detector** | Heuristic scoring to identify the supervised ML target and task type automatically |
| **Statistical Analyzer** | Outlier detection (MAD-based), VIF multicollinearity, Spearman / Cramér's V / Correlation Ratio matrices, data leakage flags, class imbalance |
| **Dimensionality Reducer** | PCA variance decomposition, t-SNE and UMAP 2-D embeddings |
| **Visualizer** | Univariate, bivariate, heatmap, and embedding scatter plot galleries |
| **LLM Summarizer** | Executive summary, per-finding narrative, ranked actionable recommendations |
| **Report Generator** | Self-contained HTML with base64-embedded images — share by email, no attachments |

**Pipeline features**

- ⚡ **Parallel execution** — wave-based `ThreadPoolExecutor`; independent modules run concurrently
- 💾 **Smart caching** — SHA-256-keyed cache resumes interrupted runs automatically
- 🔌 **Pluggable LLMs** — Gemini, Claude, OpenAI, or any local model via Ollama
- 📄 **Multiple formats** — CSV, XLSX, Parquet, Feather
- 🚀 **Fast mode** — samples N rows for a quick structural spot-check on large files

---

## Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white)
![Click](https://img.shields.io/badge/Click-000000?logo=python&logoColor=white)
![Jinja2](https://img.shields.io/badge/Jinja2-B41717?logo=jinja&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=githubactions&logoColor=white)

</div>

---

## Quick Start

### Docker (no install required)

```bash
docker pull eyadmostafa/sleuth:latest

docker run --rm \
  -v /path/to/your/data:/work \
  -e SLEUTH_LLM_PROVIDER=gemini \
  -e SLEUTH_LLM_API_KEY=your_key \
  eyadmostafa/sleuth:latest analyze /work/yourfile.csv
```

Output lands in `/path/to/your/data/sleuth_reports/`.

### pip

```bash
git clone https://github.com/EyadMostafa/sleuth.git
cd sleuth
pip install -e ".[gemini]"         # or [anthropic] / [openai] / [all-providers]

sleuth analyze yourfile.csv
```

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/EyadMostafa/sleuth.git
cd sleuth
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install the core package plus your preferred LLM provider:

```bash
pip install -e ".[gemini]"          # Google Gemini  (default)
pip install -e ".[anthropic]"       # Anthropic Claude
pip install -e ".[openai]"          # OpenAI
pip install -e ".[ollama]"          # Local model via Ollama (no API key needed)
pip install -e ".[all-providers]"   # All of the above
```

The quantitative pipeline (profiler, stats, plots, dim reduction) runs with no LLM and no API key.

---

## Configuration

Create a `.env` file at the project root:

```env
SLEUTH_LLM_PROVIDER=google              # google | anthropic | openai | ollama
SLEUTH_LLM_API_KEY=your_api_key_here    # not required for ollama
SLEUTH_LLM_MODEL_NAME=models/gemini-flash-latest   # optional override
```

For a local Ollama model (no API key, no data leaving your machine):

```env
SLEUTH_LLM_PROVIDER=ollama
SLEUTH_LLM_MODEL_NAME=llama3.2
SLEUTH_LLM_BASE_URL=http://localhost:11434   # optional — this is the default
```

Fine-grained tuning lives in `sleuth/config/config.yaml`:

```yaml
analysis:
  fast_mode_sample_rows: 50000

statistics:
  outlier_zscore_threshold: 3.0
  multicollinearity_vif_threshold: 10.0
  spearman_leakage_threshold: 0.98

dimensionality_reduction:
  tsne_perplexity: 30.0
  umap_n_neighbors: 15

visualization:
  dpi: 300
  theme: whitegrid
  color_palette: mako
```

---

## Usage

```
sleuth analyze FILE_PATH [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--task` | `supervised` | `supervised` — detect target, structure analysis around prediction. `unsupervised` — skip target detection, focus on clustering. |
| `--target COLUMN` | auto-detected | Pin the supervised target variable. |
| `--mode` | `full` | `full` — analyse entire file. `fast` — sample first N rows (configured in `config.yaml`). |
| `--modules LIST` | all | Comma-separated subset: `profiler,target,stats,dim_reduction,viz,llm,report` |
| `--output-dir PATH` | `.` | Root directory for the `sleuth_reports/` output folder. |
| `--title TEXT` | filename | Custom title for the report and output folder. |
| `--force-rerun` | off | Ignore cached results and recompute from scratch. |

### Examples

```bash
# Full pipeline — auto-detect target, LLM narrative, HTML report
sleuth analyze titanic.csv

# Unsupervised — structure discovery, no target
sleuth analyze customer_segments.csv --task unsupervised

# Quick structural check on a large file
sleuth analyze huge_dataset.csv --mode fast

# Pin target, custom output
sleuth analyze housing.csv --target SalePrice --title "Housing Q3" --output-dir ./reports

# Quantitative only — skip LLM and report generation
sleuth analyze data.csv --modules profiler,stats,dim_reduction,viz

# Specific target, specific modules
sleuth analyze data.csv --target churn --modules profiler,target,stats
```

---

## Output

Each run writes to `<output-dir>/sleuth_reports/<title>/`:

```
report_<title>.html       ← self-contained HTML dashboard (images base64-embedded)
report.json               ← full AnalysisReport as machine-readable JSON
plots/
  ├── univariate/         ← distribution plots per column
  ├── bivariate/          ← feature-vs-target scatter / box plots
  └── heatmaps/           ← Spearman, Cramér's V, Correlation Ratio
correlations/
  ├── numerical_corr.csv
  ├── categorical_corr.csv
  └── categorical_numerical_corr.csv
embeddings/
  ├── pca_embeddings.csv
  ├── tsne_embeddings.csv
  └── umap_embeddings.csv
```

A cache entry is also written to `~/.sleuth_cache/<sha256_of_file>/report.json` and reused automatically on subsequent runs against the same file.

---

## Docker

Build locally:

```bash
# Pull from Docker Hub
docker pull eyadmostafa/sleuth:latest

# Or build locally
docker build -t sleuth:dev .

# With all LLM providers
docker build --build-arg EXTRAS=all-providers -t sleuth:all .
```

Run with a bind-mounted data directory:

```bash
docker run --rm \
  -v /path/to/data:/work \
  -e SLEUTH_LLM_PROVIDER=gemini \
  -e SLEUTH_LLM_API_KEY=your_key \
  eyadmostafa/sleuth:latest analyze /work/yourfile.csv --mode fast
```

A `docker-compose.yml` is also provided for persistent volume and environment variable management:

```bash
# set SLEUTH_LLM_API_KEY in your environment or a .env file, then:
docker compose up
```

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for the branch strategy, commit conventions, and how to run the test suite before opening a PR.

---

## License

[MIT](./LICENSE) © Eyad Mostafa
