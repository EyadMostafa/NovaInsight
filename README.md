# Sleuth — Autonomous Data Diagnostics

Sleuth is a CLI tool that performs end-to-end Exploratory Data Analysis on tabular data. It runs a deterministic statistical pipeline, then uses an LLM to synthesise the findings into a narrative HTML report — all from a single command.

[View Sample Outputs](https://eyadmostafa.github.io/Sleuth/)

---

## How it works

Sleuth separates analysis from interpretation into two distinct layers:

**Quantitative Brain** — deterministic. Pandas, SciPy, Statsmodels, and scikit-learn calculate objective facts: type inference, missing value rates, outlier scores, VIF, correlation matrices, dimensionality reduction embeddings. Nothing is guessed.

**Language Brain** — probabilistic. A pluggable LLM provider (Gemini, Claude, or OpenAI) receives the computed statistics and writes the executive summary, key findings, and prioritised recommendations. The LLM never sees raw data — only pre-computed numbers.

---

## Features

| | |
|---|---|
| Data Profiling | Type inference, missing value analysis, cardinality, skewness, memory usage |
| Target Detection | Heuristic scoring to automatically identify the supervised target and task type |
| Outlier Detection | Modified Z-score (MAD-based) per numeric column |
| Correlation Analysis | Three matrices — Spearman (Num↔Num), Cramér's V (Cat↔Cat), Correlation Ratio (Cat↔Num) |
| Multicollinearity | VIF per feature with configurable threshold |
| Data Leakage | Flags features whose correlation with the target exceeds a leakage threshold |
| Dimensionality Reduction | PCA, t-SNE, and UMAP embeddings with scatter plots |
| Visualisation | Univariate, bivariate, and heatmap galleries; plots coloured by target or cluster label |
| LLM Narrative | Executive summary, dataset overview, findings, warnings, and ranked recommendations |
| Fast Mode | Samples the first N rows for a structural spot-check on large files |
| Parallel Execution | Wave-based `ThreadPoolExecutor` — independent modules run concurrently |
| Smart Caching | Persists each run to `~/.sleuth_cache`; resumes interrupted pipelines automatically |
| Single-File Reports | Self-contained HTML with base64-embedded images — share via email with no attachments |

---

## Installation

**Prerequisites:** Python 3.10+

```bash
git clone https://github.com/EyadMostafa/sleuth.git
cd sleuth
```

**macOS / Linux**
```bash
python3 -m venv .venv && source .venv/bin/activate
```

**Windows**
```bash
python -m venv .venv && .venv\Scripts\activate
```

```bash
pip install -e .
```

**LLM providers** are optional extras — install only what you need:

```bash
pip install -e ".[gemini]"       # Google Gemini (default)
pip install -e ".[anthropic]"    # Anthropic Claude
pip install -e ".[openai]"       # OpenAI
pip install -e ".[all-providers]" # all three
```

All non-LLM modules (profiler, stats, dim reduction, viz) run without any API key.

---

## Configuration

Create a `.env` file at the project root:

```env
SLEUTH_LLM_API_KEY=your_api_key_here
SLEUTH_LLM_PROVIDER=google          # google | anthropic | openai
SLEUTH_LLM_MODEL_NAME=models/gemini-flash-latest   # optional override
```

Fine-grained tuning is done in `sleuth/config/config.yaml`:

```yaml
analysis:
  fast_mode_sample_rows: 50000     # rows sampled in --mode fast

statistics:
  outlier_zscore_threshold: 3.0
  multicollinearity_vif_threshold: 10.0
  spearman_correlation_threshold: 0.85
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

### Options

| Flag | Default | Description |
|---|---|---|
| `--task` | `supervised` | `supervised` — detect target and structure analysis around prediction. `unsupervised` — skip target detection, focus on clustering. |
| `--target COLUMN` | auto-detected | Manually pin the supervised target variable. |
| `--mode` | `full` | `full` — analyse entire file. `fast` — sample first N rows (see `config.yaml`). |
| `--modules LIST` | all | Comma-separated subset to run: `profiler,target,stats,dim_reduction,viz,llm` |
| `--output-dir PATH` | `.` | Root directory for `Sleuth_reports/` output folder. |
| `--title TEXT` | filename | Custom title for the report and output folder. |
| `--force-rerun` | off | Ignore cached results and recompute everything. |

### Examples

```bash
# Supervised run — auto-detect target, full pipeline
sleuth analyze titanic.csv

# Unsupervised — clustering and structure discovery
sleuth analyze customer_data.csv --task unsupervised

# Fast structural check on a large file
sleuth analyze huge_dataset.csv --mode fast

# Pin target, set report title, write to a specific directory
sleuth analyze housing.csv --target SalePrice --title "Housing Q3" --output-dir ./reports

# Run only profiling and statistics (skip plots and LLM)
sleuth analyze data.csv --modules profiler,stats
```

---

## Output

Each run creates a folder at `sleuth_reports/<title>/`:

```
report_<title>.html    # self-contained HTML dashboard (base64 images embedded)
report.json            # full AnalysisReport as machine-readable JSON
plots/                 # high-res PNGs — univariate, bivariate, heatmaps, embeddings
correlations/          # Spearman, Cramér's V, Correlation Ratio matrices (.csv)
embeddings/            # PCA, t-SNE, UMAP coordinates (.csv)
```

The cache lives separately at `~/.sleuth_cache/<sha256_of_file>/report.json` and is reused automatically on subsequent runs against the same file.
