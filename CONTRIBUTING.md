# Contributing to Sleuth

Thank you for your interest in contributing. This document covers everything you need to get started.

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, production-ready. |
| `dev` | Active development. All PRs target this branch. |
| `feat/<name>` | New features, branched from `dev`. |
| `fix/<name>` | Bug fixes, branched from `dev`. |

Always branch from `dev`, not `main`.

---

## Commit Convention

One line only — no body, no bullet points:

```
feat: <short imperative description>
fix: <short imperative description>
refactor: <short imperative description>
chore: <short imperative description>
test: <short imperative description>
docs: <short imperative description>
```

---

## Development Setup

```bash
git clone https://github.com/EyadMostafa/sleuth.git
cd sleuth
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -e ".[test,all-providers,lint]"
```

---

## Running Tests

```bash
# Full suite
pytest

# Skip slow integration tests
pytest -m "not slow"

# Single file
pytest tests/unit/modules/test_data_profiler.py -v

# With coverage report
pytest --cov=sleuth --cov-report=term-missing
```

The canonical fixture dataset is `tests/fixtures/student_performance_small.csv`. Use it for any new integration tests.

For a quick end-to-end sanity check without touching the LLM:

```bash
sleuth analyze tests/fixtures/student_performance_small.csv \
  --modules profiler,stats \
  --mode fast \
  --force-rerun
```

---

## Linting

```bash
ruff check sleuth tests       # lint
ruff format sleuth tests      # format
ruff check --fix sleuth tests # auto-fix safe violations
```

CI enforces both `ruff check` and `ruff format --check`. Fix any violations before opening a PR.

---

## Adding a New Module

1. Create `sleuth/modules/my_module.py` extending `BaseModule` and decorate it with `@register_module`.
2. Add a value to the `Operator` enum in `sleuth/schemas/analysis_report.py`.
3. Add output schema(s) to `analysis_report.py` and a corresponding field on `AnalysisReport`.
4. Import the new module in `sleuth/modules/__init__.py` to trigger registration.
5. Add unit and integration tests under `tests/`.

The orchestrator derives execution order automatically via topological sort — no changes to `orchestrator.py` needed.

See [CLAUDE.md](./CLAUDE.md) for full architecture conventions.

---

## Adding a New LLM Provider

1. Create `sleuth/llm/providers/my_provider.py` subclassing `LLMProvider`.
2. Add a branch in `sleuth/llm/factory.py` matching the provider name string.
3. Add the provider name to `_SUPPORTED_PROVIDERS` in `factory.py`.
4. Add an optional dependency group in `pyproject.toml`.

---

## Pull Request Checklist

- [ ] Branched from `dev`
- [ ] All tests pass (`pytest -m "not slow"`)
- [ ] `ruff check` and `ruff format --check` pass
- [ ] New behaviour is covered by tests
- [ ] No new `print()` calls — use `get_logger()`
- [ ] No plain `ValueError` / `RuntimeError` — use the typed exception hierarchy in `sleuth/exceptions.py`
