# DoWhy Agent Guidelines

This file provides guidance for automated AI agents working with the DoWhy codebase. It covers the repository structure, architectural goals, and the checks that must pass before any PR can be merged.

---

## Project Overview

DoWhy is a Python library for causal inference. It supports two complementary frameworks:

1. **Potential outcomes** (classic treatment/control)
2. **Graphical causal models (GCM)** (DAG-based)

Both frameworks are unified under a four-step workflow: **Model → Identify → Estimate → Refute**.

---

## Repository Layout

```
dowhy/                  # Main package
  __init__.py           # Public exports: CausalModel, identify_effect_*, EstimandType
  causal_model.py       # CausalModel: orchestrates the 4-step workflow
  causal_estimator.py   # Base class for all estimators
  causal_graph.py       # Graph operations and DAG handling
  causal_refuter.py     # Base class for refuters
  causal_estimators/    # 15+ estimation method implementations
  causal_identifier/    # Identification algorithms (do-calculus)
  causal_refuters/      # Robustness / sensitivity tests
  gcm/                  # Graphical Causal Models module (root cause analysis, anomaly detection, etc.)
  api/                  # Public API surface
  do_samplers/          # Sampling from interventional distributions
  graph_learners/       # Structure learning algorithms
  data_transformers/    # Data preprocessing
  timeseries/           # Time series causal inference
  causal_prediction/    # ML-based prediction (neural networks, dataloaders)
  utils/                # Shared utilities
  datasets.py           # Built-in datasets for testing/demos
tests/                  # Test suite (mirrors dowhy/ structure)
docs/                   # Sphinx documentation
  source/
    contributing/       # Developer & contribution guides
    example_notebooks/  # Jupyter notebook examples
    user_guide/         # End-user documentation
pyproject.toml          # Build config, dependencies, tool config (Poetry)
.flake8                 # flake8 config
```

---

## Architecture & Design Goals

- **Four-step causal workflow**: Every contribution should fit cleanly into Model, Identify, Estimate, or Refute. Do not conflate steps.
- **Backwards compatibility**: The `CausalModel` public API is stable. New methods should extend, not replace, existing interfaces.
- **Two frameworks, one API**: Changes that touch one framework (potential outcomes vs. GCM) should not break the other.
- **Extensibility**: New estimators inherit from `CausalEstimator`; new refuters from `CausalRefuter`. Follow existing class hierarchies.
- **Minimal required dependencies**: Core functionality should remain available without optional extras (`plotting`, `pygraphviz`, `econml`). Guard optional imports with try/except and raise informative errors.
- **Python 3.9–3.13**: All code must be compatible with Python 3.9 through 3.13.

---

## Environment Setup

```bash
pip install --upgrade pip
poetry install -E "plotting"       # Standard dev install
```

Optional extras:

```bash
poetry install -E "pygraphviz"     # Graph visualization via graphviz
poetry install -E "econml"         # EconML CATE estimators
```

On Linux, `pygraphviz` may require:

```bash
sudo apt install graphviz libgraphviz-dev graphviz-dev pkg-config
pip install --global-option=build_ext \
  --global-option="-I/usr/local/include/graphviz/" \
  --global-option="-L/usr/local/lib/graphviz" pygraphviz
```

---

## Linting & Formatting

**All three checks must pass before a PR can merge.**

| Tool | Purpose | Config |
|------|---------|--------|
| `black` | Code formatting | Line length 120, targets py39–py313 |
| `isort` | Import sorting | Profile `black`, line length 120, multi-line output 3 |
| `flake8` | Linting | Max line 127, max complexity 10, hard errors on E9/F63/F7/F82 |

### Commands

```bash
# Auto-fix formatting
poetry run poe format

# Check only (what CI runs)
poetry run poe format_check
poetry run poe lint
```

### Rules to follow

- Maximum line length: **120 characters** (black) / **127 characters** (flake8 hard limit).
- Cyclomatic complexity per function: **≤ 10**.
- Import order: standard library → third-party → local, each group sorted. `isort` enforces this automatically.
- Do not add `# noqa` silences unless genuinely unavoidable; document why if you do.

---

## Testing

```bash
# Standard test run (excludes `advanced` and `econml` markers)
poetry run poe test

# Skip notebook execution (faster)
poetry run poe test_no_notebooks

# Run only EconML tests
poetry run poe test_econml

# Run the full test suite (including `advanced`)
poetry run poe test_advanced

# Run a specific subdirectory
poetry run pytest -v tests/causal_refuters

# Run only tests you marked `focused` (debug helper)
poetry run poe test_focused
```

### Test markers

| Marker | Meaning |
|--------|---------|
| `advanced` | Skipped by default; run only on package-level updates |
| `notebook` | Executes Jupyter notebooks; slow |
| `econml` | Requires `econml` extra |
| `focused` | Temporary debug marker; never commit with this marker |

### Guidelines for new tests

- Mirror the source layout: `tests/gcm/` for `dowhy/gcm/`, etc.
- New features **must** include tests.
- Avoid brittle assertions on floating-point results; use tolerances (`pytest.approx`, `np.testing.assert_allclose`).
- Use `@pytest.mark.advanced` for tests that are slow or depend on heavy external libraries.
- Do not remove or weaken existing tests.

---

## Commit Requirements

**DCO sign-off is mandatory.** Every commit must be signed off:

```bash
git commit --signoff -m "descriptive message"
# or shorthand:
git commit -s -m "descriptive message"
```

If you forgot:

```bash
# Single commit
git commit --amend --no-edit --signoff

# Multiple commits — squash then sign
git reset --soft HEAD~<N>
git commit -s -m "new descriptive message"
```

Commits without a DCO sign-off cannot be merged.

---

## PR Checklist

Before opening or updating a PR, confirm:

- [ ] `poetry run poe lint` passes (no hard flake8 errors).
- [ ] `poetry run poe format_check` passes (black + isort compliant).
- [ ] `poetry run poe test` passes (or a justification is provided for a new failure).
- [ ] New code is covered by tests.
- [ ] All commits include a DCO sign-off (`Signed-off-by:` trailer in the commit message). For example, check with `git log --format='%h %s%n%n%b'` and verify each commit body contains `Signed-off-by:`.
- [ ] If a new dependency was added or `poetry.lock` changed, a justification is included in the PR description.
- [ ] Optional imports are guarded and fail gracefully.
- [ ] Public API additions are documented (docstrings + `docs/` RST where appropriate).

---

## Dependency Management

- DoWhy uses **Poetry** for dependency management.
- Do not edit `poetry.lock` manually; use `poetry add` / `poetry update`.
- Updating `poetry.lock` in a PR requires a written justification explaining why the update is necessary.
- Keep new dependencies to a minimum; prefer libraries already used in the project.

---

## Documentation

- Source docs are in `docs/source/` (Sphinx + RST).
- Jupyter notebook examples live in `docs/source/example_notebooks/`.
- New public functions/classes need numpy-style docstrings.
- After significant API additions, update the relevant `.rst` file under `docs/source/`.

---

## Common Pitfalls

- **Mixing frameworks**: The `gcm` module and the classic `CausalModel` workflow are separate. Do not cross-wire them without a clear interface boundary.
- **Missing optional-import guards**: Imports of `matplotlib`, `pygraphviz`, `pydot`, `torch`, etc. must be inside a `try/except ImportError` block with a user-friendly error message.
- **Breaking `CausalModel` constructor or method signatures**: This is a stable public API; use keyword arguments and default values to extend it.
- **Removing or altering test markers**: Changing an `advanced` marker to a default-run test may slow CI unexpectedly.
- **Long functions**: flake8 enforces complexity ≤ 10; refactor complex logic into helper functions.
- **Not running `format` before committing**: black and isort will cause CI to fail if code is not formatted.
