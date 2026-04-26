# Installation

## Requirements

- Python 3.10+
- NumPy, SciPy, pandas, scikit-learn (installed automatically)

## Install from PyPI

```bash
pip install skillinfer
```

## With visualization support

For correlation heatmaps, scree plots, and posterior profile charts:

```bash
pip install skillinfer[viz]
```

This adds `matplotlib` as a dependency.

## Development install

Clone the repository and install in editable mode with dev dependencies:

```bash
git clone https://github.com/kostadindev/skillinfer.git
cd skillinfer
pip install -e ".[dev]"
```

The `dev` extra includes `matplotlib` and `pytest`.

## Verify installation

```python
import skillinfer
print(skillinfer.__version__)
# 0.1.0
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >= 1.24 | Array operations, linear algebra |
| `scipy` | >= 1.10 | Statistical distributions, hierarchical clustering |
| `pandas` | >= 2.0 | DataFrames for input/output |
| `scikit-learn` | >= 1.2 | Ledoit-Wolf covariance estimation, PCA |
| `matplotlib` | >= 3.7 | Visualization (optional, via `skillinfer[viz]`) |
