# Kairos: "the critical time"

[![CI](https://github.com/mlkbsalah/kairos/actions/workflows/ci.yml/badge.svg)](https://github.com/mlkbsalah/kairos/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-black)](https://opensource.org/licenses/MIT)

## Overview

A JAX-based survival analysis library for efficient implementation of survival analysis tools. Survival analysis studies the time until an event of interest occurs (e.g., disease recurrence, equipment failure).

This package provides a lightweight toolkit built entirely on JAX. All functions are `jit`-compatible and integrate seamlessly with `Flax` and `Equinox` training loops. Kairos enables flexible model definitions while maintaining a pure functional backend. With minimal dependencies, full JAX support, and a clean API, it facilitates efficient survival model implementation.

## Key Features

- **Non-parametric models**: Kaplan-Meier estimator, log-rank test, Nelson-Aalen cumulative hazard
- **Semi-parametric models**: Cox proportional hazards with partial likelihood
- **Parametric models**: Weibull, exponential, and log-normal distributions
- **Evaluation metrics**: Concordance index (C-index), integrated Brier score (IBS), negative log-likelihood (NLL)
- **Pure JAX backend**: `jit`-friendly, composable with custom models

## Installation

```bash
git clone git@github.com:mlkbsalah/kairos.git
cd kairos
uv sync
```

## Status

Under active development. See [GitHub issues](https://github.com/mlkbsalah/kairos/issues) for the roadmap and current progress.

## License

MIT
