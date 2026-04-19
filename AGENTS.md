# Repository Guidelines

## Project Structure & Module Organization
`pipeline/` contains the reusable core code: experiment orchestration, evaluation, metrics, splitters, plotting, and dataset-to-text conversion helpers under `pipeline/data_generators/` and `pipeline/data_processors/`. `1_experiments/` holds dataset-specific runs for `critical_vars`, `mimic_iv`, and `adni`; most runnable scripts are date-prefixed, for example `1_experiments/2025_02_03_adni/3_dt_gpt/2025_02_03_dt_gpt_train_full.py`. `2_various_explorations/` is for ad hoc analyses such as zero-shot studies. Treat `mimic-iv-clinical-database-demo-2.2/` as local sample data, not a place for new code.

## Build, Test, and Development Commands
Install dependencies in a fresh Python 3.8 environment:

```bash
pip install -r requirements.txt
pip install -e .
```

Run a specific experiment script directly, for example:

```bash
python 1_experiments/2025_02_03_adni/3_dt_gpt/2025_02_03_dt_gpt_train_full.py
```

Use `python -m compileall pipeline 1_experiments` as a lightweight syntax check before opening a PR. There is no repo-level `make`, `tox`, or CI wrapper checked in today.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions, variables, and scripts, and `PascalCase` for classes such as `Experiment`. Keep new reusable logic in `pipeline/`; keep one-off dataset runs inside the matching experiment folder. Preserve the repo’s filename convention for experiment scripts: `YYYY_MM_DD_description.py`. `ruff` is listed in `requirements.txt`, but no shared config is committed, so avoid large formatting-only diffs.

## Testing Guidelines
This repository currently has no dedicated `tests/` package. Run validation and smoke checks from the `conda` environment named `dtgpt`; using the system `python3` may miss required packages. For changes in `pipeline/`, add focused smoke checks by running the affected script or module and capture the exact command in your PR. When possible, validate imports and syntax with `python -m compileall` and document any dataset or model prerequisites needed to reproduce results.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit messages such as `Fix: shifted hours to be predicted` and `Added synthetic data example`. Keep commits scoped to one logical change. PRs should include: the dataset or experiment path touched, the command(s) used for validation, any required path or credential changes, and sample outputs or screenshots when plots/notebooks are affected.

## Data & Configuration Notes
Many scripts contain environment-specific filesystem paths and licensed-data assumptions. Do not commit secrets, access tokens, or private dataset extracts. If a change depends on local storage or external model hosting, call that out explicitly in the script header and PR description.
