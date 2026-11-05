# Contributing to CreditLens

Thanks for your interest in improving CreditLens! This document explains how to
set up the project, make changes, and submit them.

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Prerequisites

- **Python 3.12+** (for the scoring engine, models, and API — the exact pins in
  `requirements.txt` floor there; `shap==0.52.0` requires >= 3.12)
- **Node.js 18+** and npm (for the Next.js web demo in `frontend/`)
- **git**

No external services, API keys, or secrets are required — CreditLens runs
entirely on synthetic data generated locally.

## Getting Started (Backend)

Clone the repository and install the package in editable mode with the
explainability and dev extras:

```bash
git clone https://github.com/ROROK20/CreditLens.git
cd CreditLens

# Recommended: work in a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install the package with SHAP explainability + dev tooling
pip install -e ".[xai,dev]"
```

This installs the runtime dependencies (pandas, numpy, scikit-learn, xgboost,
fastapi, uvicorn, pydantic), the `xai` extra (`shap`), and the `dev` extra
(`pytest`, `pytest-cov`, `httpx`).

## Running the Test Suite

```bash
# Run all tests
pytest tests/

# Verbose, with coverage (same as CI)
pytest tests/ --cov=creditlens --cov-report=term-missing -v
```

The full suite (526 tests as of 2026-11-05) must pass before a change can be merged.
A change that moves the served model also needs `make snapshot` (the recording the
no-backend page runs) and `make capture` (both published images); the tests say so when
either has gone stale.
Please add or update tests for any behavior you change.

## Linting & Type Checks

The project keeps a lightweight, dependency-free style. Before opening a PR,
make sure the package imports cleanly and is byte-compilable:

```bash
# Confirm the package compiles
python -m compileall creditlens

# Optional, if you have them installed locally:
ruff check creditlens        # style/lint (not required by CI)
mypy creditlens              # type checks (not required by CI)
```

For the frontend, run the linter from inside `frontend/`:

```bash
cd frontend
npm run lint
```

## Running the API

```bash
uvicorn creditlens.api.app:app --reload --port 8002
# Interactive docs at http://localhost:8002/docs
```

The API trains its model on synthetic data at startup, so you never need to train it
by hand. `POST /train` retrains in place and answers `403` unless the server was started
with `CREDITLENS_ENABLE_TRAIN=1`, because it is unauthenticated and replaces the model
for every later caller.

## Running the Web Demo (Frontend)

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:3003
```

To produce a production build:

```bash
cd frontend
npm run build
```

## Code Style

- **Python:** PEP 8, 4-space indentation, type hints on public functions, and
  module-level docstrings. Match the existing style of the surrounding code.
- **TypeScript/React:** follow the existing component conventions in
  `frontend/`; `next/core-web-vitals` ESLint rules apply.
- Keep changes focused. Avoid introducing heavyweight new dependencies without
  discussion in an issue first.

## Branch & PR Workflow

1. Fork the repository (or create a feature branch if you have write access).
2. Create a descriptive branch: `feat/batch-scoring`, `fix/fairness-edge-case`.
3. Make your changes with clear, atomic commits.
4. Ensure `pytest tests/` passes and the frontend still builds.
5. Push your branch and open a Pull Request against `main`, filling out the
   [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
6. Link any related issue and describe how you tested the change.

## Commit Message Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <short summary>
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`, `perf`.

Examples:

```
feat(api): add /score/batch endpoint
fix(fairness): handle single-group audits without crashing
docs(readme): correct uvicorn port to 8002
test(scorer): cover gini = 2*auc - 1 invariant
```

## Reporting Bugs & Requesting Features

Use the GitHub issue forms:

- [Bug report](.github/ISSUE_TEMPLATE/bug_report.yml)
- [Feature request](.github/ISSUE_TEMPLATE/feature_request.yml)

For security vulnerabilities, **do not** open a public issue — follow the
[Security Policy](SECURITY.md) instead.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE) that covers this project.
