# Contributing Guide

Thank you for investing in iD01t Academy – Python Exercises Book 3 (Edition 2)! This guide keeps contributions fast and high‑quality.

## Ground Rules
- Use **Conventional Commits** for messages (e.g., `feat:`, `fix:`).
- Format with **Black** and lint with **Ruff**.
- Target **small, focused PRs**.

## Dev Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade -r requirements-dev.txt || pip install black ruff pytest
pip install --upgrade requests pandas matplotlib Pillow yfinance openai transformers torch pdfkit reportlab lxml beautifulsoup4 PySide6 PyQt5
```

## Running
```bash
python id01t_academy_book3.py
```

## Tests
If you add logic, add tests (pytest). Put unit tests under `tests/`.

## Code Style
- **Black** defaults.
- **Ruff** for linting (`ruff check .`).
- Avoid blocking UI threads; use `Worker(QThread)` for network/CPU tasks.

## PR Process
1. Fork & branch from `main` (e.g., `feat/quiz-runner`).
2. Ensure `black . && ruff check .` is clean.
3. Update docs (README, CHANGELOG if needed).
4. Open PR; link to the issue; describe changes and test plan.

## Release
We follow **Semantic Versioning** and keep a human‑readable `CHANGELOG.md`.
