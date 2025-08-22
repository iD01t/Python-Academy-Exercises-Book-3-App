# iD01t Academy – Python Exercises Book 3 

A polished, single-file desktop learning platform built with Qt (PySide6/PyQt5). It ships **10 chapters** of practical tools from Book 3: AI Summarizer, Crypto Tracker, Invoice Generator, Weather, Knowledge Base, Portfolio, Task Automation Bot, Quiz App, Resume Builder, and the Freelance Toolkit — all in one dark-themed app.

> Main entry point: `id01t_academy_book3.py`

## ✨ Highlights
- **One-file app** with tabs for each chapter
- **Modern dark UI** (Qt) with `Run / Code / Explain` panes
- **Online + local AI** (OpenAI + Transformers)
- **PDF export** (ReportLab) and HTML fallback
- **SQLite knowledge base** with optional FTS
- **Price & weather APIs**, **portfolio charts**, and more
- **Cross‑platform packaging** with PyInstaller

## 📦 Quick Start

```bash
# 1) Create & activate a venv (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install runtime dependencies (wide set, optional extras included)
pip install --upgrade requests pandas matplotlib Pillow yfinance openai transformers torch pdfkit reportlab lxml beautifulsoup4 PySide6 PyQt5

# 3) Run the app
python id01t_academy_book3.py
```

> **Note (Windows)**: For PDF generation via `pdfkit`, install the **wkhtmltopdf** binary and set its path in **Settings** inside the app.

## 🔑 API Keys (optional but recommended)
- **OpenAI**: `Settings → OpenAI API key` (for GPT-based summarization/resume bullets)
- **OpenWeather**: `Settings → OpenWeather key` (for Weather tab)

## 🧰 Features by Tab
- **AI Summarizer** — summaries via OpenAI or local `facebook/bart-large-cnn` (Transformers)
- **Crypto Tracker** — live prices from CoinGecko with threshold alerts
- **Invoice Generator** — export PDF (ReportLab) or HTML fallback
- **Weather** — 5‑day forecast via OpenWeather
- **Knowledge Base** — SQLite notes with tags (+ FTS when available)
- **Portfolio** — load tickers CSV, fetch with yfinance, render chart
- **Task Automation Bot** — scaffold for file rename, image resize, scraping
- **Quiz App** — scaffold for MCQ builder/runner
- **Resume Builder** — scaffold for AI bullets + PDF export
- **Freelance Toolkit** — scaffold dashboard that unifies invoicing, tasks, reminders

## 🧪 Development

```bash
# Lint & format (optional, recommended)
pip install black ruff
black .
ruff check .

# Run
python id01t_academy_book3.py
```

## 🏗️ Build (PyInstaller)

```bash
pip install pyinstaller
pyinstaller --noconfirm --onefile --windowed --name "iD01t_Academy_Python_Book3_Ed2" --icon icon.ico id01t_academy_book3.py
```

Artifacts will be in `dist/`.

## 💸 Monetization (ready-to-execute ideas)
- **One‑time desktop**: $19.99 with trial
- **Pro add‑ons**: premium invoice templates, cloud sync ($4.99–$9.99/mo)
- **Team license**: $99/seat with shared SQLite
- **Education**: institution packs, white‑label
- **Cross‑sell**: e‑book bundles, templates marketplace

## 📜 License
MIT — see [`LICENSE`](LICENSE).

## 🤝 Contributing
Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and our [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## 🔒 Security
See [`SECURITY.md`](SECURITY.md).

## 📣 Citation
If you use this in research/teaching, cite the repo via [`CITATION.cff`](CITATION.cff).

---

Built with ❤️ by **Guillaume Lessard / iD01t Productions**.
