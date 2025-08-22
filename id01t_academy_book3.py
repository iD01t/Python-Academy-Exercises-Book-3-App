#!/usr/bin/env python3
"""
iD01t Academy: Python Exercises Book 3, Edition 2
Complete Desktop Learning Platform, single file app
"""

APP_NAME = "iD01t Academy - Python Book 3 Edition 2"

import sys
import os
import json
import sqlite3
import logging
import base64
import time
import csv
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict

# Prefer PySide6, fallback PyQt5
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
        QHBoxLayout, QTextEdit, QPlainTextEdit, QListWidget, QComboBox,
        QDoubleSpinBox, QMessageBox, QGroupBox, QTabWidget, QLineEdit, QSpinBox,
        QFileDialog, QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView,
        QStatusBar, QDialog, QDialogButtonBox, QListWidgetItem
    )
    from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize
    from PySide6.QtGui import QIcon, QPixmap, QFont
    QT_VERSION = "PySide6"
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
        QHBoxLayout, QTextEdit, QPlainTextEdit, QListWidget, QComboBox,
        QDoubleSpinBox, QMessageBox, QGroupBox, QTabWidget, QLineEdit, QSpinBox,
        QFileDialog, QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView,
        QStatusBar, QDialog, QDialogButtonBox, QListWidgetItem
    )
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal as Signal, QSize
    from PyQt5.QtGui import QIcon, QPixmap, QFont
    QT_VERSION = "PyQt5"

# Third party core
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # offscreen for stability, then embed as image
import matplotlib.pyplot as plt
from PIL import Image
import yfinance as yf

# Optional deps
try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    import pdfkit
    HAS_PDFKIT = True
except Exception:
    HAS_PDFKIT = False

# Optional reportlab for invoices if available
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

# Tiny base64 logo
LOGO_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAA8AAAAQCAYAAADJViUEAAAACXBIWXMAAAsSAAALEgHS3X78AAAB"
    "M0lEQVQ4y5WSv0oDQRiGv2m9oXxq2QF2S2mE0tTg1D0Qp1JbJg0vQdQ0M8hB1fQqKq0QHDlVY7j7rJ"
    "kQ2y2Yy7sN7r3c4fN2nVx5y9w4w1+7v2xw4mB0y6xkW8y2V5c6p0o9f9c9s2H3Cwqg1b8rC4d4XxgN"
    "QhX6i7w1YFzjJb2e2g4wip1r5Jb3e4n4l0jX3Zk0B9j+6V0o4K9l1H1B5xVZ0Pq6Y7GdT2Q7Q2Qp4U"
    "Qq6xWkKk0m0l9g+g6gk1cQk8oG2qk1oVwQn8xg7y3Jq+Jf8gHk9E7N4k8AAAAAElFTkSuQmCC"
)

# Dark theme
DARK_STYLE = """
QApplication { background-color: #121212; color: #f2f2f2; }
QWidget { background-color: #121212; color: #f2f2f2; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 10pt; }
QMainWindow { background-color: #121212; }
QPushButton { background-color: #1976d2; color: white; border: none; padding: 8px 14px; border-radius: 6px; font-weight: 600; }
QPushButton:hover { background-color: #1565c0; }
QPushButton:disabled { background-color: #3a3a3a; color: #a0a0a0; }
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
  background-color: #1e1e1e; border: 1px solid #2f2f2f; padding: 6px; border-radius: 6px; color: #f2f2f2;
}
QListWidget, QTableWidget { background-color: #1a1a1a; border: 1px solid #2a2a2a; }
QTabWidget::pane { border: 1px solid #2f2f2f; }
QTabBar::tab { background: #1a1a1a; padding: 8px 12px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
QTabBar::tab:selected { background: #1976d2; color: white; }
QGroupBox { border: 1px solid #2a2a2a; border-radius: 6px; margin-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QStatusBar { background: #1a1a1a; }
"""

# Paths and logging
APP_DIR = Path(os.getcwd())
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

LOG_FILE = DATA_DIR / "app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(str(LOG_FILE), encoding="utf-8"), logging.StreamHandler(sys.stdout)]
)

# Settings
@dataclass
class Settings:
    openai_api_key: str = ""
    openweather_api_key: str = ""
    wkhtmltopdf_path: str = ""
    data_folder: str = "data"
    last_city: str = "New York"
    def to_dict(self) -> Dict: return asdict(self)
    @classmethod
    def from_dict(cls, data: Dict) -> "Settings": return cls(**data)

class SettingsManager:
    def __init__(self, settings_file: str = "settings.json"):
        self.settings_path = APP_DIR / settings_file
        self.settings = self.load()
    def load(self) -> Settings:
        try:
            if self.settings_path.exists():
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return Settings.from_dict(data)
        except Exception as e:
            logging.error(f"Settings load failed, {e}")
        return Settings()
    def save(self):
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(self.settings.to_dict(), f, indent=2)
        except Exception as e:
            logging.error(f"Settings save failed, {e}")
    def get(self, key: str, default=None): return getattr(self.settings, key, default)
    def set(self, key: str, value: Any): setattr(self.settings, key, value); self.save()

settings_manager = SettingsManager()

# HTTP helper
class SafeRequests:
    @staticmethod
    def get(url: str, params=None, timeout=12, retries=2) -> Optional[Dict]:
        for i in range(retries):
            try:
                r = requests.get(url, params=params, timeout=timeout)
                if r.status_code == 200:
                    return r.json()
                logging.warning(f"HTTP {r.status_code} for {url}")
            except Exception as e:
                logging.error(f"GET failed, attempt {i+1}, {e}")
                time.sleep(1)
        return None

# Thread worker
class Worker(QThread):
    result = Signal(object)
    error = Signal(str)
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func; self.args = args; self.kwargs = kwargs
    def run(self):
        try:
            res = self.func(*self.args, **self.kwargs)
            self.result.emit(res)
        except Exception as e:
            self.error.emit(str(e))

# Resources
class ResourceHelper:
    @staticmethod
    def get_icon() -> QIcon:
        if os.path.exists("icon.ico"): return QIcon("icon.ico")
        pix = QPixmap(); pix.loadFromData(base64.b64decode(LOGO_B64)); return QIcon(pix)

# Database
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS notes(
          id INTEGER PRIMARY KEY,
          title TEXT NOT NULL,
          content TEXT NOT NULL,
          tags TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)
        try:
            c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
              title, content, tags, content='notes', content_rowid='id'
            )
            """)
        except Exception:
            pass
        conn.commit(); conn.close()
    def connect(self): return sqlite3.connect(self.db_path)

# Base exercise tab
class ExerciseTab(QWidget):
    def __init__(self, title: str, code_content: str, explanation_html: str):
        super().__init__()
        self.title = title  # string, used by MainWindow sidebar and tab labels
        self.code_content = code_content
        self.explanation = explanation_html
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self.create_run_widget(), "Run")
        tabs.addTab(self.create_code_widget(), "Code")
        tabs.addTab(self.create_explain_widget(), "Explain")
        layout.addWidget(tabs)
    def create_run_widget(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); v.addWidget(QLabel("Run panel not implemented")); return w
    def create_code_widget(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        copy_btn = QPushButton("Copy code")
        edit = QPlainTextEdit(); edit.setReadOnly(True); edit.setPlainText(self.code_content.strip()); edit.setFont(QFont("Consolas", 9))
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self.code_content.strip()))
        v.addWidget(copy_btn); v.addWidget(edit); return w
    def create_explain_widget(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); t = QTextEdit(); t.setReadOnly(True); t.setHtml(self.explanation); v.addWidget(t); return w

# AI Summarizer
class AISummarizerTab(ExerciseTab):
    def __init__(self):
        code_content = """
import openai
from transformers import pipeline

class NoteSummarizer:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.local = None
    def summarize_openai(self, text, temperature=0.5):
        openai.api_key = self.api_key
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":"Summarize clearly"},{"role":"user","content": text}],
            max_tokens=180, temperature=temperature
        )
        return resp.choices[0].message["content"].strip()
    def summarize_local(self, text):
        if self.local is None:
            self.local = pipeline("summarization", model="facebook/bart-large-cnn")
        out = self.local(text[:1024], max_length=160, min_length=40, do_sample=False)
        return out[0]["summary_text"]
"""
        explanation = """
<h3>AI Note Summarizer</h3>
<ul>
<li>Purpose, turn long text into a clear summary</li>
<li>Backends, OpenAI if API key is set, local BART if transformers are installed</li>
<li>Threaded calls, history, file export</li>
</ul>
"""
        super().__init__("AI Summarizer", code_content, explanation)
        self.history = []
    def create_run_widget(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        self.input = QTextEdit(); self.input.setPlaceholderText("Paste text to summarize"); v.addWidget(self.input)
        row = QHBoxLayout()
        self.model = QComboBox(); options=[]
        if HAS_OPENAI: options.append("OpenAI")
        if HAS_TRANSFORMERS: options.append("Local BART")
        if not options: options = ["No model available"]
        self.model.addItems(options)
        self.temp = QDoubleSpinBox(); self.temp.setRange(0.0,1.0); self.temp.setSingleStep(0.1); self.temp.setValue(0.5)
        row.addWidget(QLabel("Model")); row.addWidget(self.model); row.addWidget(QLabel("Temperature")); row.addWidget(self.temp)
        v.addLayout(row)
        actions = QHBoxLayout()
        run = QPushButton("Summarize"); run.clicked.connect(self.do_summarize)
        save = QPushButton("Save summary"); save.clicked.connect(self.save_summary)
        actions.addWidget(run); actions.addWidget(save); v.addLayout(actions)
        self.output = QTextEdit(); self.output.setReadOnly(True); v.addWidget(self.output)
        v.addWidget(QLabel("History"))
        self.hist = QListWidget(); self.hist.itemClicked.connect(self.load_history); v.addWidget(self.hist)
        return w
    def do_summarize(self):
        text = self.input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Empty", "Provide text"); return
        model = self.model.currentText()
        def work():
            try:
                if model == "OpenAI":
                    key = settings_manager.get("openai_api_key")
                    if not key: return "Error, OpenAI key not configured"
                    openai.api_key = key
                    r = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role":"system","content":"Summarize in five bullet points"},
                                  {"role":"user","content": text}],
                        max_tokens=220, temperature=float(self.temp.value())
                    )
                    return r.choices[0].message["content"].strip()
                elif model == "Local BART":
                    if not HAS_TRANSFORMERS: return "Error, transformers not installed"
                    summ = pipeline("summarization", model="facebook/bart-large-cnn")
                    out = summ(text[:1024], max_length=160, min_length=40, do_sample=False)
                    return out[0]["summary_text"]
                return "No model available"
            except Exception as e:
                return f"Error, {e}"
        self._w = Worker(work)
        self._w.result.connect(self.on_summary_ready)
        self._w.start()
    def on_summary_ready(self, res):
        self.output.setPlainText(res)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        preview = self.input.toPlainText()[:60].replace("\n", " ")
        self.history.append({"ts": ts, "preview": preview, "summary": res})
        self.hist.addItem(f"{ts} | {preview}")
    def save_summary(self):
        if not self.output.toPlainText().strip():
            QMessageBox.information(self, "Nothing", "Run a summary first"); return
        fn, _ = QFileDialog.getSaveFileName(self, "Save summary", str(DATA_DIR / "summary.txt"), "Text files (*.txt)")
        if not fn: return
        with open(fn, "w", encoding="utf-8") as f: f.write(self.output.toPlainText())
        QMessageBox.information(self, "Saved", "Summary saved")
    def load_history(self, item: QListWidgetItem):
        i = self.hist.row(item)
        if 0 <= i < len(self.history): self.output.setPlainText(self.history[i]["summary"])

# Crypto Tracker, fixed initialization order
class CryptoTrackerTab(ExerciseTab):
    def __init__(self):
        # Init BEFORE super().__init__ because super builds UI which calls start_timer
        self.alerts = []
        self.prices = {}
        self.timer = None
        self.worker = None
        code = """
import requests
API = "https://api.coingecko.com/api/v3/simple/price"
def fetch_prices(coins):
    r = requests.get(API, params={"ids": ",".join(coins), "vs_currencies":"usd"}, timeout=10)
    return r.json() if r.status_code == 200 else {}
"""
        exp = """
<h3>Crypto Price Tracker</h3>
<ul>
<li>Live prices from CoinGecko</li>
<li>Non blocking updates every thirty seconds</li>
<li>Simple threshold alerts</li>
</ul>
"""
        super().__init__("Crypto Price Tracker", code, exp)
    def create_run_widget(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(QLabel("Live Prices, updates every thirty seconds"))
        self.price_list = QListWidget(); v.addWidget(self.price_list)
        grp = QGroupBox("Set Price Alert"); g = QHBoxLayout(grp)
        self.coin = QComboBox(); self.coin.addItems(["bitcoin","ethereum","solana","cardano","polkadot"])
        self.level = QDoubleSpinBox(); self.level.setRange(0.01, 1_000_000.0); self.level.setValue(50000.0)
        self.cond = QComboBox(); self.cond.addItems(["above","below"])
        add = QPushButton("Add"); add.clicked.connect(self.add_alert)
        for wdg in [QLabel("Coin"), self.coin, QLabel("Target"), self.level, QLabel("When"), self.cond, add]: g.addWidget(wdg)
        v.addWidget(grp)
        v.addWidget(QLabel("Active Alerts")); self.alerts_list = QListWidget(); v.addWidget(self.alerts_list)
        self.start_timer()
        return w
    def start_timer(self):
        if getattr(self, "timer", None): return
        self.timer = QTimer(); self.timer.timeout.connect(self.fetch_prices); self.timer.start(30000)
        self.fetch_prices()
    def fetch_prices(self):
        coins = ["bitcoin","ethereum","solana","cardano","polkadot"]
        def work():
            return SafeRequests.get("https://api.coingecko.com/api/v3/simple/price",
                                    params={"ids": ",".join(coins), "vs_currencies": "usd"}, timeout=10) or {}
        self.worker = Worker(work)
        self.worker.result.connect(self.on_prices)
        self.worker.start()
    def on_prices(self, data):
        self.prices = data or {}
        self.price_list.clear()
        for c, p in self.prices.items():
            self.price_list.addItem(f"{c.title()}  ${float(p.get('usd',0.0)):,.2f}")
        self.check_alerts()
    def add_alert(self):
        a = {"coin": self.coin.currentText(), "target": float(self.level.value()), "cond": self.cond.currentText()}
        self.alerts.append(a); self.render_alerts(); QMessageBox.information(self, "Alert", "Alert added")
    def render_alerts(self):
        self.alerts_list.clear()
        for a in self.alerts: self.alerts_list.addItem(f"{a['coin']} {a['cond']} ${a['target']:,.2f}")
    def check_alerts(self):
        if not self.prices: return
        fired = []
        for a in list(self.alerts):
            if a["coin"] in self.prices:
                now = float(self.prices[a["coin"]].get("usd", 0.0))
                if (a["cond"]=="above" and now>=a["target"]) or (a["cond"]=="below" and now<=a["target"]):
                    fired.append((a, now)); self.alerts.remove(a)
        if fired:
            self.render_alerts()
            for a, now in fired:
                QMessageBox.information(self, "Price alert", f"{a['coin']} hit ${now:,.2f}")

# Invoice Generator
class InvoiceGeneratorTab(ExerciseTab):
    def __init__(self):
        code = """
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
def build_invoice(filename, rows, tax_rate=0.0):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = [Paragraph("INVOICE", styles["Title"]), Spacer(1, 12)]
    data = [["Description","Qty","Rate","Total"]] + rows
    tbl = Table(data)
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
                             ("BOX",(0,0),(-1,-1),1,colors.black),
                             ("GRID",(0,0),(-1,-1),0.5,colors.grey)]))
    story.append(tbl)
    subtotal = sum(float(r[3].replace("$","").replace(",","")) for r in rows)
    tax = subtotal * (tax_rate/100.0)
    total = subtotal + tax
    story.append(Paragraph(f"Subtotal  ${subtotal:,.2f}", styles["Normal"]))
    story.append(Paragraph(f"Tax {tax_rate:.2f}%  ${tax:,.2f}", styles["Normal"]))
    story.append(Paragraph(f"Total  ${total:,.2f}", styles["Heading2"]))
    doc.build(story)
"""
        exp = """
<h3>Invoice Generator</h3>
<ul>
<li>PDF via reportlab if installed, HTML fallback included</li>
<li>Line items with qty, rate, tax and totals</li>
</ul>
"""
        super().__init__("Invoice Generator", code, exp)
        self.rows = []
    def create_run_widget(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        row = QHBoxLayout()
        self.inv_desc = QLineEdit(); self.inv_qty = QSpinBox(); self.inv_qty.setRange(1,100000)
        self.inv_rate = QDoubleSpinBox(); self.inv_rate.setRange(0.0, 1_000_000.0); self.inv_rate.setDecimals(2)
        add = QPushButton("Add item"); add.clicked.connect(self.add_item)
        for wdg in [QLabel("Description"), self.inv_desc, QLabel("Qty"), self.inv_qty, QLabel("Rate"), self.inv_rate, add]: row.addWidget(wdg)
        v.addLayout(row)
        self.inv_table = QTableWidget(0,4); self.inv_table.setHorizontalHeaderLabels(["Description","Qty","Rate","Total"])
        self.inv_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); v.addWidget(self.inv_table)
        row2 = QHBoxLayout()
        self.inv_tax = QDoubleSpinBox(); self.inv_tax.setRange(0.0, 100.0); self.inv_tax.setValue(5.0)
        pdf_btn = QPushButton("Export PDF"); pdf_btn.clicked.connect(self.export_pdf)
        html_btn = QPushButton("Export HTML"); html_btn.clicked.connect(self.export_html)
        for wdg in [QLabel("Tax percent"), self.inv_tax, pdf_btn, html_btn]: row2.addWidget(wdg)
        v.addLayout(row2)
        self.total_label = QLabel("Total  $0.00"); v.addWidget(self.total_label)
        return w
    def add_item(self):
        d = self.inv_desc.text().strip()
        if not d: QMessageBox.warning(self, "Missing", "Enter description"); return
        q = int(self.inv_qty.value()); r = float(self.inv_rate.value()); total = q*r
        self.rows.append([d, str(q), f"${r:,.2f}", f"${total:,.2f}"]); self.inv_desc.clear(); self.refresh()
    def refresh(self):
        self.inv_table.setRowCount(0)
        for r in self.rows:
            i = self.inv_table.rowCount(); self.inv_table.insertRow(i)
            for c, val in enumerate(r): self.inv_table.setItem(i, c, QTableWidgetItem(val))
        subtotal = sum(float(r[3].replace("$","").replace(",","")) for r in self.rows)
        tax = subtotal * (float(self.inv_tax.value())/100.0); total = subtotal + tax
        self.total_label.setText(f"Total  ${total:,.2f}")
    def export_pdf(self):
        if not self.rows: QMessageBox.information(self, "Nothing", "Add items first"); return
        fn, _ = QFileDialog.getSaveFileName(self, "Save PDF", str(DATA_DIR / "invoice.pdf"), "PDF files (*.pdf)")
        if not fn: return
        if HAS_REPORTLAB:
            try:
                styles = getSampleStyleSheet()
                doc = SimpleDocTemplate(fn, pagesize=letter)
                story = [Paragraph("INVOICE", styles["Title"]), ]
                data = [["Description","Qty","Rate","Total"]] + self.rows
                tbl = Table(data)
                tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
                                         ("BOX",(0,0),(-1,-1),1,colors.black),
                                         ("GRID",(0,0),(-1,-1),0.5,colors.grey)]))
                story.append(tbl)
                subtotal = sum(float(r[3].replace("$","").replace(",","")) for r in self.rows)
                tax = subtotal * (float(self.inv_tax.value())/100.0); total = subtotal + tax
                story += [Paragraph(f"Subtotal  ${subtotal:,.2f}", styles["Normal"]),
                          Paragraph(f"Tax {float(self.inv_tax.value()):.2f}%  ${tax:,.2f}", styles["Normal"]),
                          Paragraph(f"Total  ${total:,.2f}", styles["Heading2"])]
                doc.build(story)
                QMessageBox.information(self, "Done", "PDF exported")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"PDF failed, {e}")
        else:
            QMessageBox.information(self, "Reportlab missing", "Use HTML export or install reportlab")
    def export_html(self):
        if not self.rows: QMessageBox.information(self, "Nothing", "Add items first"); return
        fn, _ = QFileDialog.getSaveFileName(self, "Save HTML", str(DATA_DIR / "invoice.html"), "HTML files (*.html)")
        if not fn: return
        try:
            html = ["<html><head><meta charset='utf-8'><style>table{border-collapse:collapse}td,th{border:1px solid #444;padding:6px}</style></head><body>"]
            html.append("<h1>INVOICE</h1>")
            html.append("<table><tr><th>Description</th><th>Qty</th><th>Rate</th><th>Total</th></tr>")
            for r in self.rows: html.append(f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td></tr>")
            subtotal = sum(float(r[3].replace('$','').replace(',','')) for r in self.rows)
            tax = subtotal * (float(self.inv_tax.value())/100.0); total = subtotal + tax
            html += [ "</table>", f"<p>Subtotal  ${subtotal:,.2f}</p>", f"<p>Tax {float(self.inv_tax.value()):.2f}%  ${tax:,.2f}</p>", f"<h2>Total  ${total:,.2f}</h2>", "</body></html>" ]
            with open(fn, "w", encoding="utf-8") as f: f.write("\n".join(html))
            QMessageBox.information(self, "Done", "HTML exported")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"HTML failed, {e}")

# Weather Dashboard
class WeatherTab(ExerciseTab):
    def __init__(self):
        code = """
import requests
API = "https://api.openweathermap.org/data/2.5/forecast"
def fetch(city, key):
    r = requests.get(API, params={"q":city, "appid":key, "units":"metric"}, timeout=12)
    return r.json() if r.status_code == 200 else {}
"""
    # explanation string
        exp = """
<h3>Weather</h3>
<ul>
<li>Forecast using OpenWeather, metric units</li>
<li>Store last city in settings</li>
</ul>
"""
        super().__init__("Weather", code, exp)
    def create_run_widget(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        row = QHBoxLayout()
        self.city_edit = QLineEdit(settings_manager.get("last_city","New York"))
        btn = QPushButton("Fetch"); btn.clicked.connect(self.fetch_weather)
        row.addWidget(QLabel("City")); row.addWidget(self.city_edit); row.addWidget(btn); v.addLayout(row)
        self.weather_list = QListWidget(); v.addWidget(self.weather_list)
        return w
    def fetch_weather(self):
        key = settings_manager.get("openweather_api_key")
        if not key: QMessageBox.information(self, "Key", "Set OpenWeather key in Settings"); return
        city = self.city_edit.text().strip()
        if not city: QMessageBox.warning(self, "City", "Enter a city"); return
        settings_manager.set("last_city", city)
        def work(): return SafeRequests.get("https://api.openweathermap.org/data/2.5/forecast",
                                            params={"q": city, "appid": key, "units":"metric"})
        self._w = Worker(work); self._w.result.connect(self.on_weather); self._w.start()
    def on_weather(self, data):
        self.weather_list.clear()
        if not data or "list" not in data: self.weather_list.addItem("No data"); return
        for item in data["list"][:16]:
            ts = item.get("dt_txt",""); desc = item.get("weather",[{}])[0].get("description","")
            temp = item.get("main",{}).get("temp",0.0); self.weather_list.addItem(f"{ts}  {temp:.1f}°C  {desc}")

# Knowledge Base
class KnowledgeBaseTab(ExerciseTab):
    def __init__(self, db: DatabaseManager):
        self.db = db
        code = """
import sqlite3
def add_note(conn, title, content, tags):
    c = conn.cursor()
    c.execute("INSERT INTO notes(title, content, tags) VALUES(?,?,?)", (title, content, tags))
    nid = c.lastrowid
    try:
        c.execute("INSERT INTO notes_fts(rowid, title, content, tags) SELECT id, title, content, tags FROM notes WHERE id=?", (nid,))
    except Exception:
        pass
    conn.commit()
    return nid
"""
        exp = """
<h3>Knowledge Base</h3>
<ul>
<li>SQLite notes with tags and optional FTS search</li>
<li>Export selected note to Markdown</li>
</ul>
"""
        super().__init__("Knowledge Base", code, exp)
    def create_run_widget(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        top = QHBoxLayout()
        self.note_title_edit = QLineEdit()
        self.tags_edit = QLineEdit()
        add = QPushButton("Add note"); add.clicked.connect(self.add_note)
        for wdg in [QLabel("Title"), self.note_title_edit, QLabel("Tags"), self.tags_edit, add]: top.addWidget(wdg)
        v.addLayout(top)
        self.note_body_edit = QTextEdit(); self.note_body_edit.setPlaceholderText("Write content"); v.addWidget(self.note_body_edit)
        row = QHBoxLayout()
        self.search_edit = QLineEdit(); self.search_edit.setPlaceholderText("Search")
        find = QPushButton("Find"); find.clicked.connect(self.find_notes)
        export = QPushButton("Export selected to Markdown"); export.clicked.connect(self.export_note)
        row.addWidget(self.search_edit); row.addWidget(find); row.addWidget(export); v.addLayout(row)
        self.notes_list = QListWidget(); self.notes_list.itemClicked.connect(self.load_note); v.addWidget(self.notes_list)
        return w
    def add_note(self):
        t = self.note_title_edit.text().strip()
        if not t: QMessageBox.warning(self, "Missing", "Title required"); return
        content = self.note_body_edit.toPlainText().strip(); tags = self.tags_edit.text().strip()
        conn = self.db.connect(); c = conn.cursor()
        c.execute("INSERT INTO notes(title, content, tags) VALUES(?,?,?)", (t, content, tags))
        nid = c.lastrowid
        try: c.execute("INSERT INTO notes_fts(rowid, title, content, tags) SELECT id, title, content, tags FROM notes WHERE id=?", (nid,))
        except Exception: pass
        conn.commit(); conn.close()
        self.note_title_edit.clear(); self.note_body_edit.clear(); self.tags_edit.clear()
        self.find_notes()
    def find_notes(self):
        q = self.search_edit.text().strip(); conn = self.db.connect(); c = conn.cursor()
        try:
            if q: c.execute("SELECT n.id, n.title, n.tags FROM notes n JOIN notes_fts f ON n.id=f.rowid WHERE notes_fts MATCH ?", (q,))
            else: c.execute("SELECT id, title, tags FROM notes ORDER BY created_at DESC")
        except Exception:
            c.execute("SELECT id, title, tags FROM notes ORDER BY created_at DESC")
        rows = c.fetchall(); conn.close()
        self.notes_list.clear()
        for r in rows:
            item = QListWidgetItem(f"{r[0]} | {r[1]} | {r[2]}"); item.setData(Qt.UserRole, r[0]); self.notes_list.addItem(item)
    def load_note(self, item: QListWidgetItem):
        nid = item.data(Qt.UserRole); conn = self.db.connect(); c = conn.cursor()
        c.execute("SELECT title, content, tags FROM notes WHERE id=?", (nid,)); row = c.fetchone(); conn.close()
        if row: self.note_title_edit.setText(row[0]); self.note_body_edit.setPlainText(row[1]); self.tags_edit.setText(row[2])
    def export_note(self):
        item = self.notes_list.currentItem()
        if not item: QMessageBox.information(self, "No selection", "Choose a note"); return
        nid = item.data(Qt.UserRole); conn = self.db.connect(); c = conn.cursor()
        c.execute("SELECT title, content, tags FROM notes WHERE id=?", (nid,)); row = c.fetchone(); conn.close()
        if not row: QMessageBox.information(self, "Missing", "Note not found"); return
        fn, _ = QFileDialog.getSaveFileName(self, "Save Markdown", str(DATA_DIR / "note.md"), "Markdown files (*.md)")
        if not fn: return
        with open(fn, "w", encoding="utf-8") as f:
            f.write(f"# {row[0]}\n\n{row[1]}\n\nTags, {row[2]}\n")
        QMessageBox.information(self, "Saved", "Exported")

# Portfolio Analyzer
class PortfolioTab(ExerciseTab):
    def __init__(self):
        code = """
import yfinance as yf, pandas as pd
def load_prices(tickers):
    return yf.download(tickers, period="1mo", interval="1d", progress=False)["Adj Close"]
"""
        exp = """
<h3>Portfolio Analyzer</h3>
<ul>
<li>Load CSV of ticker and shares</li>
<li>Fetch recent prices and render a chart</li>
</ul>
"""
        super().__init__("Portfolio", code, exp)
        self.positions = []
    def create_run_widget(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        row = QHBoxLayout()
        load = QPushButton("Load positions CSV"); load.clicked.connect(self.load_csv)
        sample = QPushButton("Create sample CSV"); sample.clicked.connect(self.create_sample)
        analyze = QPushButton("Analyze"); analyze.clicked.connect(self.analyze)
        row.addWidget(load); row.addWidget(sample); row.addWidget(analyze); v.addLayout(row)
        self.pos_table = QTableWidget(0, 3); self.pos_table.setHorizontalHeaderLabels(["Ticker","Shares","Last Price"])
        self.pos_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); v.addWidget(self.pos_table)
        self.chart = QLabel("Chart will render here"); self.chart.setMinimumHeight(220); self.chart.setAlignment(Qt.AlignCenter); v.addWidget(self.chart)
        return w
    def load_csv(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV files (*.csv)")
        if not fn: return
        try:
            self.positions = []
            with open(fn, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    t = row.get("ticker","").strip().upper()
                    s = float(row.get("shares","0") or 0)
                    if t: self.positions.append((t, s))
            self.render_positions([])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read CSV, {e}")
    def create_sample(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save sample", str(DATA_DIR / "positions_sample.csv"), "CSV files (*.csv)")
        if not fn: return
        with open(fn, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["ticker","shares"]); w.writerow(["AAPL","10"]); w.writerow(["MSFT","5"]); w.writerow(["NVDA","2"])
        QMessageBox.information(self, "Saved", "Sample CSV written")
    def analyze(self):
        if not self.positions: QMessageBox.information(self, "Positions", "Load a CSV first"); return
        tickers = [t for t, _ in self.positions]
        try:
            data = yf.download(tickers, period="1mo", interval="1d", progress=False)["Adj Close"]
            last = data.iloc[-1]; rows = []
            for t, s in self.positions:
                lp = float(last.get(t, 0)); rows.append((t, s, lp))
            self.render_positions(rows)
            fig, ax = plt.subplots(figsize=(6, 2.5)); data.fillna(method="ffill").plot(ax=ax)
            ax.set_title("Last month adjusted close"); ax.set_xlabel("Date"); ax.set_ylabel("Price"); fig.tight_layout()
            img_path = DATA_DIR / "portfolio_chart.png"; fig.savefig(str(img_path)); plt.close(fig)
            pix = QPixmap(str(img_path))
            self.chart.setPixmap(pix.scaled(self.chart.width(), self.chart.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed, {e}")
    def render_positions(self, rows):
        self.pos_table.setRowCount(0)
        for t, s, lp in rows or []:
            i = self.pos_table.rowCount(); self.pos_table.insertRow(i)
            self.pos_table.setItem(i, 0, QTableWidgetItem(t))
            self.pos_table.setItem(i, 1, QTableWidgetItem(str(s)))
            self.pos_table.setItem(i, 2, QTableWidgetItem(f"${lp:,.2f}"))
# --- NEW: Chapter 7 ---
class TaskAutomationBotTab(ExerciseTab):
    def __init__(self):
        code = """
# Task Automation Bot (skeleton)
# Planned: file rename, image resize, simple web scrape
"""
        exp = """
<h3>Task Automation Bot</h3>
<p>Automate file renaming, image resizing, and basic web scraping.</p>
"""
        super().__init__("Task Automation Bot", code, exp)
    def create_run_widget(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(QLabel("Planned actions:"))
        row = QHBoxLayout()
        row.addWidget(QPushButton("Rename Files (soon)"))
        row.addWidget(QPushButton("Resize Images (soon)"))
        row.addWidget(QPushButton("Scrape URL (soon)"))
        v.addLayout(row)
        v.addWidget(QLabel("Tip: This tab is a scaffold; enable features incrementally."))
        return w

# --- NEW: Chapter 8 ---
class QuizAppTab(ExerciseTab):
    def __init__(self):
        code = """
# Interactive Quiz App (skeleton)
# Planned: MCQ editor, timer, scoring, CSV/JSON export
"""
        exp = """
<h3>Interactive Quiz App</h3>
<p>Timed quizzes with scoring and result history.</p>
"""
        super().__init__("Quiz App", code, exp)
    def create_run_widget(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(QLabel("Quiz builder and runner coming next."))
        v.addWidget(QPushButton("Create Question (soon)"))
        v.addWidget(QPushButton("Start Timed Quiz (soon)"))
        return w

# --- NEW: Chapter 9 ---
class ResumeBuilderTab(ExerciseTab):
    def __init__(self):
        code = """
# AI Resume Builder (skeleton)
# Planned: form sections, OpenAI bullet helpers, ReportLab export
"""
        exp = """
<h3>AI-Powered Resume Builder</h3>
<p>Fill sections, generate improved bullets, export styled PDF.</p>
"""
        super().__init__("Resume Builder", code, exp)
    def create_run_widget(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(QLabel("Requirements: ReportLab for PDF, OpenAI (optional) for AI bullets."))
        v.addWidget(QPushButton("Open Resume Form (soon)"))
        return w

# --- NEW: Chapter 10 ---
class FreelanceToolkitTab(ExerciseTab):
    def __init__(self):
        code = """
# Freelance Business Toolkit (skeleton)
# Planned: integrate invoice/task/reminders in one dashboard, shared SQLite
"""
        exp = """
<h3>Freelance Business Toolkit</h3>
<p>Central dashboard that unifies invoices, tasks and reminders.</p>
"""
        super().__init__("Freelance Toolkit", code, exp)
    def create_run_widget(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(QLabel("Unified dashboard coming soon."))
        return w


# Settings dialog
class SettingsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings"); self.setMinimumWidth(420)
        form = QFormLayout(self)
        self.openai_edit = QLineEdit(settings_manager.get("openai_api_key",""))
        self.owm_edit = QLineEdit(settings_manager.get("openweather_api_key",""))
        self.wk_edit = QLineEdit(settings_manager.get("wkhtmltopdf_path",""))
        self.data_edit = QLineEdit(settings_manager.get("data_folder","data"))
        form.addRow("OpenAI API key", self.openai_edit)
        form.addRow("OpenWeather key", self.owm_edit)
        form.addRow("wkhtmltopdf path", self.wk_edit)
        form.addRow("Data folder", self.data_edit)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject); form.addRow(btns)
    def accept(self):
        settings_manager.set("openai_api_key", self.openai_edit.text().strip())
        settings_manager.set("openweather_api_key", self.owm_edit.text().strip())
        settings_manager.set("wkhtmltopdf_path", self.wk_edit.text().strip())
        settings_manager.set("data_folder", self.data_edit.text().strip() or "data")
        super().accept()

# Main Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME); self.setWindowIcon(ResourceHelper.get_icon()); self.resize(1200, 780)
        menubar = self.menuBar()
        if QT_VERSION == "PySide6": menubar.setNativeMenuBar(False)
        m_file = menubar.addMenu("File")
        act_settings = m_file.addAction("Settings"); act_settings.triggered.connect(self.open_settings)
        m_file.addSeparator(); m_file.addAction("Quit").triggered.connect(self.close)
        m_help = menubar.addMenu("Help")
        m_help.addAction("About").triggered.connect(self.about)
        m_help.addAction("Upgrade ideas").triggered.connect(self.upgrade)
        central = QWidget(); self.setCentralWidget(central); h = QHBoxLayout(central)
        self.sidebar = QListWidget(); self.sidebar.setFixedWidth(220); self.sidebar.itemClicked.connect(self.switch_tab); h.addWidget(self.sidebar)
        self.stack = QTabWidget(); self.stack.tabBar().hide(); h.addWidget(self.stack)
        self.status = QStatusBar(); self.setStatusBar(self.status); self.status.showMessage(f"{APP_NAME} ready, Qt {QT_VERSION}")
        DATA_DIR.mkdir(exist_ok=True); self.db = DatabaseManager(str(DATA_DIR / "academy.db"))
        self.tabs = []
        self.add_tab(AISummarizerTab())
        self.add_tab(CryptoTrackerTab())
        self.add_tab(InvoiceGeneratorTab())
        self.add_tab(WeatherTab())
        self.add_tab(KnowledgeBaseTab(self.db))
        self.add_tab(PortfolioTab())
        self.add_tab(TaskAutomationBotTab())
        self.add_tab(QuizAppTab())
        self.add_tab(ResumeBuilderTab())
        self.add_tab(FreelanceToolkitTab())

        if self.sidebar.count(): self.sidebar.setCurrentRow(0); self.stack.setCurrentIndex(0)
    def add_tab(self, widget: ExerciseTab):
        self.stack.addTab(widget, widget.title)  # widget.title is a string
        item = QListWidgetItem(widget.title); item.setSizeHint(QSize(200, 38)); self.sidebar.addItem(item); self.tabs.append(widget)
    def switch_tab(self, item: QListWidgetItem):
        idx = self.sidebar.row(item)
        if 0 <= idx < self.stack.count(): self.stack.setCurrentIndex(idx); self.status.showMessage(f"Opened, {self.stack.tabText(idx)}")
    def open_settings(self):
        dlg = SettingsDialog(); dlg.exec_() if QT_VERSION == "PyQt5" else dlg.exec()
    def about(self):
        QMessageBox.information(self, "About", f"{APP_NAME}\n\n2025 iD01t, id01t.store\ngithub.com/id01t")
    def upgrade(self):
        tips = [
            "Offer cloud sync for notes and invoices",
            "Add batch crypto alerts and phone notifications",
            "Provide premium templates for invoices",
            "Add portfolio risk metrics and factor analytics",
            "Bundle a team license with shared SQLite"
        ]
        QMessageBox.information(self, "Upgrade ideas", "\n".join(f"• {t}" for t in tips))

# Dependency bootstrapper, includes more than strictly needed
def ensure_deps():
    needed = [
        ("requests", "requests"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("PIL", "Pillow"),
        ("yfinance", "yfinance"),
        # full feature coverage extras
        ("openai", "openai"),
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("reportlab", "reportlab"),
        ("pdfkit", "pdfkit"),
        # handy extras for future extensions
        ("lxml", "lxml"),
        ("bs4", "beautifulsoup4")
    ]
    missing = []
    for mod, pipname in needed:
        try:
            __import__(mod if mod != "PIL" else "PIL.Image")
        except Exception:
            missing.append(pipname)
    if missing:
        reply = QMessageBox.question(None, "Install dependencies",
                                     f"The following packages are missing: {', '.join(missing)}\nInstall now using pip",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            python = sys.executable
            for pkg in missing:
                try:
                    subprocess.check_call([python, "-m", "pip", "install", "--upgrade", pkg])
                except Exception as e:
                    QMessageBox.warning(None, "Install failed", f"{pkg} failed, {e}")

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(ResourceHelper.get_icon())
    ensure_deps()
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

# ------------------------------
# Wide install command, includes required and optional packages
# pip install --upgrade requests pandas matplotlib Pillow yfinance openai transformers torch pdfkit reportlab lxml beautifulsoup4
# If pdfkit fails to export, install wkhtmltopdf system binary and set path in Settings
# Windows: install from https://wkhtmltopdf.org/downloads.html

# PyInstaller, onefile, windowed, use icon.ico if present
# pyinstaller --noconfirm --onefile --windowed --name "iD01t_Academy_Python_Book3_Ed2" --icon icon.ico "%~dp0YOUR_SCRIPT_NAME.py"

# Quick start
# 1) Install Python 3.10+
# 2) Run the wide pip command above
# 3) Launch the script, set API keys in Settings as needed
# 4) Package with PyInstaller if you want a distributable
