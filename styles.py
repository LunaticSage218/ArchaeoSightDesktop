"""
Central dark-theme stylesheet and shared widget helpers for ArchaeoSight Desktop.

Import DARK_STYLESHEET in main.py and apply it once to the top-level widget.
Import the helper functions (_section, _bold_label, _primary_btn, _h_line,
PAGE_HEADER_BG, PAGE_HEADER_FG, IMG_BORDER_STYLE) in each page module.
"""

from PyQt6.QtWidgets import QGroupBox, QLabel, QPushButton, QFrame
from PyQt6.QtGui import QFont

# ── Colour tokens ──────────────────────────────────────────────────────────────
BG_BASE      = "#1e1e2e"   # main background
BG_SURFACE   = "#272738"   # cards / group boxes
BG_INPUT     = "#2a2a3d"   # input fields
BG_HEADER    = "#1a1a2e"   # page header banner

BORDER       = "#3b3b54"   # borders, separators
BORDER_LIGHT = "#44445e"   # lighter accent border

TEXT_PRIMARY  = "#e5e7eb"   # body text – high contrast on dark
TEXT_SECONDARY = "#9ca3af"  # subtle / placeholder
TEXT_HEADING  = "#f3f4f6"   # headings / bold labels

ACCENT       = "#3b82f6"   # blue accent (buttons, selected tabs)
ACCENT_HOVER = "#2563eb"
ACCENT_ALT   = "#7c3aed"   # purple accent (export buttons)
ACCENT_ALT_H = "#6d28d9"
GREEN        = "#059669"
GREEN_HOVER  = "#047857"
DISABLED_BG  = "#4b5563"
DISABLED_FG  = "#9ca3af"

TABLE_BG     = "#272738"
TABLE_GRID   = "#3b3b54"
HEADER_SEC   = "#2a2a3d"   # table header sections

SCROLLBAR_BG = "#272738"
SCROLLBAR_FG = "#3b3b54"

# ── Page header helpers (used inline per-page) ────────────────────────────────
PAGE_HEADER_BG = BG_HEADER
PAGE_HEADER_FG = TEXT_PRIMARY
IMG_BORDER_STYLE = f"border:1px solid {BORDER}; background:{BG_SURFACE};"


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLESHEET  – applied once on the top-level QWidget in main.py
# ══════════════════════════════════════════════════════════════════════════════
DARK_STYLESHEET = f"""
    /* ── Base ──────────────────────────────────────────────────────── */
    QWidget {{
        background-color: {BG_BASE};
        color: {TEXT_PRIMARY};
        font-family: "Segoe UI", sans-serif;
        font-size: 13px;
    }}

    /* ── Tabs (top-level and nested) ───────────────────────────────── */
    QTabWidget::pane {{
        border: none;
        background-color: {BG_BASE};
    }}
    QTabBar::tab {{
        background: {BG_SURFACE};
        color: {TEXT_SECONDARY};
        padding: 10px 20px;
        margin-right: 2px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-size: 13px;
    }}
    QTabBar::tab:selected {{
        background: {ACCENT};
        color: #ffffff;
        font-weight: bold;
    }}
    QTabBar::tab:hover:!selected {{
        background: {BORDER};
    }}

    /* ── Labels ────────────────────────────────────────────────────── */
    QLabel {{
        color: {TEXT_PRIMARY};
        background: transparent;
    }}

    /* ── GroupBox ───────────────────────────────────────────────────── */
    QGroupBox {{
        color: {TEXT_HEADING};
        background-color: {BG_SURFACE};
        border: 1px solid {BORDER};
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 6px;
        font-weight: bold;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: {TEXT_HEADING};
    }}

    /* ── Input widgets ─────────────────────────────────────────────── */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {{
        background: {BG_INPUT};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 3px 6px;
        font-family: "Segoe UI", sans-serif;
        font-size: 13px;
        selection-background-color: {ACCENT};
        selection-color: #ffffff;
    }}
    QComboBox QAbstractItemView {{
        background: {BG_INPUT};
        color: {TEXT_PRIMARY};
        selection-background-color: {ACCENT};
        selection-color: #ffffff;
        border: 1px solid {BORDER};
    }}
    QComboBox::drop-down {{
        border: none;
    }}

    /* ── Tables ────────────────────────────────────────────────────── */
    QTableWidget {{
        background: {TABLE_BG};
        color: {TEXT_PRIMARY};
        gridline-color: {TABLE_GRID};
        border: 1px solid {BORDER};
    }}
    QHeaderView::section {{
        background: {HEADER_SEC};
        color: {TEXT_HEADING};
        font-weight: bold;
        border: 1px solid {BORDER};
        padding: 4px;
    }}

    /* ── Radio / Check ─────────────────────────────────────────────── */
    QRadioButton, QCheckBox {{
        color: {TEXT_PRIMARY};
        background: transparent;
    }}
    QRadioButton::indicator, QCheckBox::indicator {{
        width: 14px;
        height: 14px;
    }}

    /* ── Buttons (base – overridden per-button via helpers) ─────── */
    QPushButton {{
        font-family: "Segoe UI", sans-serif;
        font-size: 13px;
        color: {TEXT_PRIMARY};
        background: {BG_SURFACE};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 4px 10px;
    }}
    QPushButton:hover {{
        background: {BORDER};
    }}

    /* ── Progress bar ──────────────────────────────────────────────── */
    QProgressBar {{
        background: {BG_INPUT};
        border: 1px solid {BORDER};
        border-radius: 4px;
        text-align: center;
        color: {TEXT_PRIMARY};
    }}
    QProgressBar::chunk {{
        background: {ACCENT};
        border-radius: 3px;
    }}

    /* ── Scroll areas ──────────────────────────────────────────────── */
    QScrollArea {{
        background: {BG_BASE};
        border: none;
    }}
    QScrollBar:vertical {{
        background: {SCROLLBAR_BG};
        width: 10px;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical {{
        background: {SCROLLBAR_FG};
        border-radius: 5px;
        min-height: 20px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: {SCROLLBAR_BG};
        height: 10px;
        border-radius: 5px;
    }}
    QScrollBar::handle:horizontal {{
        background: {SCROLLBAR_FG};
        border-radius: 5px;
        min-width: 20px;
    }}

    /* ── Splitter handle ───────────────────────────────────────────── */
    QSplitter::handle {{
        background: {BORDER};
    }}

    /* ── Tooltip ───────────────────────────────────────────────────── */
    QToolTip {{
        background: {BG_SURFACE};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        padding: 4px;
    }}
"""


# ══════════════════════════════════════════════════════════════════════════════
# SHARED WIDGET HELPERS  – imported by each page
# ══════════════════════════════════════════════════════════════════════════════
def _section(title: str) -> QGroupBox:
    """Styled QGroupBox section card. Inherits colours from the global stylesheet."""
    box = QGroupBox(title)
    box.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
    return box


def _bold_label(text: str, size: int = 10) -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(QFont("Segoe UI", size, QFont.Weight.Bold))
    lbl.setStyleSheet(f"color: {TEXT_HEADING};")
    return lbl


def _h_line() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setStyleSheet(f"color: {BORDER};")
    return line


def _primary_btn(text: str, color: str = ACCENT, hover: str = ACCENT_HOVER) -> QPushButton:
    btn = QPushButton(text)
    btn.setFixedHeight(38)
    btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
    btn.setStyleSheet(
        f"QPushButton {{ background:{color}; color:#ffffff; border-radius:6px; border:none; }}"
        f"QPushButton:hover {{ background:{hover}; }}"
        f"QPushButton:disabled {{ background:{DISABLED_BG}; color:{DISABLED_FG}; }}"
    )
    return btn
