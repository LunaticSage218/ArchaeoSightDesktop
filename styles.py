"""Centralized dark-mode stylesheet and UI helpers for ArchaeoSight Desktop."""

from PyQt6.QtWidgets import QGroupBox, QLabel, QPushButton, QFrame
from PyQt6.QtGui import QFont

# ── Dark-mode colour palette (Tailwind Slate) ──────────────────────────────
BG           = "#0f172a"
SURFACE      = "#1e293b"
PANEL        = "#334155"
TEXT         = "#f1f5f9"
TEXT_DIM     = "#94a3b8"
ACCENT       = "#3b82f6"
ACCENT_HOVER = "#2563eb"
GREEN        = "#10b981"
GREEN_HOVER  = "#059669"
BORDER       = "#475569"
INPUT_BG     = "#1e293b"
HEADER_BG    = "#0c1526"
DISABLED_BG  = "#475569"
DISABLED_TXT = "#94a3b8"

# Pre-built style string for image-display QLabels
IMAGE_STYLE = f"border:1px solid {BORDER}; background:{SURFACE};"

# ── Global stylesheet (applied once on QApplication) ───────────────────────
GLOBAL_STYLESHEET = f"""
    QWidget {{
        background-color: {BG};
        color: {TEXT};
        font-family: "Segoe UI", sans-serif;
    }}

    /* ── Tabs ─────────────────────────────────────────────── */
    QTabWidget::pane {{
        border: none;
        background-color: {BG};
    }}
    QTabBar::tab {{
        background: {PANEL};
        color: {TEXT_DIM};
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

    /* ── Labels / GroupBoxes ──────────────────────────────── */
    QLabel {{
        color: {TEXT};
        background: transparent;
    }}
    QGroupBox {{
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 6px;
        font-weight: bold;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
        color: {TEXT};
    }}

    /* ── Inputs ───────────────────────────────────────────── */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {{
        background: {INPUT_BG};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 3px 6px;
        font-family: "Segoe UI", sans-serif;
        font-size: 13px;
    }}
    QLineEdit:focus, QComboBox:focus,
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {ACCENT};
    }}

    /* ── Tables ───────────────────────────────────────────── */
    QTableWidget {{
        background: {SURFACE};
        color: {TEXT};
        gridline-color: {BORDER};
    }}
    QHeaderView::section {{
        background: {PANEL};
        color: {TEXT};
        font-weight: bold;
        border: 1px solid {BORDER};
        padding: 4px;
    }}

    /* ── Radio / Check ────────────────────────────────────── */
    QRadioButton, QCheckBox {{
        color: {TEXT};
        background: transparent;
    }}
    QRadioButton::indicator, QCheckBox::indicator {{
        width: 14px;
        height: 14px;
    }}

    /* ── Buttons ──────────────────────────────────────────── */
    QPushButton {{
        font-family: "Segoe UI", sans-serif;
        font-size: 13px;
        background: {PANEL};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 4px 12px;
    }}
    QPushButton:hover {{
        background: {BORDER};
    }}

    /* ── Scroll / Splitter ────────────────────────────────── */
    QScrollArea {{
        background: {BG};
        border: none;
    }}
    QScrollBar:vertical {{
        background: {SURFACE}; width: 10px; border-radius: 5px;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER}; border-radius: 5px; min-height: 20px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{
        background: {SURFACE}; height: 10px; border-radius: 5px;
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER}; border-radius: 5px; min-width: 20px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    QSplitter::handle {{
        background: {BORDER};
    }}

    /* ── Progress bar ─────────────────────────────────────── */
    QProgressBar {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 4px;
        text-align: center;
        color: {TEXT};
    }}
    QProgressBar::chunk {{
        background: {ACCENT};
        border-radius: 3px;
    }}

    /* ── ComboBox dropdown ────────────────────────────────── */
    QComboBox QAbstractItemView {{
        background: {SURFACE};
        color: {TEXT};
        selection-background-color: {ACCENT};
        border: 1px solid {BORDER};
    }}
    QComboBox::drop-down {{
        border: none;
    }}
"""


# ── Reusable widget helpers ────────────────────────────────────────────────

def section(title: str) -> QGroupBox:
    """Styled QGroupBox used as a section container."""
    box = QGroupBox(title)
    box.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
    return box


def bold_label(text: str, size: int = 10) -> QLabel:
    """Bold QLabel."""
    lbl = QLabel(text)
    lbl.setFont(QFont("Segoe UI", size, QFont.Weight.Bold))
    return lbl


def h_line() -> QFrame:
    """Horizontal separator line."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


def primary_btn(text: str, color: str = ACCENT,
                hover: str = ACCENT_HOVER) -> QPushButton:
    """Prominent action button."""
    btn = QPushButton(text)
    btn.setFixedHeight(38)
    btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
    btn.setStyleSheet(
        f"QPushButton {{ background:{color}; color:white;"
        f"  border:none; border-radius:6px; }}"
        f"QPushButton:hover {{ background:{hover}; }}"
        f"QPushButton:disabled {{ background:{DISABLED_BG};"
        f"  color:{DISABLED_TXT}; }}"
    )
    return btn
