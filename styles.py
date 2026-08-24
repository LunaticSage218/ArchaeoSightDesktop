"""
Central dark-theme stylesheet and shared widget helpers for ArchaeoSight Desktop.

Import DARK_STYLESHEET in main.py and apply it once to the top-level widget.
Import the helper functions (_section, _bold_label, _primary_btn, _h_line,
_cell, _blend, PAGE_HEADER_BG, PAGE_HEADER_FG, IMG_BORDER_STYLE) in each page
module.

Table cells with a custom background must be built with _cell(), which pairs
the background with a readable foreground — see the CELL_* tokens below.
"""

from PyQt6.QtWidgets import QGroupBox, QLabel, QPushButton, QFrame, QTableWidgetItem
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCore import Qt

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

# ── Table cell highlight colours ──────────────────────────────────────────────
# Qt does NOT derive a text colour from an item's background, so a cell painted
# with a light background keeps the global light-on-dark text and becomes
# invisible until it is selected. Every background below is therefore paired
# with a foreground, and cells are built with _cell() so the two can't drift
# apart. Never call QTableWidgetItem.setBackground() without also setting the
# foreground.
CELL_INFO_BG = "#1e3a5f"   # highlighted prediction / result columns
CELL_INFO_FG = "#dbeafe"
CELL_GOOD_BG = "#14532d"   # confusion-matrix diagonal (correct)
CELL_GOOD_FG = "#dcfce7"
CELL_BAD_BG  = "#7f1d1d"   # confusion-matrix off-diagonal (misclassified)
CELL_BAD_FG  = "#fee2e2"
CELL_WARN_BG = "#78350f"   # noise / caution rows
CELL_WARN_FG = "#fef3c7"

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
        selection-background-color: {ACCENT};
        selection-color: #ffffff;
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


def _cell(text: str, bg: str | None = None, fg: str | None = None,
          center: bool = False) -> QTableWidgetItem:
    """Build a table cell that stays readable on the dark theme.

    Setting only a background is the bug this exists to prevent: Qt leaves the
    text at the stylesheet's near-white, so a light background renders as
    white-on-white and only becomes legible once the cell is selected. Passing
    `bg` here always sets a foreground too (`fg`, or TEXT_PRIMARY).
    """
    item = QTableWidgetItem(text)
    if center:
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    if bg is not None:
        item.setBackground(QColor(bg))
        item.setForeground(QColor(fg or TEXT_PRIMARY))
    return item


def _blend(c1: str, c2: str, t: float) -> str:
    """Linear blend of two hex colours; t=0 gives c1, t=1 gives c2.

    Used for intensity ramps (e.g. confusion-matrix cells) so a stronger value
    reads as a deeper tint of the same hue instead of a lighter one — keeping
    the paired foreground readable across the whole range.
    """
    t = min(max(t, 0.0), 1.0)
    a = QColor(c1)
    b = QColor(c2)
    return QColor(
        round(a.red()   + (b.red()   - a.red())   * t),
        round(a.green() + (b.green() - a.green()) * t),
        round(a.blue()  + (b.blue()  - a.blue())  * t),
    ).name()


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
