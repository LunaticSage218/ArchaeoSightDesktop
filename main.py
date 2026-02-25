import sys
from PyQt6.QtWidgets import QApplication, QWidget, QTabWidget, QVBoxLayout

from pages.ClusteringPage import ClusteringPage
from pages.KrigingPage import KrigingPage
from pages.GradientBoostedDecisionTreePage import GradientBoostedDecisionTreePage

# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ArchaeoSight Desktop")
        self.resize(900, 600)
        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.addTab(GradientBoostedDecisionTreePage(), "Gradient Boosted Decision Tree")
        self.tabs.addTab(ClusteringPage(), "Clustering with HDBSCAN + Autoencoders")
        self.tabs.addTab(KrigingPage(), "Kriging")

        layout.addWidget(self.tabs)

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow, QWidget#MainWindow {
                background-color: #f9fafb;
            }
            QTabWidget::pane {
                border: none;
                background-color: #f9fafb;
            }
            QTabBar::tab {
                background: #e5e7eb;
                color: #374151;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background: #1d4ed8;
                color: #ffffff;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background: #d1d5db;
            }
            QLabel {
                color: #111827;
                background: transparent;
            }
            QGroupBox {
                color: #111827;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
                background: #ffffff;
                color: #111827;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 3px 6px;
                font-family: "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QTableWidget {
                background: #ffffff;
                color: #111827;
                gridline-color: #e5e7eb;
            }
            QHeaderView::section {
                background: #f3f4f6;
                color: #374151;
                font-weight: bold;
                border: 1px solid #e5e7eb;
                padding: 4px;
            }
            QRadioButton {
                color: #111827;
                background: transparent;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
            }
            QPushButton {
                font-family: "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                color: #111827;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
                color: #111827;
            }
        """)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()