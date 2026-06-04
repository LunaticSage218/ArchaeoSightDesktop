import sys
from PyQt6.QtWidgets import QApplication, QWidget, QTabWidget, QVBoxLayout

from styles import DARK_STYLESHEET
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
        self.setStyleSheet(DARK_STYLESHEET)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()