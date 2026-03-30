import sys
from PyQt6.QtWidgets import QApplication, QWidget, QTabWidget, QVBoxLayout

from pages.ClusteringPage import ClusteringPage
from pages.KrigingPage import KrigingPage
from pages.GradientBoostedDecisionTreePage import GradientBoostedDecisionTreePage
from styles import GLOBAL_STYLESHEET

# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ArchaeoSight Desktop")
        self.resize(900, 600)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.addTab(GradientBoostedDecisionTreePage(), "Gradient Boosted Decision Tree")
        self.tabs.addTab(ClusteringPage(), "Clustering with HDBSCAN + Autoencoders")
        self.tabs.addTab(KrigingPage(), "Kriging")

        layout.addWidget(self.tabs)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(GLOBAL_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()