import sys
from PyQt6.QtWidgets import QApplication, QWidget, QTabWidget, QVBoxLayout

from styles import DARK_STYLESHEET
from pages.ClusteringPage import ClusteringPage
from pages.KrigingPage import KrigingPage
from pages.NextDigPage import NextDigPage
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
        self.gbdt_page = GradientBoostedDecisionTreePage()
        self.next_dig_page = NextDigPage()
        self.tabs.addTab(self.gbdt_page, "Gradient Boosted Decision Tree")
        self.tabs.addTab(ClusteringPage(), "Clustering with HDBSCAN + Autoencoders")
        self.tabs.addTab(KrigingPage(), "Kriging")
        self.tabs.addTab(self.next_dig_page, "Next Dig")

        self.gbdt_page.send_to_next_dig.connect(self._on_send_to_next_dig)

        layout.addWidget(self.tabs)

    def _on_send_to_next_dig(self, df, source_name):
        self.next_dig_page.load_dataframe(df, source_name)
        self.tabs.setCurrentWidget(self.next_dig_page)

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