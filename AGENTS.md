# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

ArchaeoSight Desktop is a PyQt6 desktop application for archaeological pXRF (portable X-ray fluorescence) data analysis. It provides ML-based material classification and clustering tools through a tabbed GUI.

## Build and Run Commands

```pwsh
# Install dependencies (use a virtualenv)
pip install -r requirements.txt

# Run the application
python main.py
```

There are no tests, linter, or type-checker configured in this project.

## Architecture

### Entry Point

`main.py` — Creates the `QApplication`, sets the Fusion style, and displays `MainWindow`, which holds a `QTabWidget` with three page tabs.

### Pages (`pages/`)

Each page is a self-contained `QWidget` subclass registered as a tab in `MainWindow`:

- **`GradientBoostedDecisionTreePage.py`** — Supervised classification using sklearn `GradientBoostingClassifier`. Contains `TrainTab` (train + export model) and `TestTab` (load model + predict on new data). Supports saving models as pickle or ONNX. Uses `TrainWorker` and `TestWorker` for background processing.

- **`ClusteringPage.py`** — Unsupervised clustering pipeline: autoencoder (TensorFlow/Keras) → latent space → PCA → HDBSCAN. Contains `TrainTab` (full pipeline) and `ApplyTab` (apply trained models to new data). Uses `TrainWorker` and `ApplyWorker` for background processing. Also trains baseline models (PCA+HDBSCAN, KMeans) and optional RandomForest classifiers when labels are available.

- **`KrigingPage.py`** — Spatial interpolation using PyKrige (Ordinary and Universal Kriging). Contains `InterpolationTab` (fit variogram + generate grid + plots) and `PredictTab` (apply saved model to new coordinates). Uses `KrigingWorker` and `PredictWorker` for background processing. Offers configurable variogram model, anisotropy, grid resolution, custom bounds, and leave-one-out cross-validation. Saves `kriging_model.pkl`, `kriging_config.json`, `interpolated_grid.csv`, and plot PNGs.

### Background Processing Pattern

All heavy ML work runs in `QThread` via `QObject`-based workers (`TrainWorker`, `ApplyWorker`, `TestWorker`). Workers communicate with the UI through `pyqtSignal`:
- `log(str)` — progress messages appended to a `QTextEdit`
- `finished(dict)` — results payload
- `error(str)` — traceback string

The standard wiring pattern is: create `QThread` + worker, `moveToThread`, connect `started → run`, connect signals, then `start()`.

### Data Conventions

- Input data is CSV or Excel (`.csv`, `.xlsx`, `.xls`).
- Feature columns are auto-detected by matching column names against `PERIODIC_TABLE_ELEMENTS` (a set of all element symbols). This set is duplicated in both `ClusteringPage.py` and `GradientBoostedDecisionTreePage.py`.
- Non-element columns are offered as label/target columns.
- Negative values are clipped to 0; NaNs are filled with 0.

### Models Directory (`models/`)

Gitignored. Stores trained artifacts: `.keras` models (autoencoder, encoder), `.pkl` files (scaler, HDBSCAN, PCA, classifiers, KMeans, kriging), `.npy` latent arrays, `.csv` cluster assignments/grids, `.png` plots, and `.json` baselines/configs. The user selects the output folder and an optional model folder name (subfolder) at training time.

## Key Dependencies

- **PyQt6** — GUI framework
- **TensorFlow/Keras** — Autoencoder in the clustering pipeline
- **scikit-learn** — GradientBoosting, RandomForest, PCA, StandardScaler, KMeans, metrics
- **hdbscan** — Density-based clustering (with `approximate_predict` for new data)
- **pykrige** — Ordinary and Universal Kriging for spatial interpolation
- **skl2onnx / onnx** — Optional ONNX export for GBDT models
- **matplotlib** — Training loss, cluster, variogram, and interpolation plots (uses `Agg` backend)

## Style Conventions

- All UI styling is centralized in `styles.py`, which defines a dark-mode Tailwind-slate colour palette and a `GLOBAL_STYLESHEET` applied once via `app.setStyleSheet()` in `main.py`.
- Shared widget helpers (`section()`, `bold_label()`, `h_line()`, `primary_btn()`) live in `styles.py` and are imported by each page.
- Font family is "Segoe UI" throughout; monospace log areas use "Consolas".
- Color palette (dark mode): background #0f172a, surface #1e293b, panel #334155, text #f1f5f9, accent blue #3b82f6, accent green #10b981, borders #475569.
