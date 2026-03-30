import os
import io
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QTabWidget, QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QScrollArea, QProgressBar, QMessageBox, QHeaderView,
    QCheckBox, QTextEdit, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QPixmap

from styles import (
    section, bold_label, h_line, primary_btn,
    HEADER_BG, IMAGE_STYLE, GREEN, GREEN_HOVER, SURFACE, BORDER,
)

VARIOGRAM_MODELS = [
    "linear", "power", "gaussian", "spherical", "exponential", "hole-effect",
]

DRIFT_TERMS = ["regional_linear", "point_log"]


# ══════════════════════════════════════════════════════════════════════════════
# KRIGING WORKER
# ══════════════════════════════════════════════════════════════════════════════
class KrigingWorker(QObject):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    log      = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.p = params

    def run(self):
        try:
            from pykrige.ok import OrdinaryKriging
            from pykrige.uk import UniversalKriging
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            p = self.p
            out_dir = Path(p["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)

            # ── Step 1: Load data ─────────────────────────────────────────
            self.log.emit("Step 1: Loading data…")
            ext = os.path.splitext(p["csv_path"])[1].lower()
            try:
                df = (pd.read_excel(p["csv_path"])
                      if ext in (".xlsx", ".xls")
                      else pd.read_csv(p["csv_path"]))
            except Exception:
                df = pd.read_csv(p["csv_path"], on_bad_lines="skip")

            x = pd.to_numeric(df[p["x_col"]], errors="coerce").fillna(0).values
            y = pd.to_numeric(df[p["y_col"]], errors="coerce").fillna(0).values
            z = pd.to_numeric(df[p["z_col"]], errors="coerce").fillna(0).values
            self.log.emit(f"  {len(x)} data points loaded.")

            # ── Step 2: Variogram parameters ──────────────────────────────
            vario_params = None
            if not p["auto_fit"]:
                vario_params = {
                    "sill":   p["sill"],
                    "range":  p["range"],
                    "nugget": p["nugget"],
                }

            # ── Step 3: Fit kriging model ─────────────────────────────────
            self.log.emit(f"Step 2: Fitting {p['kriging_type']} kriging "
                          f"({p['variogram_model']} variogram)…")
            kw = dict(
                variogram_model=p["variogram_model"],
                variogram_parameters=vario_params,
                nlags=p["nlags"],
                weight=p["weight"],
                anisotropy_scaling=p["anisotropy_scaling"],
                anisotropy_angle=p["anisotropy_angle"],
                exact_values=p["exact_values"],
                enable_plotting=False,
            )
            if p["kriging_type"] == "ordinary":
                krig = OrdinaryKriging(x, y, z, **kw)
            else:
                krig = UniversalKriging(x, y, z,
                                        drift_terms=[p["drift_term"]], **kw)

            fitted = list(krig.variogram_model_parameters)
            self.log.emit(f"  Fitted variogram parameters: {fitted}")

            # ── Step 4: Build interpolation grid ──────────────────────────
            self.log.emit("Step 3: Creating interpolation grid…")
            if p["custom_bounds"]:
                gridx = np.linspace(p["xmin"], p["xmax"], p["nx"])
                gridy = np.linspace(p["ymin"], p["ymax"], p["ny"])
            else:
                margin = 0.05
                xr = (x.max() - x.min()) or 1.0
                yr = (y.max() - y.min()) or 1.0
                gridx = np.linspace(x.min() - margin * xr,
                                    x.max() + margin * xr, p["nx"])
                gridy = np.linspace(y.min() - margin * yr,
                                    y.max() + margin * yr, p["ny"])
            self.log.emit(f"  Grid: {p['nx']}×{p['ny']} = "
                          f"{p['nx'] * p['ny']} cells.")

            z_pred, ss_pred = krig.execute("grid", gridx, gridy)

            # ── Step 5: Cross-validation (optional) ───────────────────────
            cv_results = {}
            if p["cross_validate"]:
                self.log.emit("Step 4: Leave-one-out cross-validation…")
                errors = []
                for i in range(len(x)):
                    xi = np.delete(x, i)
                    yi = np.delete(y, i)
                    zi = np.delete(z, i)
                    try:
                        if p["kriging_type"] == "ordinary":
                            k_cv = OrdinaryKriging(
                                xi, yi, zi,
                                variogram_model=p["variogram_model"],
                                enable_plotting=False,
                            )
                        else:
                            k_cv = UniversalKriging(
                                xi, yi, zi,
                                variogram_model=p["variogram_model"],
                                drift_terms=[p["drift_term"]],
                                enable_plotting=False,
                            )
                        zh, _ = k_cv.execute("points",
                                             np.array([x[i]]),
                                             np.array([y[i]]))
                        errors.append(z[i] - zh.flatten()[0])
                    except Exception:
                        pass
                if errors:
                    errors = np.array(errors)
                    rmse = float(np.sqrt(np.mean(errors ** 2)))
                    mae  = float(np.mean(np.abs(errors)))
                    me   = float(np.mean(errors))
                    cv_results = {"rmse": rmse, "mae": mae, "me": me,
                                  "n_valid": len(errors)}
                    self.log.emit(f"  LOO-CV  RMSE={rmse:.4f}  "
                                  f"MAE={mae:.4f}  ME={me:.4f}")

            # ── Step 6: Plots ─────────────────────────────────────────────
            self.log.emit("Step 5: Generating plots…")

            # Variogram
            fig_v, ax_v = plt.subplots(figsize=(8, 5))
            ax_v.scatter(krig.lags, krig.semivariance,
                         c="#3b82f6", s=40, zorder=3, label="Experimental")
            lags_fine = np.linspace(0, krig.lags[-1] * 1.1, 300)
            sv_model = krig.variogram_function(
                krig.variogram_model_parameters, lags_fine)
            ax_v.plot(lags_fine, sv_model, "r-", lw=2,
                      label=f"Fitted ({p['variogram_model']})")
            ax_v.set_xlabel("Lag Distance")
            ax_v.set_ylabel("Semivariance")
            ax_v.set_title("Experimental Variogram & Fitted Model",
                           fontweight="bold")
            ax_v.legend()
            ax_v.grid(True, alpha=0.3)
            fig_v.tight_layout()
            vario_path = str(out_dir / "variogram.png")
            fig_v.savefig(vario_path, dpi=150)
            plt.close(fig_v)

            # Interpolation heatmap
            fig_i, ax_i = plt.subplots(figsize=(9, 7))
            im = ax_i.pcolormesh(gridx, gridy, z_pred,
                                 shading="auto", cmap="viridis")
            ax_i.scatter(x, y, c="red", s=15, edgecolors="white",
                         linewidths=0.5, label="Data points", zorder=3)
            fig_i.colorbar(im, ax=ax_i, label=p["z_col"])
            ax_i.set_xlabel(p["x_col"])
            ax_i.set_ylabel(p["y_col"])
            ax_i.set_title(f"Kriging Interpolation — {p['z_col']}",
                           fontweight="bold")
            ax_i.legend()
            fig_i.tight_layout()
            interp_path = str(out_dir / "interpolation.png")
            fig_i.savefig(interp_path, dpi=150)
            plt.close(fig_i)

            # Variance heatmap
            fig_s, ax_s = plt.subplots(figsize=(9, 7))
            im_s = ax_s.pcolormesh(gridx, gridy, ss_pred,
                                   shading="auto", cmap="magma")
            ax_s.scatter(x, y, c="cyan", s=15, edgecolors="white",
                         linewidths=0.5, label="Data points", zorder=3)
            fig_s.colorbar(im_s, ax=ax_s, label="Variance")
            ax_s.set_xlabel(p["x_col"])
            ax_s.set_ylabel(p["y_col"])
            ax_s.set_title("Kriging Variance (Uncertainty)",
                           fontweight="bold")
            ax_s.legend()
            fig_s.tight_layout()
            var_path = str(out_dir / "variance.png")
            fig_s.savefig(var_path, dpi=150)
            plt.close(fig_s)

            # ── Step 7: Save artefacts ────────────────────────────────────
            self.log.emit("Step 6: Saving artefacts…")
            grid_xx, grid_yy = np.meshgrid(gridx, gridy)
            grid_df = pd.DataFrame({
                "X": grid_xx.ravel(),
                "Y": grid_yy.ravel(),
                "Z_predicted": z_pred.data.ravel()
                    if hasattr(z_pred, "data") else z_pred.ravel(),
                "Variance": ss_pred.data.ravel()
                    if hasattr(ss_pred, "data") else ss_pred.ravel(),
            })
            grid_df.to_csv(out_dir / "interpolated_grid.csv", index=False)

            with open(out_dir / "kriging_model.pkl", "wb") as f:
                pickle.dump(krig, f)

            config = {
                "kriging_type":     p["kriging_type"],
                "variogram_model":  p["variogram_model"],
                "variogram_params": fitted,
                "x_col": p["x_col"],
                "y_col": p["y_col"],
                "z_col": p["z_col"],
                "nlags": p["nlags"],
                "n_samples": len(x),
            }
            if p["kriging_type"] == "universal":
                config["drift_term"] = p["drift_term"]
            with open(out_dir / "kriging_config.json", "w") as f:
                json.dump(config, f, indent=2)

            z_flat = z_pred.data.ravel() if hasattr(z_pred, "data") \
                     else z_pred.ravel()
            ss_flat = ss_pred.data.ravel() if hasattr(ss_pred, "data") \
                      else ss_pred.ravel()

            results = {
                "variogram_plot": vario_path,
                "interp_plot":    interp_path,
                "variance_plot":  var_path,
                "grid_shape":     (p["ny"], p["nx"]),
                "z_mean":  float(np.nanmean(z_flat)),
                "z_std":   float(np.nanstd(z_flat)),
                "z_min":   float(np.nanmin(z_flat)),
                "z_max":   float(np.nanmax(z_flat)),
                "var_mean": float(np.nanmean(ss_flat)),
                "n_samples": len(x),
                "variogram_params": fitted,
                "cv": cv_results,
            }

            self.log.emit("Pipeline complete!")
            self.finished.emit(results)

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT WORKER
# ══════════════════════════════════════════════════════════════════════════════
class PredictWorker(QObject):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    log      = pyqtSignal(str)

    def __init__(self, model_dir: str, csv_path: str):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.csv_path  = csv_path

    def run(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            d = self.model_dir
            self.log.emit("Loading kriging model…")
            with open(d / "kriging_model.pkl", "rb") as f:
                krig = pickle.load(f)
            with open(d / "kriging_config.json", "r") as f:
                config = json.load(f)
            self.log.emit("  Model loaded.")

            x_col = config["x_col"]
            y_col = config["y_col"]
            z_col = config["z_col"]

            self.log.emit("Loading new data…")
            ext = os.path.splitext(self.csv_path)[1].lower()
            df = (pd.read_excel(self.csv_path)
                  if ext in (".xlsx", ".xls")
                  else pd.read_csv(self.csv_path))

            if x_col not in df.columns or y_col not in df.columns:
                raise ValueError(
                    f"Data must contain '{x_col}' and '{y_col}' columns.")

            x_new = pd.to_numeric(df[x_col], errors="coerce").fillna(0).values
            y_new = pd.to_numeric(df[y_col], errors="coerce").fillna(0).values

            self.log.emit(f"Predicting for {len(x_new)} points…")
            z_hat, ss_hat = krig.execute("points", x_new, y_new)
            z_hat = z_hat.flatten()
            ss_hat = ss_hat.flatten()

            result_df = df.copy()
            result_df[f"{z_col}_predicted"] = z_hat
            result_df["Variance"]           = ss_hat
            result_df["StdDev"]             = np.sqrt(np.maximum(ss_hat, 0))

            # Scatter plot
            fig, ax = plt.subplots(figsize=(9, 7))
            sc = ax.scatter(x_new, y_new, c=z_hat, cmap="viridis",
                            s=30, edgecolors="black", linewidths=0.3)
            fig.colorbar(sc, ax=ax, label=f"{z_col} (predicted)")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Kriging Predictions — {z_col}", fontweight="bold")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            buf.seek(0)
            plot_bytes = buf.read()

            self.log.emit(f"Done. {len(df)} predictions generated.")
            self.finished.emit({
                "result_df":   result_df,
                "plot_bytes":  plot_bytes,
                "z_col":       z_col,
            })

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# INTERPOLATION TAB
# ══════════════════════════════════════════════════════════════════════════════
class InterpolationTab(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = self._worker = None
        self._build_ui()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ── Left controls ───────────────────────────────────────────────
        scroll_left = QScrollArea()
        scroll_left.setWidgetResizable(True)
        scroll_left_widget = QWidget()
        lv = QVBoxLayout(scroll_left_widget)
        lv.setSpacing(10)
        lv.setContentsMargins(4, 4, 4, 4)

        # -- 1. Data file --
        fb = section("1. Data File")
        fl = QVBoxLayout(fb)
        fr = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("CSV or Excel…")
        self.file_edit.setReadOnly(True)
        bb = QPushButton("Browse")
        bb.setFixedWidth(70)
        bb.clicked.connect(self._browse_file)
        fr.addWidget(self.file_edit)
        fr.addWidget(bb)
        fl.addLayout(fr)

        fl.addWidget(QLabel("X coordinate column:"))
        self.x_combo = QComboBox()
        self.x_combo.setEnabled(False)
        fl.addWidget(self.x_combo)

        fl.addWidget(QLabel("Y coordinate column:"))
        self.y_combo = QComboBox()
        self.y_combo.setEnabled(False)
        fl.addWidget(self.y_combo)

        fl.addWidget(QLabel("Z value column (to interpolate):"))
        self.z_combo = QComboBox()
        self.z_combo.setEnabled(False)
        fl.addWidget(self.z_combo)
        lv.addWidget(fb)

        # -- 2. Kriging options --
        kb = section("2. Kriging Options")
        kf = QFormLayout(kb)
        kf.setSpacing(6)

        self.type_combo = QComboBox()
        self.type_combo.addItems(["ordinary", "universal"])
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        kf.addRow("Kriging type:", self.type_combo)

        self.drift_combo = QComboBox()
        self.drift_combo.addItems(DRIFT_TERMS)
        self.drift_combo.setEnabled(False)
        kf.addRow("Drift term:", self.drift_combo)

        self.vario_combo = QComboBox()
        self.vario_combo.addItems(VARIOGRAM_MODELS)
        self.vario_combo.setCurrentText("spherical")
        kf.addRow("Variogram model:", self.vario_combo)

        self.nlags_spin = QSpinBox()
        self.nlags_spin.setRange(2, 100)
        self.nlags_spin.setValue(6)
        kf.addRow("nlags:", self.nlags_spin)

        self.weight_chk = QCheckBox("Weight variogram")
        kf.addRow(self.weight_chk)

        self.exact_chk = QCheckBox("Exact values")
        self.exact_chk.setChecked(True)
        kf.addRow(self.exact_chk)
        lv.addWidget(kb)

        # -- 3. Variogram parameters --
        vb = section("3. Variogram Parameters")
        vl = QVBoxLayout(vb)
        self.auto_fit_chk = QCheckBox("Auto-fit (recommended)")
        self.auto_fit_chk.setChecked(True)
        self.auto_fit_chk.toggled.connect(self._on_autofit_toggled)
        vl.addWidget(self.auto_fit_chk)

        vf = QFormLayout()
        self.sill_spin = QDoubleSpinBox()
        self.sill_spin.setRange(0.0001, 1e9)
        self.sill_spin.setDecimals(4)
        self.sill_spin.setValue(1.0)
        self.sill_spin.setEnabled(False)
        vf.addRow("Sill:", self.sill_spin)

        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(0.0001, 1e9)
        self.range_spin.setDecimals(4)
        self.range_spin.setValue(1.0)
        self.range_spin.setEnabled(False)
        vf.addRow("Range:", self.range_spin)

        self.nugget_spin = QDoubleSpinBox()
        self.nugget_spin.setRange(0.0, 1e9)
        self.nugget_spin.setDecimals(4)
        self.nugget_spin.setValue(0.0)
        self.nugget_spin.setEnabled(False)
        vf.addRow("Nugget:", self.nugget_spin)
        vl.addLayout(vf)
        lv.addWidget(vb)

        # -- 4. Anisotropy --
        ab = section("4. Anisotropy")
        af = QFormLayout(ab)
        self.aniso_angle_spin = QDoubleSpinBox()
        self.aniso_angle_spin.setRange(0.0, 360.0)
        self.aniso_angle_spin.setDecimals(1)
        self.aniso_angle_spin.setValue(0.0)
        af.addRow("Angle (°):", self.aniso_angle_spin)

        self.aniso_scale_spin = QDoubleSpinBox()
        self.aniso_scale_spin.setRange(1.0, 100.0)
        self.aniso_scale_spin.setDecimals(2)
        self.aniso_scale_spin.setValue(1.0)
        af.addRow("Scaling:", self.aniso_scale_spin)
        lv.addWidget(ab)

        # -- 5. Grid --
        gb = section("5. Interpolation Grid")
        gf = QFormLayout(gb)
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(10, 2000)
        self.nx_spin.setValue(100)
        gf.addRow("Grid points X:", self.nx_spin)

        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(10, 2000)
        self.ny_spin.setValue(100)
        gf.addRow("Grid points Y:", self.ny_spin)

        self.custom_bounds_chk = QCheckBox("Custom grid bounds")
        self.custom_bounds_chk.toggled.connect(self._on_bounds_toggled)
        gf.addRow(self.custom_bounds_chk)

        self.xmin_spin = QDoubleSpinBox()
        self.xmin_spin.setRange(-1e9, 1e9)
        self.xmin_spin.setDecimals(2)
        self.xmin_spin.setEnabled(False)
        gf.addRow("X min:", self.xmin_spin)

        self.xmax_spin = QDoubleSpinBox()
        self.xmax_spin.setRange(-1e9, 1e9)
        self.xmax_spin.setDecimals(2)
        self.xmax_spin.setValue(100.0)
        self.xmax_spin.setEnabled(False)
        gf.addRow("X max:", self.xmax_spin)

        self.ymin_spin = QDoubleSpinBox()
        self.ymin_spin.setRange(-1e9, 1e9)
        self.ymin_spin.setDecimals(2)
        self.ymin_spin.setEnabled(False)
        gf.addRow("Y min:", self.ymin_spin)

        self.ymax_spin = QDoubleSpinBox()
        self.ymax_spin.setRange(-1e9, 1e9)
        self.ymax_spin.setDecimals(2)
        self.ymax_spin.setValue(100.0)
        self.ymax_spin.setEnabled(False)
        gf.addRow("Y max:", self.ymax_spin)
        lv.addWidget(gb)

        # -- 6. Cross-validation --
        cb = section("6. Cross-Validation")
        cl = QVBoxLayout(cb)
        self.cv_chk = QCheckBox("Leave-one-out CV (slow for large datasets)")
        cl.addWidget(self.cv_chk)
        lv.addWidget(cb)

        # -- 7. Output --
        ob = section("7. Output")
        ol = QVBoxLayout(ob)
        or_ = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Select output folder…")
        self.out_edit.setReadOnly(True)
        ob2 = QPushButton("Browse")
        ob2.setFixedWidth(70)
        ob2.clicked.connect(self._browse_out)
        or_.addWidget(self.out_edit)
        or_.addWidget(ob2)
        ol.addLayout(or_)

        ol.addWidget(QLabel("Model folder name:"))
        self.folder_name_edit = QLineEdit()
        self.folder_name_edit.setPlaceholderText("e.g. kriging_run_01")
        ol.addWidget(self.folder_name_edit)
        lv.addWidget(ob)

        # Run button
        self.run_btn = primary_btn("▶  Run Kriging")
        self.run_btn.clicked.connect(self._start)
        lv.addWidget(self.run_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        lv.addWidget(self.progress)

        lv.addStretch()
        scroll_left.setWidget(scroll_left_widget)

        left_container = QWidget()
        left_container.setFixedWidth(320)
        left_wrapper = QVBoxLayout(left_container)
        left_wrapper.setContentsMargins(0, 0, 0, 0)
        left_wrapper.addWidget(scroll_left)
        root.addWidget(left_container)

        # ── Right results ───────────────────────────────────────────────
        right_split = QSplitter(Qt.Orientation.Vertical)

        log_box = section("Pipeline Log")
        log_layout = QVBoxLayout(log_box)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9))
        self.log_edit.setFixedHeight(130)
        log_layout.addWidget(self.log_edit)
        right_split.addWidget(log_box)

        res_scroll = QScrollArea()
        res_scroll.setWidgetResizable(True)
        res_widget = QWidget()
        self.res_layout = QVBoxLayout(res_widget)
        self.res_layout.setSpacing(10)

        self.metrics_label = QLabel("Run the pipeline to see results.")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setFont(QFont("Segoe UI", 10))
        self.res_layout.addWidget(self.metrics_label)

        self.res_layout.addWidget(bold_label("Variogram"))
        self.vario_img = QLabel()
        self.vario_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vario_img.setStyleSheet(IMAGE_STYLE)
        self.vario_img.setMinimumHeight(220)
        self.res_layout.addWidget(self.vario_img)

        self.res_layout.addWidget(bold_label("Interpolation Map"))
        self.interp_img = QLabel()
        self.interp_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.interp_img.setStyleSheet(IMAGE_STYLE)
        self.interp_img.setMinimumHeight(280)
        self.res_layout.addWidget(self.interp_img)

        self.res_layout.addWidget(bold_label("Variance (Uncertainty) Map"))
        self.var_img = QLabel()
        self.var_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.var_img.setStyleSheet(IMAGE_STYLE)
        self.var_img.setMinimumHeight(280)
        self.res_layout.addWidget(self.var_img)

        self.res_layout.addStretch()
        res_scroll.setWidget(res_widget)
        right_split.addWidget(res_scroll)
        right_split.setSizes([180, 620])
        root.addWidget(right_split, 1)

    # ── Toggle helpers ──────────────────────────────────────────────────────
    def _on_type_changed(self, text):
        self.drift_combo.setEnabled(text == "universal")

    def _on_autofit_toggled(self, checked):
        manual = not checked
        self.sill_spin.setEnabled(manual)
        self.range_spin.setEnabled(manual)
        self.nugget_spin.setEnabled(manual)

    def _on_bounds_toggled(self, checked):
        self.xmin_spin.setEnabled(checked)
        self.xmax_spin.setEnabled(checked)
        self.ymin_spin.setEnabled(checked)
        self.ymax_spin.setEnabled(checked)

    # ── Browse helpers ──────────────────────────────────────────────────────
    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "",
            "Data Files (*.csv *.xlsx *.xls);;All Files (*)")
        if not path:
            return
        self.file_edit.setText(path)
        try:
            ext = os.path.splitext(path)[1].lower()
            df = (pd.read_excel(path, nrows=5)
                  if ext in (".xlsx", ".xls")
                  else pd.read_csv(path, nrows=5))
            cols = list(df.columns)
            numeric_cols = [
                c for c in cols
                if pd.to_numeric(df[c], errors="coerce").notna().any()
            ]
            for combo in (self.x_combo, self.y_combo, self.z_combo):
                combo.clear()
                combo.addItems(numeric_cols)
                combo.setEnabled(True)
            # Try to auto-select coordinate columns
            for c in numeric_cols:
                cl = c.lower()
                if "x" in cl and "coord" in cl:
                    self.x_combo.setCurrentText(c)
                elif "y" in cl and "coord" in cl:
                    self.y_combo.setCurrentText(c)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.out_edit.setText(path)

    # ── Start pipeline ──────────────────────────────────────────────────────
    def _start(self):
        csv_path = self.file_edit.text().strip()
        out_dir  = self.out_edit.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "Missing", "Select a data file.")
            return
        if not out_dir:
            QMessageBox.warning(self, "Missing", "Select an output folder.")
            return
        if not self.x_combo.currentText():
            QMessageBox.warning(self, "Missing", "Select X/Y/Z columns.")
            return

        folder_name = self.folder_name_edit.text().strip()
        if folder_name:
            out_dir = os.path.join(out_dir, folder_name)

        params = {
            "csv_path":       csv_path,
            "out_dir":        out_dir,
            "x_col":          self.x_combo.currentText(),
            "y_col":          self.y_combo.currentText(),
            "z_col":          self.z_combo.currentText(),
            "kriging_type":   self.type_combo.currentText(),
            "drift_term":     self.drift_combo.currentText(),
            "variogram_model": self.vario_combo.currentText(),
            "nlags":          self.nlags_spin.value(),
            "weight":         self.weight_chk.isChecked(),
            "exact_values":   self.exact_chk.isChecked(),
            "auto_fit":       self.auto_fit_chk.isChecked(),
            "sill":           self.sill_spin.value(),
            "range":          self.range_spin.value(),
            "nugget":         self.nugget_spin.value(),
            "anisotropy_angle":   self.aniso_angle_spin.value(),
            "anisotropy_scaling": self.aniso_scale_spin.value(),
            "nx":             self.nx_spin.value(),
            "ny":             self.ny_spin.value(),
            "custom_bounds":  self.custom_bounds_chk.isChecked(),
            "xmin":           self.xmin_spin.value(),
            "xmax":           self.xmax_spin.value(),
            "ymin":           self.ymin_spin.value(),
            "ymax":           self.ymax_spin.value(),
            "cross_validate": self.cv_chk.isChecked(),
        }

        self.log_edit.clear()
        self.metrics_label.setText("Running pipeline…")
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)

        self._thread = QThread()
        self._worker = KrigingWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log_edit.append)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(lambda: (
            self.run_btn.setEnabled(True), self.progress.setVisible(False)
        ))
        self._thread.start()

    # ── Callbacks ───────────────────────────────────────────────────────────
    def _on_finished(self, r):
        lines = [
            f"<b>Grid:</b> {r['grid_shape'][1]}×{r['grid_shape'][0]}  |  "
            f"<b>Samples:</b> {r['n_samples']}",
            f"<b>Predicted Z —</b>  mean={r['z_mean']:.4f}  "
            f"std={r['z_std']:.4f}  min={r['z_min']:.4f}  "
            f"max={r['z_max']:.4f}",
            f"<b>Mean variance:</b> {r['var_mean']:.4f}",
            f"<b>Variogram params:</b> {r['variogram_params']}",
        ]
        cv = r.get("cv")
        if cv:
            lines.append(
                f"<b>LOO-CV</b>  RMSE={cv['rmse']:.4f}  "
                f"MAE={cv['mae']:.4f}  ME={cv['me']:.4f}  "
                f"(n={cv['n_valid']})"
            )
        self.metrics_label.setText("<br>".join(lines))

        self._show_image(r["variogram_plot"], self.vario_img, 500, 220)
        self._show_image(r["interp_plot"],    self.interp_img, 600, 300)
        self._show_image(r["variance_plot"],  self.var_img,    600, 300)

    def _show_image(self, path, label_widget, w, h):
        if path and os.path.exists(path):
            pix = QPixmap(path).scaled(
                w, h, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            label_widget.setPixmap(pix)

    def _on_error(self, msg):
        self.log_edit.append(f"ERROR:\n{msg}")
        self.metrics_label.setText("Pipeline failed. See log.")
        QMessageBox.critical(self, "Error", msg[:600])


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT TAB
# ══════════════════════════════════════════════════════════════════════════════
class PredictTab(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = self._worker = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ── Left controls ───────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(320)
        lv = QVBoxLayout(left)
        lv.setSpacing(10)
        lv.setContentsMargins(0, 0, 0, 0)

        mb = section("1. Prior Run Output Folder")
        ml = QVBoxLayout(mb)
        ml.addWidget(QLabel("Select the output folder from a previous\n"
                            "kriging run to load the model."))
        mr = QHBoxLayout()
        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setPlaceholderText("Select folder…")
        self.model_dir_edit.setReadOnly(True)
        mbtn = QPushButton("Browse")
        mbtn.setFixedWidth(70)
        mbtn.clicked.connect(self._browse_model_dir)
        mr.addWidget(self.model_dir_edit)
        mr.addWidget(mbtn)
        ml.addLayout(mr)
        self.model_status = QLabel("")
        self.model_status.setWordWrap(True)
        self.model_status.setStyleSheet(f"color:{BORDER}; font-size:11px;")
        ml.addWidget(self.model_status)
        lv.addWidget(mb)

        db = section("2. New Data File")
        dl = QVBoxLayout(db)
        dr = QHBoxLayout()
        self.data_edit = QLineEdit()
        self.data_edit.setPlaceholderText("CSV or Excel with coordinates…")
        self.data_edit.setReadOnly(True)
        dbtn = QPushButton("Browse")
        dbtn.setFixedWidth(70)
        dbtn.clicked.connect(self._browse_data)
        dr.addWidget(self.data_edit)
        dr.addWidget(dbtn)
        dl.addLayout(dr)
        lv.addWidget(db)

        self.apply_btn = primary_btn("▶  Predict", color=GREEN,
                                     hover=GREEN_HOVER)
        self.apply_btn.clicked.connect(self._start)
        lv.addWidget(self.apply_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        lv.addWidget(self.progress)
        lv.addStretch()
        root.addWidget(left)

        # ── Right results ───────────────────────────────────────────────
        right_split = QSplitter(Qt.Orientation.Vertical)

        log_box = section("Log")
        log_layout = QVBoxLayout(log_box)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9))
        self.log_edit.setFixedHeight(110)
        log_layout.addWidget(self.log_edit)
        right_split.addWidget(log_box)

        res_scroll = QScrollArea()
        res_scroll.setWidgetResizable(True)
        res_widget = QWidget()
        rv = QVBoxLayout(res_widget)
        rv.setSpacing(10)

        self.apply_metrics = QLabel(
            "Predict on new coordinates to see results.")
        self.apply_metrics.setWordWrap(True)
        self.apply_metrics.setFont(QFont("Segoe UI", 10))
        rv.addWidget(self.apply_metrics)

        rv.addWidget(bold_label("Prediction Map"))
        self.pred_img = QLabel()
        self.pred_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pred_img.setStyleSheet(IMAGE_STYLE)
        self.pred_img.setMinimumHeight(260)
        rv.addWidget(self.pred_img)

        rv.addWidget(bold_label("Predictions Table"))
        self.pred_table = QTableWidget(0, 0)
        self.pred_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self.pred_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        rv.addWidget(self.pred_table, 1)

        rv.addStretch()
        res_scroll.setWidget(res_widget)
        right_split.addWidget(res_scroll)
        right_split.setSizes([150, 650])
        root.addWidget(right_split, 1)

    # ── Browse helpers ──────────────────────────────────────────────────────
    def _browse_model_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Prior Output Folder")
        if not path:
            return
        self.model_dir_edit.setText(path)
        p = Path(path)
        found, missing = [], []
        for fn in ["kriging_model.pkl", "kriging_config.json"]:
            (found if (p / fn).exists() else missing).append(fn)
        status = f"✔ Found: {', '.join(found)}" if found else ""
        if missing:
            status += f"\n✘ Missing: {', '.join(missing)}"
        self.model_status.setText(status)

    def _browse_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open New Data File", "",
            "Data Files (*.csv *.xlsx *.xls);;All Files (*)")
        if path:
            self.data_edit.setText(path)

    # ── Start ───────────────────────────────────────────────────────────────
    def _start(self):
        model_dir = self.model_dir_edit.text().strip()
        csv_path  = self.data_edit.text().strip()
        if not model_dir:
            QMessageBox.warning(self, "Missing",
                                "Select a prior output folder.")
            return
        if not csv_path:
            QMessageBox.warning(self, "Missing", "Select a data file.")
            return

        self.log_edit.clear()
        self.apply_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.apply_metrics.setText("Predicting…")

        self._thread = QThread()
        self._worker = PredictWorker(model_dir, csv_path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log_edit.append)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(lambda: (
            self.apply_btn.setEnabled(True),
            self.progress.setVisible(False)
        ))
        self._thread.start()

    # ── Callbacks ───────────────────────────────────────────────────────────
    def _on_finished(self, r):
        df = r["result_df"]
        z_col = r["z_col"]
        pred_col = f"{z_col}_predicted"
        self.apply_metrics.setText(
            f"<b>Samples:</b> {len(df)}  |  "
            f"<b>Predicted column:</b> {pred_col}"
        )

        # Plot
        pix = QPixmap()
        pix.loadFromData(r["plot_bytes"])
        pix = pix.scaled(600, 280, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
        self.pred_img.setPixmap(pix)

        # Table
        self.pred_table.setRowCount(len(df))
        self.pred_table.setColumnCount(len(df.columns))
        self.pred_table.setHorizontalHeaderLabels(list(df.columns))
        highlight = {pred_col, "Variance", "StdDev"}
        for ci, col in enumerate(df.columns):
            for ri in range(len(df)):
                val  = df.iloc[ri, ci]
                item = QTableWidgetItem(str(val))
                if col in highlight:
                    item.setBackground(QColor(30, 58, 138, 150))
                self.pred_table.setItem(ri, ci, item)

    def _on_error(self, msg):
        self.log_edit.append(f"ERROR:\n{msg}")
        self.apply_metrics.setText("Failed. See log.")
        QMessageBox.critical(self, "Error", msg[:600])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
class KrigingPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QWidget()
        header.setStyleSheet(f"QWidget {{ background:{HEADER_BG}; }}")
        header.setFixedHeight(56)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 0, 16, 0)
        title = QLabel("Kriging  –  Spatial Interpolation")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color:white;")
        hl.addWidget(title)
        layout.addWidget(header)

        tabs = QTabWidget()
        tabs.setFont(QFont("Segoe UI", 10))
        tabs.addTab(InterpolationTab(), "  Interpolation  ")
        tabs.addTab(PredictTab(), "  Predict New Points  ")
        layout.addWidget(tabs)

