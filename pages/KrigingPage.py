import os
import io
import numpy as np
import pandas as pd
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QScrollArea, QProgressBar, QMessageBox, QHeaderView,
    QTextEdit, QSplitter, QFrame, QCheckBox, QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QPixmap

from styles import (
    _section, _bold_label, _primary_btn,
    PAGE_HEADER_BG, PAGE_HEADER_FG, IMG_BORDER_STYLE,
    ACCENT_ALT, ACCENT_ALT_H,
)


VARIOGRAM_MODELS = ["linear", "power", "gaussian", "spherical", "exponential", "hole-effect"]

KRIGING_TYPES = ["Ordinary", "Universal"]


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
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from pykrige.ok import OrdinaryKriging
            from pykrige.uk import UniversalKriging

            p = self.p

            # ── Step 1: Load data ─────────────────────────────────────────
            self.log.emit("Step 1: Loading data…")
            ext = os.path.splitext(p['csv_path'])[1].lower()
            try:
                df = pd.read_excel(p['csv_path']) if ext in ('.xlsx', '.xls') \
                     else pd.read_csv(p['csv_path'])
            except Exception:
                df = pd.read_csv(p['csv_path'], on_bad_lines='skip', low_memory=False)

            x_col = p['x_col']
            y_col = p['y_col']
            z_col = p['z_col']

            # Drop rows with NaN in any of the three columns
            df_clean = df[[x_col, y_col, z_col]].dropna()
            df_clean = df_clean.apply(pd.to_numeric, errors='coerce').dropna()

            x = df_clean[x_col].values.astype(float)
            y = df_clean[y_col].values.astype(float)
            z = df_clean[z_col].values.astype(float)
            self.log.emit(f"  {len(x)} valid data points loaded.")

            if len(x) < 3:
                raise ValueError("Need at least 3 valid data points for kriging.")

            # Optional log-transform for skewed (e.g. concentration) data.
            # All kriging math runs on z_work; only the final interpolation
            # surface is back-transformed to original units for display.
            log_transform = bool(p.get('log_transform'))
            if log_transform:
                if np.nanmin(z) < 0:
                    raise ValueError(
                        "Log-transform requires non-negative Z values "
                        f"(min is {np.nanmin(z):.4g})."
                    )
                z_work = np.log1p(z)   # ln(1 + z): handles zeros, monotonic
                self.log.emit(
                    "  Log-transform ON: kriging on ln(1+Z). Interpolation map "
                    "is back-transformed to original units; variance and "
                    "variogram are in log space."
                )
            else:
                z_work = z

            # ── Step 2: Build interpolation grid ──────────────────────────
            self.log.emit("Step 2: Building interpolation grid…")
            grid_nx = p['grid_nx']
            grid_ny = p['grid_ny']
            x_margin = (x.max() - x.min()) * 0.05
            y_margin = (y.max() - y.min()) * 0.05
            grid_x = np.linspace(x.min() - x_margin, x.max() + x_margin, grid_nx)
            grid_y = np.linspace(y.min() - y_margin, y.max() + y_margin, grid_ny)
            self.log.emit(f"  Grid: {grid_nx} x {grid_ny} = {grid_nx * grid_ny} cells")

            # ── Step 3: Configure variogram / kriging ─────────────────────
            variogram_model = p['variogram_model']
            nlags = p['nlags']
            exact_values = p['exact_values']
            kriging_type = p['kriging_type']

            # Manual variogram parameters (keys already match the selected
            # model: linear->slope/nugget, power->scale/exponent/nugget,
            # bounded->sill/range/nugget). None means auto-fit.
            variogram_parameters = p.get('variogram_parameters')
            if variogram_parameters:
                pretty = ", ".join(f"{k}={v}" for k, v in variogram_parameters.items())
                self.log.emit(f"  Manual variogram: {pretty}")
            else:
                self.log.emit("  Variogram: auto-fit")

            self.log.emit(f"  Kriging type: {kriging_type}")
            self.log.emit(f"  Variogram model: {variogram_model}")
            self.log.emit(f"  NLags: {nlags}, Exact values: {exact_values}")

            # ── Step 4: Run kriging ───────────────────────────────────────
            self.log.emit("Step 3: Running kriging interpolation…")

            common_kwargs = dict(
                variogram_model=variogram_model,
                nlags=nlags,
                exact_values=exact_values,
                verbose=False,
                enable_plotting=False,
            )
            if variogram_parameters is not None:
                common_kwargs['variogram_parameters'] = variogram_parameters

            drift_terms = []
            if kriging_type == "Ordinary":
                krige = OrdinaryKriging(x, y, z_work, **common_kwargs)
            else:
                if p.get('drift_regional_linear'):
                    drift_terms.append('regional_linear')
                uk_kwargs = dict(common_kwargs)
                if drift_terms:
                    uk_kwargs['drift_terms'] = drift_terms
                krige = UniversalKriging(x, y, z_work, **uk_kwargs)

            z_grid_work, ss_grid = krige.execute('grid', grid_x, grid_y)
            # Back-transform the surface for display/output (conditional median).
            z_grid = np.expm1(z_grid_work) if log_transform else z_grid_work
            self.log.emit(f"  Interpolation complete. Z range: [{z_grid.min():.4f}, {z_grid.max():.4f}]")
            self.log.emit(f"  Variance range: [{ss_grid.min():.4f}, {ss_grid.max():.4f}]")

            # ── Step 5: Extract fitted variogram parameters ───────────────
            self.log.emit("Step 4: Extracting variogram information…")
            vp = krige.variogram_model_parameters
            fitted_info = {
                'model': variogram_model,
            }
            if variogram_model == 'linear':
                fitted_info['slope'] = float(vp[0])
                fitted_info['nugget'] = float(vp[1])
            elif variogram_model == 'power':
                fitted_info['scale'] = float(vp[0])
                fitted_info['exponent'] = float(vp[1])
                fitted_info['nugget'] = float(vp[2])
            else:
                # gaussian, spherical, exponential, hole-effect
                fitted_info['sill'] = float(vp[0])
                fitted_info['range'] = float(vp[1])
                fitted_info['nugget'] = float(vp[2])

            for k, v in fitted_info.items():
                if k != 'model':
                    self.log.emit(f"    {k}: {v:.6f}")

            # ── Step 4b: Leave-one-out cross-validation ───────────────────
            # Refits the predictor on n-1 points and predicts the held-out
            # point, reusing the full-data variogram fit (vp) so the variogram
            # is not re-estimated each fold (standard, much faster). Reports
            # RMSE, mean error (bias), MAE, observed/predicted correlation, and
            # the mean squared deviation ratio (MSDR ~ 1 if variance is well
            # calibrated).
            cv = None
            cv_png = None
            if p.get('cross_validate'):
                n = len(x)
                cv_cap = p.get('cv_max_points', 1000)
                if n > cv_cap:
                    self.log.emit(
                        f"  Cross-validation skipped (n={n} > {cv_cap} cap)."
                    )
                else:
                    self.log.emit(
                        f"Step 4b: Leave-one-out cross-validation ({n} folds)…"
                    )
                    # Rebuild the fitted parameters as a model-appropriate dict.
                    # vp is pykrige's internal order; bounded models store the
                    # partial sill in vp[0], so feed it back as 'psill'.
                    if variogram_model == 'linear':
                        refit = {'slope': float(vp[0]), 'nugget': float(vp[1])}
                    elif variogram_model == 'power':
                        refit = {'scale': float(vp[0]), 'exponent': float(vp[1]),
                                 'nugget': float(vp[2])}
                    else:
                        refit = {'psill': float(vp[0]), 'range': float(vp[1]),
                                 'nugget': float(vp[2])}

                    cv_kwargs = dict(
                        variogram_model=variogram_model,
                        variogram_parameters=refit,
                        nlags=nlags,
                        exact_values=exact_values,
                        verbose=False,
                        enable_plotting=False,
                        pseudo_inv=True,
                    )
                    if kriging_type == "Universal" and drift_terms:
                        cv_kwargs['drift_terms'] = drift_terms

                    pred = np.full(n, np.nan)
                    pvar = np.full(n, np.nan)
                    idx_all = np.arange(n)
                    n_fail = 0
                    for i in range(n):
                        sel = idx_all != i
                        try:
                            if kriging_type == "Ordinary":
                                kcv = OrdinaryKriging(x[sel], y[sel], z_work[sel], **cv_kwargs)
                            else:
                                kcv = UniversalKriging(x[sel], y[sel], z_work[sel], **cv_kwargs)
                            zp, sp = kcv.execute('points',
                                                 np.array([x[i]]), np.array([y[i]]))
                            pred[i] = float(zp[0])
                            pvar[i] = float(sp[0])
                        except Exception:
                            n_fail += 1

                    mask = ~np.isnan(pred)
                    m = int(mask.sum())
                    if m >= 2:
                        # All quantities are in the kriging space (log space when
                        # the log-transform is enabled), so MSDR stays valid.
                        zt = z_work[mask]
                        err = pred[mask] - zt
                        rmse = float(np.sqrt(np.mean(err ** 2)))
                        me   = float(np.mean(err))
                        mae  = float(np.mean(np.abs(err)))
                        with np.errstate(divide='ignore', invalid='ignore'):
                            std_err = err / np.sqrt(pvar[mask])
                        msdr = float(np.nanmean(std_err ** 2))
                        if np.std(pred[mask]) > 0 and np.std(zt) > 0:
                            r = float(np.corrcoef(pred[mask], zt)[0, 1])
                        else:
                            r = float('nan')
                        cv = {
                            'n': m, 'n_fail': n_fail,
                            'rmse': rmse, 'me': me, 'mae': mae,
                            'msdr': msdr, 'r': r,
                        }
                        self.log.emit(
                            f"  CV: RMSE={rmse:.4f}  ME={me:.4f}  MAE={mae:.4f}  "
                            f"r={r:.4f}  MSDR={msdr:.4f}"
                        )
                        if n_fail:
                            self.log.emit(f"  {n_fail} fold(s) failed and were skipped.")

                        # Observed vs predicted scatter
                        obs = zt
                        prd = pred[mask]
                        sp_sfx = ' (log)' if log_transform else ''
                        fig_cv, ax_cv = plt.subplots(figsize=(6, 6))
                        ax_cv.scatter(obs, prd, c='#2563eb', s=25,
                                      edgecolors='black', linewidths=0.3, alpha=0.7)
                        lo = float(min(obs.min(), prd.min()))
                        hi = float(max(obs.max(), prd.max()))
                        ax_cv.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='1:1')
                        ax_cv.set_xlabel(f'Observed{sp_sfx}', fontsize=11)
                        ax_cv.set_ylabel(f'Predicted (LOOCV){sp_sfx}', fontsize=11)
                        ax_cv.set_title(f'Cross-Validation (RMSE={rmse:.3f}, r={r:.3f})',
                                        fontsize=12, fontweight='bold')
                        ax_cv.legend(fontsize=9)
                        ax_cv.grid(True, alpha=0.3)
                        ax_cv.set_aspect('equal', adjustable='datalim')
                        fig_cv.tight_layout()
                        buf_cv = io.BytesIO()
                        fig_cv.savefig(buf_cv, format='png', dpi=150)
                        plt.close(fig_cv)
                        buf_cv.seek(0)
                        cv_png = buf_cv.read()
                    else:
                        self.log.emit("  Cross-validation produced too few valid folds.")

            # ── Step 6: Generate variogram plot ───────────────────────────
            self.log.emit("Step 5: Generating variogram plot…")
            lags = krige.lags
            semivariance = krige.semivariance

            lag_fine = np.linspace(0, lags.max() * 1.1, 200)
            fitted_sv = krige.variogram_function(vp, lag_fine)

            fig_var, ax_var = plt.subplots(figsize=(8, 5))
            ax_var.scatter(lags, semivariance, c='#2563eb', s=50, zorder=5,
                           edgecolors='black', linewidths=0.5, label='Experimental')
            ax_var.plot(lag_fine, fitted_sv, 'r-', linewidth=2,
                        label=f'Fitted ({variogram_model})')
            ax_var.set_xlabel('Lag Distance', fontsize=11)
            ax_var.set_ylabel('Semivariance' + (' (log Z)' if log_transform else ''),
                              fontsize=11)
            ax_var.set_title('Experimental Variogram & Fitted Model',
                             fontsize=13, fontweight='bold')
            ax_var.legend(fontsize=10)
            ax_var.grid(True, alpha=0.3)
            fig_var.tight_layout()

            buf_var = io.BytesIO()
            fig_var.savefig(buf_var, format='png', dpi=150)
            plt.close(fig_var)
            buf_var.seek(0)
            variogram_png = buf_var.read()

            # ── Step 7: Generate interpolation heatmap ────────────────────
            self.log.emit("Step 6: Generating interpolation heatmap…")
            fig_int, ax_int = plt.subplots(figsize=(9, 7))
            im = ax_int.pcolormesh(grid_x, grid_y, z_grid, shading='auto', cmap='viridis')
            ax_int.scatter(x, y, c='red', s=12, edgecolors='black', linewidths=0.3,
                           label='Sample points', zorder=5)
            cbar = fig_int.colorbar(im, ax=ax_int, shrink=0.85)
            cbar.set_label(z_col, fontsize=11)
            ax_int.set_xlabel(x_col, fontsize=11)
            ax_int.set_ylabel(y_col, fontsize=11)
            ax_int.set_title(
                f'Kriging Interpolation — {z_col}'
                + (' (log-kriged, back-transformed)' if log_transform else ''),
                fontsize=13, fontweight='bold')
            ax_int.legend(loc='upper right', fontsize=9)
            ax_int.set_aspect('equal')
            fig_int.tight_layout()

            buf_int = io.BytesIO()
            fig_int.savefig(buf_int, format='png', dpi=150)
            plt.close(fig_int)
            buf_int.seek(0)
            interp_png = buf_int.read()

            # ── Step 8: Generate variance / uncertainty heatmap ───────────
            self.log.emit("Step 7: Generating variance (uncertainty) heatmap…")
            fig_unc, ax_unc = plt.subplots(figsize=(9, 7))
            im2 = ax_unc.pcolormesh(grid_x, grid_y, ss_grid, shading='auto', cmap='magma')
            ax_unc.scatter(x, y, c='cyan', s=12, edgecolors='black', linewidths=0.3,
                           label='Sample points', zorder=5)
            cbar2 = fig_unc.colorbar(im2, ax=ax_unc, shrink=0.85)
            cbar2.set_label('Kriging Variance' + (' (log space)' if log_transform else ''),
                            fontsize=11)
            ax_unc.set_xlabel(x_col, fontsize=11)
            ax_unc.set_ylabel(y_col, fontsize=11)
            ax_unc.set_title(f'Kriging Variance (Uncertainty) — {z_col}',
                             fontsize=13, fontweight='bold')
            ax_unc.legend(loc='upper right', fontsize=9)
            ax_unc.set_aspect('equal')
            fig_unc.tight_layout()

            buf_unc = io.BytesIO()
            fig_unc.savefig(buf_unc, format='png', dpi=150)
            plt.close(fig_unc)
            buf_unc.seek(0)
            variance_png = buf_unc.read()

            # ── Step 9: Build gridded CSV data ────────────────────────────
            self.log.emit("Step 8: Building gridded result data…")
            gx, gy = np.meshgrid(grid_x, grid_y)
            var_suffix = '_variance_log' if log_transform else '_variance'
            grid_df = pd.DataFrame({
                x_col: gx.ravel(),
                y_col: gy.ravel(),
                f'{z_col}_interpolated': z_grid.ravel(),
                f'{z_col}{var_suffix}': ss_grid.ravel(),
            })

            # ── Save outputs if out_dir provided ─────────────────────────
            out_dir = p.get('out_dir')
            saved_paths = {}
            if out_dir:
                out_path = Path(out_dir)
                out_path.mkdir(parents=True, exist_ok=True)

                grid_csv = str(out_path / f'kriging_{z_col}.csv')
                grid_df.to_csv(grid_csv, index=False)
                saved_paths['grid_csv'] = grid_csv
                self.log.emit(f"  Saved grid CSV: {grid_csv}")

                var_plot_path = str(out_path / f'variogram_{z_col}.png')
                with open(var_plot_path, 'wb') as f:
                    f.write(variogram_png)
                saved_paths['variogram_plot'] = var_plot_path

                interp_path = str(out_path / f'interpolation_{z_col}.png')
                with open(interp_path, 'wb') as f:
                    f.write(interp_png)
                saved_paths['interp_plot'] = interp_path

                unc_path = str(out_path / f'variance_{z_col}.png')
                with open(unc_path, 'wb') as f:
                    f.write(variance_png)
                saved_paths['variance_plot'] = unc_path

                if cv_png is not None:
                    cv_path = str(out_path / f'crossval_{z_col}.png')
                    with open(cv_path, 'wb') as f:
                        f.write(cv_png)
                    saved_paths['crossval_plot'] = cv_path

                self.log.emit(f"  All outputs saved to: {out_dir}")

            self.log.emit("Kriging pipeline complete!")
            self.finished.emit({
                'variogram_png':  variogram_png,
                'interp_png':     interp_png,
                'variance_png':   variance_png,
                'grid_df':        grid_df,
                'fitted_info':    fitted_info,
                'n_points':       len(x),
                'grid_shape':     (grid_ny, grid_nx),
                'z_range':        (float(z_grid.min()), float(z_grid.max())),
                'var_range':      (float(ss_grid.min()), float(ss_grid.max())),
                'saved_paths':    saved_paths,
                'z_col':          z_col,
                'cv':             cv,
                'cv_png':         cv_png,
                'log_transform':  log_transform,
            })

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# KRIGING PAGE
# ══════════════════════════════════════════════════════════════════════════════
class KrigingPage(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = self._worker = None
        self._last_grid_df = None
        self._build_ui()

    def _build_ui(self):
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)

        # ── Header ────────────────────────────────────────────────────────
        header = QWidget()
        header.setStyleSheet(f"QWidget {{ background:{PAGE_HEADER_BG}; }}")
        header.setFixedHeight(56)
        hl = QHBoxLayout(header); hl.setContentsMargins(16, 0, 16, 0)
        title = QLabel("Kriging  –  Spatial Interpolation (Gaussian Process Regression)")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{PAGE_HEADER_FG};")
        hl.addWidget(title)
        page_layout.addWidget(header)

        # ── Body (left controls / right results) ──────────────────────────
        body = QHBoxLayout()
        body.setContentsMargins(12, 12, 12, 12)
        body.setSpacing(12)

        # ── Left controls (scrollable) ────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(320)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_widget = QWidget()
        lv = QVBoxLayout(left_widget)
        lv.setSpacing(10)
        lv.setContentsMargins(0, 0, 4, 0)

        # 1. Data File
        fb = _section("1. Data File")
        fl = QVBoxLayout(fb)
        fr = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("CSV or Excel…")
        self.file_edit.setReadOnly(True)
        bb = QPushButton("Browse"); bb.setFixedWidth(70)
        bb.clicked.connect(self._browse_file)
        fr.addWidget(self.file_edit); fr.addWidget(bb)
        fl.addLayout(fr)

        fl.addWidget(QLabel("X Coordinate column:"))
        self.x_combo = QComboBox()
        self.x_combo.setPlaceholderText("Select X column…")
        self.x_combo.setEnabled(False)
        fl.addWidget(self.x_combo)

        fl.addWidget(QLabel("Y Coordinate column:"))
        self.y_combo = QComboBox()
        self.y_combo.setPlaceholderText("Select Y column…")
        self.y_combo.setEnabled(False)
        fl.addWidget(self.y_combo)

        fl.addWidget(QLabel("Target value column (Z):"))
        self.z_combo = QComboBox()
        self.z_combo.setPlaceholderText("Select value to interpolate…")
        self.z_combo.setEnabled(False)
        fl.addWidget(self.z_combo)
        lv.addWidget(fb)

        # 2. Kriging Configuration
        kb = _section("2. Kriging Configuration")
        kf = QFormLayout(kb); kf.setSpacing(6)

        self.type_combo = QComboBox()
        self.type_combo.addItems(KRIGING_TYPES)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        kf.addRow("Kriging type:", self.type_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems(VARIOGRAM_MODELS)
        self.model_combo.setCurrentText("spherical")
        kf.addRow("Variogram model:", self.model_combo)

        self.nlags_spin = QSpinBox()
        self.nlags_spin.setRange(4, 100)
        self.nlags_spin.setValue(6)
        self.nlags_spin.setToolTip("Number of lag bins for the experimental variogram.")
        kf.addRow("NLags:", self.nlags_spin)

        self.exact_check = QCheckBox("Exact values")
        self.exact_check.setChecked(True)
        self.exact_check.setToolTip(
            "If checked, the interpolated value at each data point\n"
            "will match the measured value exactly (no smoothing)."
        )
        kf.addRow(self.exact_check)

        self.log_check = QCheckBox("Log-transform Z (skewed data)")
        self.log_check.setChecked(False)
        self.log_check.setToolTip(
            "Krige on ln(1+Z) instead of raw Z. Recommended for\n"
            "strongly right-skewed values (e.g. element concentrations).\n"
            "The interpolation map is back-transformed to original units;\n"
            "variance and variogram remain in log space.\n"
            "Requires non-negative Z."
        )
        kf.addRow(self.log_check)

        self.cv_check = QCheckBox("Cross-validate (LOOCV)")
        self.cv_check.setChecked(True)
        self.cv_check.setToolTip(
            "Leave-one-out cross-validation to assess accuracy\n"
            "(RMSE, bias, correlation). Reuses the fitted variogram;\n"
            "skipped automatically for very large datasets."
        )
        kf.addRow(self.cv_check)
        lv.addWidget(kb)

        # 2b. Universal Kriging drift terms
        self.drift_box = _section("2b. Drift Terms (Universal)")
        drift_layout = QVBoxLayout(self.drift_box)
        self.drift_regional = QCheckBox("Regional Linear Drift")
        self.drift_regional.setChecked(True)
        self.drift_regional.setToolTip(
            "Assumes a linear spatial trend in the data.\n"
            "Suitable when there is a large-scale gradient."
        )
        drift_layout.addWidget(self.drift_regional)
        self.drift_box.setVisible(False)
        lv.addWidget(self.drift_box)

        # 3. Variogram Parameters
        vb = _section("3. Variogram Parameters")
        vl = QVBoxLayout(vb)
        self.auto_fit_check = QCheckBox("Auto-fit (recommended)")
        self.auto_fit_check.setChecked(True)
        self.auto_fit_check.setToolTip(
            "Automatically estimate sill, range, and nugget\n"
            "from the data using least-squares fitting."
        )
        self.auto_fit_check.toggled.connect(self._on_autofit_toggled)
        vl.addWidget(self.auto_fit_check)

        # Manual fields are rebuilt to match the selected model:
        # linear -> slope/nugget, power -> scale/exponent/nugget,
        # bounded models -> sill/range/nugget.
        self.manual_form = QWidget()
        self.manual_form_layout = QFormLayout(self.manual_form)
        self.manual_form_layout.setSpacing(6)
        self.manual_form.setVisible(False)
        vl.addWidget(self.manual_form)
        self._manual_spins = {}
        self._rebuild_manual_form()
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        lv.addWidget(vb)

        # 4. Grid Resolution
        gb = _section("4. Grid Resolution")
        gf = QFormLayout(gb); gf.setSpacing(6)

        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(10, 2000)
        self.nx_spin.setValue(100)
        self.nx_spin.setToolTip("Number of grid points in the X direction.")
        gf.addRow("Grid NX:", self.nx_spin)

        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(10, 2000)
        self.ny_spin.setValue(100)
        self.ny_spin.setToolTip("Number of grid points in the Y direction.")
        gf.addRow("Grid NY:", self.ny_spin)
        lv.addWidget(gb)

        # 5. Output folder (optional)
        ob = _section("5. Output Folder (optional)")
        ol = QVBoxLayout(ob)
        or_ = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Select to save outputs…")
        self.out_edit.setReadOnly(True)
        ob2 = QPushButton("Browse"); ob2.setFixedWidth(70)
        ob2.clicked.connect(self._browse_out)
        or_.addWidget(self.out_edit); or_.addWidget(ob2)
        ol.addLayout(or_)
        lv.addWidget(ob)

        # Run button
        self.run_btn = _primary_btn("▶  Run Kriging")
        self.run_btn.clicked.connect(self._start)
        lv.addWidget(self.run_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        lv.addWidget(self.progress)

        # Export CSV button
        self.export_btn = _primary_btn("💾  Export Grid CSV", color=ACCENT_ALT, hover=ACCENT_ALT_H)
        self.export_btn.setEnabled(False)
        self.export_btn.setToolTip("Save the interpolated grid as a CSV file.")
        self.export_btn.clicked.connect(self._export_csv)
        lv.addWidget(self.export_btn)

        lv.addStretch()
        left_scroll.setWidget(left_widget)
        body.addWidget(left_scroll)

        # ── Right results ─────────────────────────────────────────────────
        right_split = QSplitter(Qt.Orientation.Vertical)

        # Log
        log_box = _section("Kriging Log")
        log_layout = QVBoxLayout(log_box)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9))
        self.log_edit.setFixedHeight(130)
        log_layout.addWidget(self.log_edit)
        right_split.addWidget(log_box)

        # Results area (scrollable)
        res_scroll = QScrollArea()
        res_scroll.setWidgetResizable(True)
        res_widget = QWidget()
        self.res_layout = QVBoxLayout(res_widget)
        self.res_layout.setSpacing(10)

        self.metrics_label = QLabel("Run kriging to see results.")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setFont(QFont("Segoe UI", 10))
        self.res_layout.addWidget(self.metrics_label)

        # Variogram fit summary table
        self.res_layout.addWidget(_bold_label("Fitted Variogram Parameters"))
        self.vario_table = QTableWidget(0, 2)
        self.vario_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.vario_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.vario_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.vario_table.setFixedHeight(140)
        self.res_layout.addWidget(self.vario_table)

        # Variogram plot
        self.res_layout.addWidget(_bold_label("Variogram Plot"))
        self.vario_img = QLabel()
        self.vario_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vario_img.setStyleSheet(IMG_BORDER_STYLE)
        self.vario_img.setMinimumHeight(240)
        self.res_layout.addWidget(self.vario_img)

        # Interpolation heatmap
        self.res_layout.addWidget(_bold_label("Interpolation Map"))
        self.interp_img = QLabel()
        self.interp_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.interp_img.setStyleSheet(IMG_BORDER_STYLE)
        self.interp_img.setMinimumHeight(300)
        self.res_layout.addWidget(self.interp_img)

        # Variance / uncertainty heatmap
        self.res_layout.addWidget(_bold_label("Variance (Uncertainty) Map"))
        self.variance_img = QLabel()
        self.variance_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.variance_img.setStyleSheet(IMG_BORDER_STYLE)
        self.variance_img.setMinimumHeight(300)
        self.res_layout.addWidget(self.variance_img)

        # Cross-validation scatter (hidden until LOOCV runs)
        self.cv_label = _bold_label("Cross-Validation (Observed vs Predicted)")
        self.cv_label.setVisible(False)
        self.res_layout.addWidget(self.cv_label)
        self.cv_img = QLabel()
        self.cv_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cv_img.setStyleSheet(IMG_BORDER_STYLE)
        self.cv_img.setMinimumHeight(300)
        self.cv_img.setVisible(False)
        self.res_layout.addWidget(self.cv_img)

        self.res_layout.addStretch()
        res_scroll.setWidget(res_widget)
        right_split.addWidget(res_scroll)
        right_split.setSizes([180, 620])
        body.addWidget(right_split, 1)

        page_layout.addLayout(body)

    # ── Toggle handlers ────────────────────────────────────────────────────
    def _on_type_changed(self, text):
        self.drift_box.setVisible(text == "Universal")

    def _on_autofit_toggled(self, checked):
        self.manual_form.setVisible(not checked)

    def _on_model_changed(self, _text):
        self._rebuild_manual_form()

    def _rebuild_manual_form(self):
        while self.manual_form_layout.rowCount():
            self.manual_form_layout.removeRow(0)
        self._manual_spins = {}

        model = self.model_combo.currentText()
        if model == 'linear':
            specs = [
                ('slope',  "Slope:",  1.0, 0.0001, 1e9, "Slope of the linear variogram."),
                ('nugget', "Nugget:", 0.0, 0.0,    1e9, "Micro-scale variability at lag 0."),
            ]
        elif model == 'power':
            specs = [
                ('scale',    "Scale:",    1.0, 0.0001, 1e9,    "Scaling factor."),
                ('exponent', "Exponent:", 1.0, 0.0001, 1.9999, "Exponent (0 < exponent < 2)."),
                ('nugget',   "Nugget:",   0.0, 0.0,    1e9,    "Micro-scale variability at lag 0."),
            ]
        else:
            specs = [
                ('sill',   "Sill:",   1.0, 0.0001, 1e9, "Total variance (partial sill + nugget)."),
                ('range',  "Range:",  1.0, 0.0001, 1e9, "Distance at which the semivariance plateaus."),
                ('nugget', "Nugget:", 0.0, 0.0,    1e9, "Micro-scale variability at lag 0."),
            ]

        for key, label, default, lo, hi, tip in specs:
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(4)
            spin.setValue(default)
            spin.setToolTip(tip)
            self.manual_form_layout.addRow(label, spin)
            self._manual_spins[key] = spin

    # ── File / dir browsing ────────────────────────────────────────────────
    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", "Data Files (*.csv *.xlsx *.xls);;All Files (*)"
        )
        if not path:
            return
        self.file_edit.setText(path)
        try:
            ext = os.path.splitext(path)[1].lower()
            df = pd.read_excel(path, nrows=5) if ext in ('.xlsx', '.xls') \
                 else pd.read_csv(path, nrows=5)
            numeric_cols = [c for c in df.columns
                           if pd.to_numeric(df[c], errors='coerce').notna().any()]

            self.x_combo.clear()
            self.x_combo.addItems(numeric_cols)
            self.x_combo.setEnabled(True)
            for hint in ('X_Coord', 'x_coord', 'X', 'Easting', 'Longitude', 'lon', 'lng'):
                idx = self.x_combo.findText(hint)
                if idx >= 0:
                    self.x_combo.setCurrentIndex(idx)
                    break

            self.y_combo.clear()
            self.y_combo.addItems(numeric_cols)
            self.y_combo.setEnabled(True)
            for hint in ('Y_Coord', 'y_coord', 'Y', 'Northing', 'Latitude', 'lat'):
                idx = self.y_combo.findText(hint)
                if idx >= 0:
                    self.y_combo.setCurrentIndex(idx)
                    break

            self.z_combo.clear()
            self.z_combo.addItems(numeric_cols)
            self.z_combo.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.out_edit.setText(path)

    # ── Start kriging ──────────────────────────────────────────────────────
    def _start(self):
        csv_path = self.file_edit.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "Missing", "Please select a data file."); return

        x_col = self.x_combo.currentText()
        y_col = self.y_combo.currentText()
        z_col = self.z_combo.currentText()
        if not x_col or not y_col or not z_col:
            QMessageBox.warning(self, "Missing", "Please select X, Y, and Z columns."); return
        if len({x_col, y_col, z_col}) < 3:
            QMessageBox.warning(self, "Invalid", "X, Y, and Z columns must be different."); return

        manual = not self.auto_fit_check.isChecked()

        params = {
            'csv_path':           csv_path,
            'x_col':              x_col,
            'y_col':              y_col,
            'z_col':              z_col,
            'kriging_type':       self.type_combo.currentText(),
            'variogram_model':    self.model_combo.currentText(),
            'nlags':              self.nlags_spin.value(),
            'exact_values':       self.exact_check.isChecked(),
            'grid_nx':            self.nx_spin.value(),
            'grid_ny':            self.ny_spin.value(),
            'out_dir':            self.out_edit.text().strip() or None,
            'log_transform':      self.log_check.isChecked(),
            'cross_validate':     self.cv_check.isChecked(),
            'variogram_parameters': (
                {k: s.value() for k, s in self._manual_spins.items()}
                if manual else None
            ),
        }

        if self.type_combo.currentText() == "Universal":
            params['drift_regional_linear'] = self.drift_regional.isChecked()

        self.log_edit.clear()
        self.metrics_label.setText("Running kriging…")
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.export_btn.setEnabled(False)

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

    # ── Results display ────────────────────────────────────────────────────
    def _on_finished(self, r):
        self._last_grid_df = r['grid_df']
        self.export_btn.setEnabled(True)

        z_lo, z_hi = r['z_range']
        v_lo, v_hi = r['var_range']
        ny, nx = r['grid_shape']
        lines = [
            f"<b>Data points:</b> {r['n_points']}  |  "
            f"<b>Grid:</b> {nx} x {ny} = {nx * ny} cells",
            f"<b>Interpolated Z range:</b> [{z_lo:.4f}, {z_hi:.4f}]",
            f"<b>Variance range:</b> [{v_lo:.4f}, {v_hi:.4f}]",
        ]
        logt = r.get('log_transform')
        if logt:
            lines.append(
                "<b>Log-transform:</b> ON (ln(1+Z)); map in original units, "
                "variance/variogram in log space"
            )
        cv = r.get('cv')
        if cv:
            cv_units = " (log space)" if logt else ""
            lines.append(
                f"<b>LOOCV (n={cv['n']}){cv_units}:</b> RMSE={cv['rmse']:.4f}, "
                f"ME={cv['me']:.4f}, MAE={cv['mae']:.4f}, "
                f"r={cv['r']:.4f}, MSDR={cv['msdr']:.4f}"
            )
        self.metrics_label.setText("<br>".join(lines))

        # Variogram parameters table
        info = r['fitted_info']
        params_items = [(k, v) for k, v in info.items() if k != 'model']
        self.vario_table.setRowCount(len(params_items) + 1)
        self.vario_table.setItem(0, 0, QTableWidgetItem("model"))
        self.vario_table.setItem(0, 1, QTableWidgetItem(info['model']))
        for i, (k, v) in enumerate(params_items):
            self.vario_table.setItem(i + 1, 0, QTableWidgetItem(k))
            item = QTableWidgetItem(f"{v:.6f}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.vario_table.setItem(i + 1, 1, item)

        # Images
        self._show_png(r['variogram_png'], self.vario_img,  560, 280)
        self._show_png(r['interp_png'],    self.interp_img, 620, 360)
        self._show_png(r['variance_png'],  self.variance_img, 620, 360)

        cv_png = r.get('cv_png')
        if cv_png:
            self._show_png(cv_png, self.cv_img, 480, 480)
            self.cv_img.setVisible(True)
            self.cv_label.setVisible(True)
        else:
            self.cv_img.clear()
            self.cv_img.setVisible(False)
            self.cv_label.setVisible(False)

    def _show_png(self, data: bytes, label_widget, w, h):
        pix = QPixmap()
        pix.loadFromData(data)
        pix = pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
        label_widget.setPixmap(pix)

    def _on_error(self, msg):
        self.log_edit.append(f"ERROR:\n{msg}")
        self.metrics_label.setText("Kriging failed. See log.")
        QMessageBox.critical(self, "Error", msg[:600])

    # ── Export grid CSV ────────────────────────────────────────────────────
    def _export_csv(self):
        if self._last_grid_df is None:
            QMessageBox.warning(self, "No Data", "Run kriging first."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Grid CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self._last_grid_df.to_csv(path, index=False)
            QMessageBox.information(self, "Saved", f"Grid exported to:\n{path}")

