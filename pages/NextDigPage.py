import os
import io
import math
import numpy as np
import pandas as pd
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QScrollArea, QProgressBar, QMessageBox, QHeaderView,
    QTextEdit, QSplitter, QCheckBox, QSlider,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QPixmap, QImage

from styles import (
    _section, _bold_label, _primary_btn,
    PAGE_HEADER_BG, PAGE_HEADER_FG, IMG_BORDER_STYLE,
    ACCENT_ALT, ACCENT_ALT_H, TEXT_SECONDARY,
)


VARIOGRAM_MODELS = ["spherical", "exponential", "gaussian", "linear", "power", "hole-effect"]


# ══════════════════════════════════════════════════════════════════════════════
# NEXT-DIG WORKER
# ══════════════════════════════════════════════════════════════════════════════
class NextDigWorker(QObject):
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
            import matplotlib.image as mpimg
            from pykrige.ok import OrdinaryKriging
            from scipy.spatial import cKDTree, Delaunay

            p = self.p

            # ── Step 1: Load data ─────────────────────────────────────────
            if p.get('df') is not None:
                df = p['df']
                self.log.emit("Step 1: Using in-memory results (no file load)…")
            else:
                self.log.emit("Step 1: Loading data…")
                ext = os.path.splitext(p['csv_path'])[1].lower()
                try:
                    df = pd.read_excel(p['csv_path']) if ext in ('.xlsx', '.xls') \
                         else pd.read_csv(p['csv_path'])
                except Exception:
                    df = pd.read_csv(p['csv_path'], on_bad_lines='skip', low_memory=False)

            x_col, y_col, z_col = p['x_col'], p['y_col'], p['z_col']
            d = df[[x_col, y_col, z_col]].apply(pd.to_numeric, errors='coerce').dropna()
            x = d[x_col].values.astype(float)
            y = d[y_col].values.astype(float)
            z = d[z_col].values.astype(float)
            self.log.emit(f"  {len(x)} valid samples loaded.")
            if len(x) < 4:
                raise ValueError("Need at least 4 valid samples.")

            # ── Step 2: Build grid ────────────────────────────────────────
            self.log.emit("Step 2: Building grid…")
            nx, ny = p['grid_nx'], p['grid_ny']
            xmar = (x.max() - x.min()) * 0.05
            ymar = (y.max() - y.min()) * 0.05
            grid_x = np.linspace(x.min() - xmar, x.max() + xmar, nx)
            grid_y = np.linspace(y.min() - ymar, y.max() + ymar, ny)

            # ── Step 3: Kriging surrogate (value + uncertainty) ───────────
            # Kriging is unbounded, so bounded targets are kriged in a
            # transformed space and back-transformed for display:
            #   log   – skewed non-negative concentrations (log1p/expm1)
            #   logit – probabilities; keeps predictions inside [0, 1]
            transform = p.get('transform', 'none')
            if transform == 'log':
                if np.nanmin(z) < 0:
                    raise ValueError("Log transform requires non-negative target values.")
                z_work = np.log1p(z)
            elif transform == 'logit':
                if np.nanmin(z) < 0 or np.nanmax(z) > 1:
                    raise ValueError("Logit transform requires target values in [0, 1].")
                eps = 1e-4
                zc = np.clip(z, eps, 1 - eps)
                z_work = np.log(zc / (1 - zc))
            else:
                z_work = z
            self.log.emit(f"Step 3: Kriging surrogate ({p['variogram_model']}"
                          f"{', ' + transform if transform != 'none' else ''})…")
            ok = OrdinaryKriging(
                x, y, z_work,
                variogram_model=p['variogram_model'],
                nlags=6, verbose=False, enable_plotting=False, pseudo_inv=True,
            )
            z_grid_work, ss_grid = ok.execute('grid', grid_x, grid_y)

            def _back(a):
                # transformed (kriging) space -> display units
                if transform == 'log':
                    return np.expm1(a)
                if transform == 'logit':
                    return 1.0 / (1.0 + np.exp(-a))
                return a

            z_grid = _back(z_grid_work)
            sigma = np.sqrt(np.clip(ss_grid, 0, None))                 # kriging std-dev
            self.log.emit(f"  Predicted range: [{z_grid.min():.3f}, {z_grid.max():.3f}]")

            # ── Step 4: Acquisition (UCB-style explore/exploit blend) ─────
            w = float(p['explore_weight'])     # 0 = exploit value, 1 = explore uncertainty
            self.log.emit(f"Step 4: Acquisition surface (explore weight = {w:.2f})…")

            def _norm(a):
                lo, hi = np.nanmin(a), np.nanmax(a)
                return np.zeros_like(a) if hi - lo < 1e-12 else (a - lo) / (hi - lo)

            value_norm = _norm(z_grid_work)    # rank value in kriging space
            sigma_norm = _norm(sigma)
            priority = (1.0 - w) * value_norm + w * sigma_norm

            # ── Step 5: Candidate mask (surveyed area + avoid resampling) ─
            gx, gy = np.meshgrid(grid_x, grid_y)
            flat_xy = np.column_stack([gx.ravel(), gy.ravel()])
            tree = cKDTree(np.column_stack([x, y]))

            # avoid radius: keep recommendations away from existing samples
            nn = tree.query(np.column_stack([x, y]), k=2)[0][:, 1]
            auto_avoid = float(np.median(nn)) if len(x) > 1 else 0.0
            avoid_r = p['avoid_radius'] if p['avoid_radius'] > 0 else auto_avoid
            dist_to_sample = tree.query(flat_xy, k=1)[0].reshape(gx.shape)

            mask = dist_to_sample >= avoid_r
            if p.get('restrict_hull', True):
                hull = Delaunay(np.column_stack([x, y]))
                inside = (hull.find_simplex(flat_xy) >= 0).reshape(gx.shape)
                mask &= inside
            self.log.emit(f"  Avoid radius: {avoid_r:.3f}  |  candidate cells: {int(mask.sum())}")

            pri_masked = np.where(mask, priority, np.nan)
            if not np.isfinite(pri_masked).any():
                raise ValueError("No candidate cells left after masking. "
                                 "Lower the avoid radius or disable surveyed-area restriction.")

            # ── Step 6: Pick top-N with minimum separation ────────────────
            n_pts = int(p['n_points'])
            xr, yr = x.max() - x.min(), y.max() - y.min()
            auto_sep = 0.5 * math.sqrt(max(xr * yr, 1e-9) / max(n_pts, 1))
            min_sep = p['min_sep'] if p['min_sep'] > 0 else auto_sep
            self.log.emit(f"Step 5: Selecting up to {n_pts} sites (min spacing {min_sep:.3f})…")

            pr_flat = pri_masked.ravel()
            valid = np.isfinite(pr_flat)
            xs, ys, prs = flat_xy[valid, 0], flat_xy[valid, 1], pr_flat[valid]
            zv = z_grid.ravel()[valid]
            # 95% interval (μ ± 1.96σ, Gaussian in kriging space),
            # back-transformed to display units so it is directly
            # comparable with the predicted value
            zw = np.asarray(z_grid_work).ravel()[valid]
            sv = sigma.ravel()[valid]
            lo = _back(zw - 1.96 * sv)
            hi = _back(zw + 1.96 * sv)
            order = np.argsort(-prs)

            picks = []   # (x, y, value, low, high, score)
            for idx in order:
                px, py = xs[idx], ys[idx]
                if all(math.hypot(px - qx, py - qy) >= min_sep for qx, qy, *_ in picks):
                    picks.append((float(px), float(py), float(zv[idx]),
                                  float(lo[idx]), float(hi[idx]), float(prs[idx])))
                    if len(picks) >= n_pts:
                        break
            self.log.emit(f"  {len(picks)} site(s) selected.")

            picks_df = pd.DataFrame(
                [(i + 1, *pk) for i, pk in enumerate(picks)],
                columns=['rank', x_col, y_col, f'{z_col}_predicted',
                         f'{z_col}_low', f'{z_col}_high', 'priority_score'],
            )

            # ── Step 7: Plots ─────────────────────────────────────────────
            self.log.emit("Step 6: Rendering maps…")

            def _draw_overlay(ax):
                ax.scatter(x, y, c='#38bdf8', s=10, edgecolors='black',
                           linewidths=0.2, label='Existing samples', zorder=4)
                for i, (px, py, *_rest) in enumerate(picks):
                    ax.scatter(px, py, s=170, facecolors='none', edgecolors='white',
                               linewidths=1.8, zorder=6)
                    ax.text(px, py, str(i + 1), color='white', fontsize=9,
                            fontweight='bold', ha='center', va='center', zorder=7)
                ax.set_xlabel(x_col, fontsize=10)
                ax.set_ylabel(y_col, fontsize=10)

            # Priority heatmap
            fig_p, ax_p = plt.subplots(figsize=(8.5, 7))
            im = ax_p.pcolormesh(grid_x, grid_y, pri_masked, shading='auto', cmap='inferno')
            cb = fig_p.colorbar(im, ax=ax_p, shrink=0.85)
            cb.set_label('Dig priority', fontsize=10)
            _draw_overlay(ax_p)
            ax_p.set_title(f'Recommended Dig Sites — {z_col}', fontsize=13, fontweight='bold')
            ax_p.legend(loc='upper right', fontsize=8)
            ax_p.set_aspect('equal')
            fig_p.tight_layout()
            buf = io.BytesIO(); fig_p.savefig(buf, format='png', dpi=150); plt.close(fig_p)
            buf.seek(0); priority_png = buf.read()

            # Overlay on site image (optional)
            overlay_png = None
            img_path = p.get('image_path')
            if img_path:
                try:
                    img = mpimg.imread(img_path)
                    ext_vals = p.get('img_extent')   # (xmin,xmax,ymin,ymax) or None
                    if not ext_vals:
                        # No known georeference: center the image on the data
                        # extent, preserving its pixel aspect ratio so it is
                        # not squished. Placement is approximate.
                        xmin, xmax = grid_x.min(), grid_x.max()
                        ymin, ymax = grid_y.min(), grid_y.max()
                        dw, dh = xmax - xmin, ymax - ymin
                        img_aspect = img.shape[1] / img.shape[0]   # width/height
                        if dw / dh < img_aspect:
                            new_w = dh * img_aspect
                            cx = (xmin + xmax) / 2
                            ext_vals = (cx - new_w / 2, cx + new_w / 2, ymin, ymax)
                        else:
                            new_h = dw / img_aspect
                            cy = (ymin + ymax) / 2
                            ext_vals = (xmin, xmax, cy - new_h / 2, cy + new_h / 2)
                        self.log.emit(
                            "  No image extent given: image centered on data with "
                            "aspect preserved (placement approximate). For an exact "
                            "fit, use a georeferenced image with a world file or "
                            "enter the extent manually.")
                    origin = 'lower' if p.get('img_flip') else 'upper'
                    fig_o, ax_o = plt.subplots(figsize=(8.5, 7))
                    ax_o.imshow(img, extent=ext_vals, origin=origin, aspect='equal')
                    im2 = ax_o.pcolormesh(grid_x, grid_y, pri_masked, shading='auto',
                                          cmap='inferno', alpha=0.55)
                    cb2 = fig_o.colorbar(im2, ax=ax_o, shrink=0.85)
                    cb2.set_label('Dig priority', fontsize=10)
                    _draw_overlay(ax_o)
                    ax_o.set_title('Dig Priority over Site Image', fontsize=13, fontweight='bold')
                    ax_o.legend(loc='upper right', fontsize=8)
                    fig_o.tight_layout()
                    buf = io.BytesIO(); fig_o.savefig(buf, format='png', dpi=150); plt.close(fig_o)
                    buf.seek(0); overlay_png = buf.read()
                except Exception as e:
                    self.log.emit(f"  Image overlay skipped: {e}")

            # ── Save outputs ─────────────────────────────────────────────
            saved = {}
            out_dir = p.get('out_dir')
            if out_dir:
                op = Path(out_dir); op.mkdir(parents=True, exist_ok=True)
                csv_path = str(op / f'next_dig_{z_col}.csv')
                picks_df.to_csv(csv_path, index=False); saved['csv'] = csv_path
                with open(op / f'priority_{z_col}.png', 'wb') as f:
                    f.write(priority_png)
                if overlay_png:
                    with open(op / f'overlay_{z_col}.png', 'wb') as f:
                        f.write(overlay_png)
                self.log.emit(f"  Saved outputs to {out_dir}")

            self.log.emit("Done.")
            self.finished.emit({
                'priority_png': priority_png,
                'overlay_png':  overlay_png,
                'picks_df':     picks_df,
                'n_samples':    len(x),
                'n_picks':      len(picks),
                'explore_weight': w,
                'avoid_radius': avoid_r,
                'min_sep':      min_sep,
                'z_col':        z_col,
            })

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# NEXT-DIG PAGE
# ══════════════════════════════════════════════════════════════════════════════
class NextDigPage(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = self._worker = None
        self._last_picks = None
        self._df = None          # in-memory data sent from another page
        self._build_ui()

    def _build_ui(self):
        page = QVBoxLayout(self)
        page.setContentsMargins(0, 0, 0, 0)

        header = QWidget()
        header.setStyleSheet(f"QWidget {{ background:{PAGE_HEADER_BG}; }}")
        header.setFixedHeight(56)
        hl = QHBoxLayout(header); hl.setContentsMargins(16, 0, 16, 0)
        title = QLabel("Next Dig  –  Adaptive Sampling Recommendations")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{PAGE_HEADER_FG};")
        hl.addWidget(title)
        page.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(12, 12, 12, 12); body.setSpacing(12)

        # ── Left controls ─────────────────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(340)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left = QWidget()
        lv = QVBoxLayout(left); lv.setSpacing(10); lv.setContentsMargins(0, 0, 4, 0)

        # 1. Data
        fb = _section("1. Sample Data")
        fl = QVBoxLayout(fb)
        fr = QHBoxLayout()
        self.file_edit = QLineEdit(); self.file_edit.setPlaceholderText("CSV or Excel…")
        self.file_edit.setReadOnly(True)
        bb = QPushButton("Browse"); bb.setFixedWidth(70); bb.clicked.connect(self._browse_file)
        fr.addWidget(self.file_edit); fr.addWidget(bb); fl.addLayout(fr)
        fl.addWidget(QLabel("X coordinate column:"))
        self.x_combo = QComboBox(); self.x_combo.setEnabled(False); fl.addWidget(self.x_combo)
        fl.addWidget(QLabel("Y coordinate column:"))
        self.y_combo = QComboBox(); self.y_combo.setEnabled(False); fl.addWidget(self.y_combo)
        fl.addWidget(QLabel("Target element / value (Z):"))
        self.z_combo = QComboBox(); self.z_combo.setEnabled(False); fl.addWidget(self.z_combo)
        lv.addWidget(fb)

        # 2. Surrogate model
        mb = _section("2. Surrogate (Kriging)")
        mf = QFormLayout(mb); mf.setSpacing(6)
        self.model_combo = QComboBox(); self.model_combo.addItems(VARIOGRAM_MODELS)
        mf.addRow("Variogram model:", self.model_combo)
        self.transform_combo = QComboBox()
        self.transform_combo.addItems([
            "None",
            "Log — skewed concentrations",
            "Logit — probabilities in [0, 1]",
        ])
        self.transform_combo.setCurrentIndex(1)
        self.transform_combo.setToolTip(
            "Kriging is unbounded, so transform bounded targets:\n"
            "Log for element concentrations (right-skewed, non-negative).\n"
            "Logit for probability targets (e.g. NonSoil_Probability);\n"
            "keeps predictions inside [0, 1].")
        mf.addRow("Target transform:", self.transform_combo)
        self.nx_spin = QSpinBox(); self.nx_spin.setRange(20, 600); self.nx_spin.setValue(120)
        mf.addRow("Grid NX:", self.nx_spin)
        self.ny_spin = QSpinBox(); self.ny_spin.setRange(20, 600); self.ny_spin.setValue(120)
        mf.addRow("Grid NY:", self.ny_spin)
        lv.addWidget(mb)

        # 3. Strategy
        sb = _section("3. Strategy")
        sl = QVBoxLayout(sb)
        sl.addWidget(QLabel("Explore vs exploit:"))
        self.w_slider = QSlider(Qt.Orientation.Horizontal)
        self.w_slider.setRange(0, 100); self.w_slider.setValue(50)
        self.w_slider.valueChanged.connect(self._update_w_label)
        sl.addWidget(self.w_slider)
        self.w_label = QLabel()
        self.w_label.setStyleSheet(f"color:{TEXT_SECONDARY};")
        self.w_label.setFont(QFont("Segoe UI", 9))
        sl.addWidget(self.w_label)
        self._update_w_label(50)

        sf = QFormLayout(); sf.setSpacing(6)
        self.n_spin = QSpinBox(); self.n_spin.setRange(1, 50); self.n_spin.setValue(5)
        self.n_spin.setToolTip("How many dig sites to recommend.")
        sf.addRow("Sites to recommend:", self.n_spin)
        self.sep_spin = QDoubleSpinBox(); self.sep_spin.setRange(0.0, 1e9)
        self.sep_spin.setDecimals(3); self.sep_spin.setValue(0.0)
        self.sep_spin.setToolTip("Minimum spacing between recommended sites,\n"
                                 "in coordinate units. 0 = auto.")
        sf.addRow("Min spacing (0=auto):", self.sep_spin)
        self.avoid_spin = QDoubleSpinBox(); self.avoid_spin.setRange(0.0, 1e9)
        self.avoid_spin.setDecimals(3); self.avoid_spin.setValue(0.0)
        self.avoid_spin.setToolTip("Keep recommendations at least this far from\n"
                                   "existing samples. 0 = auto (median sample spacing).")
        sf.addRow("Avoid radius (0=auto):", self.avoid_spin)
        sl.addLayout(sf)
        self.hull_check = QCheckBox("Restrict to surveyed area (convex hull)")
        self.hull_check.setChecked(True)
        self.hull_check.setToolTip("Keeps recommendations inside the area covered by\n"
                                   "existing samples (avoids edge/extrapolation artifacts).")
        sl.addWidget(self.hull_check)
        lv.addWidget(sb)

        # 4. Site image (optional)
        ib = _section("4. Site Image (optional)")
        il = QVBoxLayout(ib)
        ir = QHBoxLayout()
        self.img_edit = QLineEdit(); self.img_edit.setPlaceholderText("Overhead image (PNG/JPG)…")
        self.img_edit.setReadOnly(True)
        ibtn = QPushButton("Browse"); ibtn.setFixedWidth(70); ibtn.clicked.connect(self._browse_image)
        ir.addWidget(self.img_edit); ir.addWidget(ibtn); il.addLayout(ir)
        il.addWidget(QLabel("Image world extent (auto-filled from world file\n"
                            "if present; blank = centered on data):"))
        ef = QFormLayout(); ef.setSpacing(4)
        self.ext_xmin = QLineEdit(); self.ext_xmin.setPlaceholderText("auto")
        self.ext_xmax = QLineEdit(); self.ext_xmax.setPlaceholderText("auto")
        self.ext_ymin = QLineEdit(); self.ext_ymin.setPlaceholderText("auto")
        self.ext_ymax = QLineEdit(); self.ext_ymax.setPlaceholderText("auto")
        ef.addRow("X min:", self.ext_xmin); ef.addRow("X max:", self.ext_xmax)
        ef.addRow("Y min:", self.ext_ymin); ef.addRow("Y max:", self.ext_ymax)
        il.addLayout(ef)
        self.flip_check = QCheckBox("Flip image vertically")
        self.flip_check.setToolTip("Toggle if the overlay appears upside-down relative\n"
                                   "to the sample coordinates.")
        il.addWidget(self.flip_check)
        lv.addWidget(ib)

        # 5. Output
        ob = _section("5. Output Folder (optional)")
        ol = QVBoxLayout(ob); orow = QHBoxLayout()
        self.out_edit = QLineEdit(); self.out_edit.setPlaceholderText("Select to save outputs…")
        self.out_edit.setReadOnly(True)
        obtn = QPushButton("Browse"); obtn.setFixedWidth(70); obtn.clicked.connect(self._browse_out)
        orow.addWidget(self.out_edit); orow.addWidget(obtn); ol.addLayout(orow)
        lv.addWidget(ob)

        self.run_btn = _primary_btn("▶  Recommend Dig Sites")
        self.run_btn.clicked.connect(self._start)
        lv.addWidget(self.run_btn)

        self.progress = QProgressBar(); self.progress.setRange(0, 0)
        self.progress.setVisible(False); lv.addWidget(self.progress)

        self.export_btn = _primary_btn("💾  Export Recommendations CSV",
                                       color=ACCENT_ALT, hover=ACCENT_ALT_H)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_csv)
        lv.addWidget(self.export_btn)

        lv.addStretch()
        left_scroll.setWidget(left)
        body.addWidget(left_scroll)

        # ── Right results ─────────────────────────────────────────────────
        right = QSplitter(Qt.Orientation.Vertical)

        log_box = _section("Log")
        ll = QVBoxLayout(log_box)
        self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9)); self.log_edit.setFixedHeight(120)
        ll.addWidget(self.log_edit)
        right.addWidget(log_box)

        res_scroll = QScrollArea(); res_scroll.setWidgetResizable(True)
        res = QWidget(); self.res_layout = QVBoxLayout(res); self.res_layout.setSpacing(10)

        self.metrics_label = QLabel("Load samples and run to see recommendations.")
        self.metrics_label.setWordWrap(True); self.metrics_label.setFont(QFont("Segoe UI", 10))
        self.res_layout.addWidget(self.metrics_label)

        self.res_layout.addWidget(_bold_label("Recommended Sites"))
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Rank", "X", "Y", "Predicted", "Low (95%)", "High (95%)", "Score"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setFixedHeight(180)
        self.res_layout.addWidget(self.table)

        self.res_layout.addWidget(_bold_label("Dig Priority Map"))
        self.priority_img = QLabel(); self.priority_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.priority_img.setStyleSheet(IMG_BORDER_STYLE); self.priority_img.setMinimumHeight(320)
        self.res_layout.addWidget(self.priority_img)

        self.overlay_title = _bold_label("Overlay on Site Image")
        self.overlay_title.setVisible(False)
        self.res_layout.addWidget(self.overlay_title)
        self.overlay_img = QLabel(); self.overlay_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_img.setStyleSheet(IMG_BORDER_STYLE); self.overlay_img.setMinimumHeight(320)
        self.overlay_img.setVisible(False)
        self.res_layout.addWidget(self.overlay_img)

        self.res_layout.addStretch()
        res_scroll.setWidget(res)
        right.addWidget(res_scroll)
        right.setSizes([160, 640])
        body.addWidget(right, 1)

        page.addLayout(body)

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _update_w_label(self, val):
        w = val / 100.0
        if w <= 0.2:
            mode = "find hotspots (exploit value)"
        elif w >= 0.8:
            mode = "reduce uncertainty (explore)"
        else:
            mode = "balanced"
        self.w_label.setText(f"weight = {w:.2f}  →  {mode}")

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Sample Data", "", "Data Files (*.csv *.xlsx *.xls);;All Files (*)")
        if not path:
            return
        self._df = None
        self.file_edit.setText(path)
        try:
            ext = os.path.splitext(path)[1].lower()
            df = pd.read_excel(path, nrows=5) if ext in ('.xlsx', '.xls') \
                 else pd.read_csv(path, nrows=5)
            self._populate_combos(df)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _populate_combos(self, df, z_hints=()):
        cols = [c for c in df.columns
                if pd.to_numeric(df[c], errors='coerce').notna().any()]
        for combo, hints in (
            (self.x_combo, ('X_Coord', 'x_coord', 'X', 'Easting', 'Longitude', 'lon', 'lng')),
            (self.y_combo, ('Y_Coord', 'y_coord', 'Y', 'Northing', 'Latitude', 'lat')),
            (self.z_combo, z_hints),
        ):
            combo.clear(); combo.addItems(cols); combo.setEnabled(True)
            for h in hints:
                i = combo.findText(h)
                if i >= 0:
                    combo.setCurrentIndex(i); break

    def load_dataframe(self, df, source_name):
        """Receive classified results from another page (no file round-trip)."""
        self._df = df
        self.file_edit.setText(f"[{source_name}] {len(df)} rows, in-memory")
        self._populate_combos(df.head(5),
                              z_hints=('NonSoil_Probability', 'Soil_Probability'))
        # Probability targets get the logit transform so kriged
        # predictions stay inside [0, 1].
        self.transform_combo.setCurrentIndex(2)
        self.log_edit.append(
            f"Received {len(df)} rows from {source_name}. "
            "Target defaults to NonSoil_Probability (high = likely material); "
            "logit transform selected so predictions stay in [0, 1].")

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Site Image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*)")
        if path:
            self.img_edit.setText(path)
            self._apply_world_file(path)

    def _apply_world_file(self, img_path):
        """Auto-fill the extent fields from a sidecar world file, if one exists.

        World files (.jgw/.pgw/.tfw/.wld) hold six numbers: x-scale, two
        rotation terms, y-scale, and the world coordinates of the top-left
        pixel center. QGIS and most GIS tools write one when exporting
        georeferenced JPG/PNG/TIF.
        """
        p = Path(img_path)
        ext = p.suffix.lower()
        candidates = [p.with_suffix(ext + 'w'), p.with_suffix('.wld')]
        if len(ext) >= 4:
            # .jpg -> .jgw, .png -> .pgw, .tif/.tiff -> .tfw, .jpeg -> .jgw
            candidates.insert(1, p.with_suffix(ext[:2] + ext[-1] + 'w'))
        wf = next((c for c in candidates if c.exists()), None)
        if wf is None:
            return
        try:
            with open(wf) as f:
                nums = [float(t) for t in f.read().split()[:6]]
            if len(nums) < 6:
                raise ValueError("expected 6 numbers")
            A, D, B, E, C, F = nums
            if abs(D) > 1e-9 or abs(B) > 1e-9:
                QMessageBox.warning(
                    self, "World File",
                    f"{wf.name} contains rotation terms, which are not "
                    "supported. Re-export the image north-up, or enter the "
                    "extent manually.")
                return
            qimg = QImage(img_path)
            if qimg.isNull():
                raise ValueError("could not read image dimensions")
            w_px, h_px = qimg.width(), qimg.height()
            x_edges = (C - A / 2, C - A / 2 + A * w_px)
            y_edges = (F - E / 2, F - E / 2 + E * h_px)
            self.ext_xmin.setText(f"{min(x_edges):.3f}")
            self.ext_xmax.setText(f"{max(x_edges):.3f}")
            self.ext_ymin.setText(f"{min(y_edges):.3f}")
            self.ext_ymax.setText(f"{max(y_edges):.3f}")
            # Negative y-scale (the normal case) means row 0 is the north
            # edge, which matches the default top-down drawing; positive
            # means the image is stored bottom-up and needs the flip.
            self.flip_check.setChecked(E > 0)
            self.log_edit.append(
                f"World file {wf.name} found: image extent auto-filled "
                f"(X {min(x_edges):.2f}..{max(x_edges):.2f}, "
                f"Y {min(y_edges):.2f}..{max(y_edges):.2f}).")
        except Exception as e:
            QMessageBox.warning(
                self, "World File",
                f"Found {wf.name} but could not parse it:\n{e}\n\n"
                "Enter the extent manually.")

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.out_edit.setText(path)

    def _extent(self):
        vals = []
        for w in (self.ext_xmin, self.ext_xmax, self.ext_ymin, self.ext_ymax):
            t = w.text().strip()
            if t == "":
                return None
            vals.append(float(t))
        return tuple(vals)

    # ── Run ───────────────────────────────────────────────────────────────
    def _start(self):
        path = self.file_edit.text().strip()
        if not path and self._df is None:
            QMessageBox.warning(self, "Missing", "Please select a data file."); return
        xc, yc, zc = self.x_combo.currentText(), self.y_combo.currentText(), self.z_combo.currentText()
        if not (xc and yc and zc):
            QMessageBox.warning(self, "Missing", "Select X, Y, and target columns."); return
        if len({xc, yc, zc}) < 3:
            QMessageBox.warning(self, "Invalid", "X, Y, and target must be different."); return
        try:
            img_extent = self._extent()
        except ValueError:
            QMessageBox.warning(self, "Invalid", "Image extent values must be numbers (or all blank)."); return

        params = {
            'csv_path': path, 'df': self._df,
            'x_col': xc, 'y_col': yc, 'z_col': zc,
            'variogram_model': self.model_combo.currentText(),
            'transform': ('none', 'log', 'logit')[self.transform_combo.currentIndex()],
            'grid_nx': self.nx_spin.value(), 'grid_ny': self.ny_spin.value(),
            'explore_weight': self.w_slider.value() / 100.0,
            'n_points': self.n_spin.value(),
            'min_sep': self.sep_spin.value(),
            'avoid_radius': self.avoid_spin.value(),
            'restrict_hull': self.hull_check.isChecked(),
            'image_path': self.img_edit.text().strip() or None,
            'img_extent': img_extent,
            'img_flip': self.flip_check.isChecked(),
            'out_dir': self.out_edit.text().strip() or None,
        }

        self.log_edit.clear()
        self.metrics_label.setText("Working…")
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.export_btn.setEnabled(False)

        self._thread = QThread()
        self._worker = NextDigWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log_edit.append)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(lambda: (
            self.run_btn.setEnabled(True), self.progress.setVisible(False)))
        self._thread.start()

    def _on_finished(self, r):
        self._last_picks = r['picks_df']
        self.export_btn.setEnabled(True)
        self.metrics_label.setText(
            f"<b>Samples:</b> {r['n_samples']}  |  "
            f"<b>Sites recommended:</b> {r['n_picks']}  |  "
            f"<b>Explore weight:</b> {r['explore_weight']:.2f}<br>"
            f"<b>Avoid radius:</b> {r['avoid_radius']:.3f}  |  "
            f"<b>Min spacing:</b> {r['min_sep']:.3f}")

        df = r['picks_df']
        self.table.setRowCount(len(df))
        for i, row in df.iterrows():
            vals = [str(int(row['rank'])),
                    f"{row.iloc[1]:.3f}", f"{row.iloc[2]:.3f}",
                    f"{row.iloc[3]:.3f}", f"{row.iloc[4]:.3f}",
                    f"{row.iloc[5]:.3f}", f"{row.iloc[6]:.3f}"]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(i, c, item)

        self._show_png(r['priority_png'], self.priority_img, 640, 420)
        if r.get('overlay_png'):
            self._show_png(r['overlay_png'], self.overlay_img, 640, 420)
            self.overlay_img.setVisible(True); self.overlay_title.setVisible(True)
        else:
            self.overlay_img.clear(); self.overlay_img.setVisible(False)
            self.overlay_title.setVisible(False)

    def _show_png(self, data, widget, w, h):
        pix = QPixmap(); pix.loadFromData(data)
        widget.setPixmap(pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation))

    def _on_error(self, msg):
        self.log_edit.append(f"ERROR:\n{msg}")
        self.metrics_label.setText("Failed. See log.")
        QMessageBox.critical(self, "Error", msg[:600])

    def _export_csv(self):
        if self._last_picks is None:
            QMessageBox.warning(self, "No Data", "Run first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save Recommendations", "",
                                              "CSV Files (*.csv);;All Files (*)")
        if path:
            self._last_picks.to_csv(path, index=False)
            QMessageBox.information(self, "Saved", f"Saved to:\n{path}")
