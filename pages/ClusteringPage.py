import os
import io
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QTabWidget, QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QScrollArea, QProgressBar, QMessageBox, QHeaderView,
    QTextEdit, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QPixmap

from styles import (
    _section, _bold_label, _primary_btn, _h_line, _cell,
    PAGE_HEADER_BG, PAGE_HEADER_FG, IMG_BORDER_STYLE,
    ACCENT_ALT, ACCENT_ALT_H, GREEN, GREEN_HOVER, TEXT_SECONDARY,
    CELL_INFO_BG, CELL_INFO_FG, CELL_WARN_BG, CELL_WARN_FG,
)


# ── Periodic table symbols ─────────────────────────────────────────────────────
PERIODIC_TABLE_ELEMENTS = {
    'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S',
    'Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga',
    'Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd',
    'Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm',
    'Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os',
    'Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa',
    'U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg',
    'Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og'
}

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN WORKER
# ══════════════════════════════════════════════════════════════════════════════
class TrainWorker(QObject):
    finished   = pyqtSignal(dict)
    error      = pyqtSignal(str)
    log        = pyqtSignal(str)


    def __init__(self, params: dict):
        super().__init__()
        self.p = params

    def run(self):
        try:
            import tensorflow as tf
            from tensorflow.keras import callbacks as keras_cb
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import silhouette_score, accuracy_score, f1_score, classification_report
            import hdbscan
            from collections import Counter, defaultdict

            p = self.p
            out_dir = Path(p['out_dir'])
            out_dir.mkdir(parents=True, exist_ok=True)

            seed = 42

            # ── Step 1: load data ─────────────────────────────────────────────
            self.log.emit("Step 1: Loading data…")
            ext = os.path.splitext(p['csv_path'])[1].lower()
            try:
                df = pd.read_excel(p['csv_path']) if ext in ('.xlsx', '.xls') \
                     else pd.read_csv(p['csv_path'])
            except Exception:
                df = pd.read_csv(p['csv_path'], on_bad_lines='skip', low_memory=False)

            feature_cols = p['feature_cols']   # already filtered list
            used_cols = [c for c in feature_cols if c in df.columns]
            self.log.emit(f"  Features ({len(used_cols)}): {', '.join(used_cols)}")

            X_raw = df[used_cols].apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0).values.astype(float)
            n_samples, n_features = X_raw.shape
            self.log.emit(f"  {n_samples} samples, {n_features} features.")

            target_col = p.get('target_col') or None
            has_labels = target_col and target_col in df.columns
            if has_labels:
                y_raw = df[target_col].fillna('unknown').values
                y_binary = np.array(['soil' if m == 'soil' else 'non-soil' for m in y_raw])
            else:
                y_raw = np.array(['unknown'] * n_samples)
                y_binary = None

            coords = None
            if 'X_Coord' in df.columns and 'Y_Coord' in df.columns:
                coords = df[['X_Coord', 'Y_Coord']].values

            # ── Step 2: split ─────────────────────────────────────────────────
            self.log.emit("Step 2: Splitting data…")
            if has_labels and y_raw[0] != 'unknown':
                X_tr, X_te, y_tr, y_te, yb_tr, yb_te, idx_tr, idx_te = train_test_split(
                    X_raw, y_raw, y_binary, np.arange(n_samples),
                    test_size=p['test_size'], random_state=seed, stratify=y_binary
                )
            else:
                X_tr = X_te = X_raw
                y_tr = y_te = np.array(['unknown'] * n_samples)
                yb_tr = yb_te = None
                idx_tr = idx_te = np.arange(n_samples)

            self.log.emit(f"  Train: {len(X_tr)}, Test: {len(X_te)}")

            # ── Step 3: scale ─────────────────────────────────────────────────
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # ── Step 4: autoencoder ───────────────────────────────────────────
            self.log.emit("Step 3: Building autoencoder…")
            latent_dim = p['latent_dim']
            inp = tf.keras.layers.Input(shape=(n_features,))
            x = inp
            for h in (64, 32):
                x = tf.keras.layers.Dense(h, activation='relu')(x)
            z = tf.keras.layers.Dense(latent_dim, activation=None, name='latent')(x)
            x = z
            for h in (32, 64):
                x = tf.keras.layers.Dense(h, activation='relu')(x)
            out_layer = tf.keras.layers.Dense(n_features, activation='linear')(x)
            ae      = tf.keras.Model(inp, out_layer, name='autoencoder')
            encoder = tf.keras.Model(inp, z,         name='encoder')
            ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

            early_stop = keras_cb.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )

            self.log.emit(f"Step 4: Training autoencoder ({p['epochs']} epochs max)…")
            history = ae.fit(
                X_tr_s, X_tr_s,
                validation_split=0.1,
                epochs=p['epochs'],
                batch_size=p['batch_size'],
                callbacks=[early_stop],
                verbose=0
            )

            # Save encoder bytes via temp file (Keras 3 requires a file path)
            import io as _io, tempfile
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as _tmp:
                _tmp_path = _tmp.name
            encoder.save(_tmp_path)
            with open(_tmp_path, 'rb') as _f:
                _encoder_bytes = _f.read()
            os.remove(_tmp_path)
            self.log.emit(f"  Autoencoder trained. Stopped at epoch {len(history.history['loss'])}.")

            # Save training loss plot
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label='Train loss', linewidth=2)
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Val loss', linewidth=2)
            plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
            plt.title('Autoencoder Training Progress', fontsize=14, fontweight='bold')
            plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
            loss_plot_path = str(out_dir / 'training_loss.png')
            plt.savefig(loss_plot_path, dpi=150)
            plt.close()

            # ── Step 5: latent + PCA ──────────────────────────────────────────
            self.log.emit("Step 5: Encoding to latent space + PCA…")
            Z_tr = encoder.predict(X_tr_s, verbose=0)
            Z_te = encoder.predict(X_te_s, verbose=0)
            np.save(str(out_dir / 'latent_train.npy'), Z_tr)
            np.save(str(out_dir / 'latent_test.npy'),  Z_te)

            pca_latent = PCA(n_components=min(5, latent_dim), random_state=seed)
            Z_tr_pca = pca_latent.fit_transform(Z_tr)
            Z_te_pca = pca_latent.transform(Z_te)

            # ── Step 6: HDBSCAN ───────────────────────────────────────────────
            self.log.emit("Step 6: Running HDBSCAN…")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=p['min_cluster_size'],
                min_samples=p['min_samples'],
                prediction_data=True
            )
            cl_tr = clusterer.fit_predict(Z_tr_pca)
            unique_cl = np.unique(cl_tr)
            self.log.emit(f"  Unique cluster labels: {unique_cl.tolist()}")

            real_clusters = [l for l in unique_cl if l != -1]
            if not real_clusters:
                self.log.emit("  WARNING: No clusters found — all points marked as noise.")
                cl_te = np.full(len(Z_te_pca), -1, dtype=int)
            else:
                cl_te, _ = hdbscan.approximate_predict(clusterer, Z_te_pca)

            with open(out_dir / 'hdbscan_model.pkl', 'wb') as f:
                pickle.dump(clusterer, f)

            # ── Compute cluster centroids for KNN approximation ───────────────
            unique_labels  = np.unique(cl_tr)
            real_labels    = [int(l) for l in unique_labels if l != -1]
            centroids      = np.array([
                clusterer.weighted_cluster_centroid(l) for l in real_labels
            ]) if real_labels else np.empty((0, Z_tr_pca.shape[1]))
            centroid_labels = np.array(real_labels, dtype=int)

            # Noise threshold: mean + threshold_sigma * std of min distances to centroids
            noise_sigma = p.get('noise_threshold_sigma', 2.0)
            if len(centroids) > 0:
                dists_tr = np.min(
                    np.linalg.norm(Z_tr_pca[:, None, :] - centroids[None, :, :], axis=2), axis=1
                )
                noise_threshold = float(np.mean(dists_tr) + noise_sigma * np.std(dists_tr))
            else:
                noise_threshold = float('inf')

            pd.DataFrame({'cluster': cl_tr}).to_csv(out_dir / 'hdbscan_labels_train.csv', index=False)
            pd.DataFrame({'cluster': cl_te}).to_csv(out_dir / 'hdbscan_labels_test.csv',  index=False)

            # ── Step 7: Evaluation + classifiers (if labels) ─────────────────
            results = {}
            if has_labels and y_tr[0] != 'unknown':
                self.log.emit("Step 7: Evaluating clusters + training classifiers…")

                valid_tr = cl_tr != -1
                if valid_tr.sum() >= 2:
                    results['silhouette_train'] = float(silhouette_score(Z_tr_pca[valid_tr], cl_tr[valid_tr]))
                valid_te = cl_te != -1
                if valid_te.sum() >= 2:
                    results['silhouette_test'] = float(silhouette_score(Z_te_pca[valid_te], cl_te[valid_te]))

                # Majority-vote cluster map
                cluster_to_mats = defaultdict(list)
                for cl, mat in zip(cl_tr, y_tr):
                    if cl != -1:
                        cluster_to_mats[cl].append(mat)
                cluster_map = {cl: Counter(mats).most_common(1)[0][0]
                               for cl, mats in cluster_to_mats.items()}

                # Binary classifier
                clf_binary = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=10)
                clf_binary.fit(Z_tr, yb_tr)
                yb_pred = clf_binary.predict(Z_te)
                results['binary_accuracy'] = float(accuracy_score(yb_te, yb_pred))
                results['binary_f1'] = float(f1_score(yb_te, yb_pred, pos_label='non-soil', zero_division=0))
                results['binary_report'] = classification_report(yb_te, yb_pred, output_dict=True, zero_division=0)

                # Multi-class classifier
                clf_multi = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=10)
                clf_multi.fit(Z_tr, y_tr)
                ym_pred = clf_multi.predict(Z_te)
                results['multi_accuracy'] = float(accuracy_score(y_te, ym_pred))
                results['multi_f1'] = float(f1_score(y_te, ym_pred, average='weighted', zero_division=0))
                results['multi_report'] = classification_report(y_te, ym_pred, output_dict=True, zero_division=0)

                self.log.emit(f"  Binary accuracy: {results['binary_accuracy']:.4f}")
                self.log.emit(f"  Multi accuracy:  {results['multi_accuracy']:.4f}")
            else:
                self.log.emit("Step 7: No labels — skipping evaluation & classifiers.")

            # ── Step 8: Cluster visualization ─────────────────────────────────
            self.log.emit("Step 8: Generating cluster visualization…")
            pca2 = PCA(n_components=2, random_state=seed)
            Z2 = pca2.fit_transform(Z_tr_pca)
            cluster_plot_path = str(out_dir / 'latent_pca_hdbscan_train.png')

            fig, ax = plt.subplots(figsize=(10, 7))
            colors = plt.cm.tab20.colors
            for i, cl in enumerate(np.unique(cl_tr)):
                mask = cl_tr == cl
                label = f'Cluster {cl}' if cl != -1 else 'Noise'
                color = '#aaaaaa' if cl == -1 else colors[i % len(colors)]
                ax.scatter(Z2[mask, 0], Z2[mask, 1], label=label, color=color,
                           s=30, alpha=0.6, edgecolors='black', linewidths=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
            ax.set_title('HDBSCAN Clusters on PCA(Latent) — Training Set',
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(cluster_plot_path, dpi=150)
            plt.close(fig)

            # ── Step 9: Save cluster assignment CSVs ──────────────────────────
            self.log.emit("Step 9: Saving cluster assignment CSVs…")
            tr_df = pd.DataFrame({'cluster': cl_tr})
            te_df = pd.DataFrame({'cluster': cl_te})
            if has_labels and y_tr[0] != 'unknown':
                tr_df['material'] = y_tr
                te_df['material'] = y_te
            if coords is not None:
                coords_tr = coords[idx_tr]
                coords_te = coords[idx_te]
                tr_df['X_Coord'] = coords_tr[:, 0]
                tr_df['Y_Coord'] = coords_tr[:, 1]
                te_df['X_Coord'] = coords_te[:, 0]
                te_df['Y_Coord'] = coords_te[:, 1]
            tr_df.to_csv(out_dir / 'cluster_assignments_train.csv', index=False)
            te_df.to_csv(out_dir / 'cluster_assignments_test.csv',  index=False)

            # ── Save single model bundle ──────────────────────────────────────
            self.log.emit("Saving model_bundle.pkl…")
            model_bundle = {
                'encoder_bytes':     _encoder_bytes,
                'scaler':            scaler,
                'pca_latent':        pca_latent,
                'hdbscan':           clusterer,
                'centroids':         centroids,
                'centroid_labels':   centroid_labels,
                'noise_threshold':   noise_threshold,
                'binary_clf':        clf_binary if has_labels and y_tr[0] != 'unknown' else None,
                'multi_clf':         clf_multi  if has_labels and y_tr[0] != 'unknown' else None,
                'feature_cols':      used_cols,
                'n_features':        n_features,
                'latent_dim':        latent_dim,
            }
            with open(out_dir / 'model_bundle.pkl', 'wb') as f:
                pickle.dump(model_bundle, f)
            self.log.emit("  model_bundle.pkl saved.")

            results['n_clusters']        = len(real_clusters)
            results['n_noise']           = int((cl_tr == -1).sum())
            results['n_train']           = len(X_tr)
            results['n_test']            = len(X_te)
            results['used_cols']         = used_cols
            results['loss_plot']         = loss_plot_path
            results['cluster_plot']      = cluster_plot_path
            results['has_labels']        = has_labels
            results['cluster_counts_tr'] = {int(k): int(v) for k, v in
                                             zip(*np.unique(cl_tr, return_counts=True))}
            results['out_dir']           = str(out_dir)
            results['model_bundle']      = model_bundle

            self.log.emit("Pipeline complete!")
            self.finished.emit(results)

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT WORKER  (desktop → mobile_export/)
# ══════════════════════════════════════════════════════════════════════════════
class ExportWorker(QObject):
    finished = pyqtSignal(str)   # path to mobile_export dir
    error    = pyqtSignal(str)
    log      = pyqtSignal(str)

    def __init__(self, bundle: dict, out_dir: str):
        super().__init__()
        self.bundle  = bundle
        self.out_dir = Path(out_dir)

    def run(self):
        try:
            import tensorflow as tf
            import onnx
            import tf2onnx
            import json
            import io as _io
            import numpy as np
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            b          = self.bundle
            mobile_dir = self.out_dir / 'mobile_export'
            mobile_dir.mkdir(parents=True, exist_ok=True)

            n_features = b['n_features']
            latent_dim = b['latent_dim']

            # ── 1. Reconstruct encoder from bytes ─────────────────────────────
            self.log.emit("Loading encoder from bundle…")
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as _tmp:
                _tmp.write(b['encoder_bytes'])
                _tmp_path = _tmp.name
            encoder = tf.keras.models.load_model(_tmp_path)
            os.remove(_tmp_path)

            # ── 2. Encoder → ONNX ─────────────────────────────────────────────
            self.log.emit("Converting encoder → ONNX…")
            input_spec = (tf.TensorSpec((None, n_features), tf.float32, name='input'),)
            onnx_enc, _ = tf2onnx.convert.from_keras(encoder, input_signature=input_spec)
            onnx.save(onnx_enc, str(mobile_dir / 'encoder.onnx'))
            self.log.emit("  encoder.onnx saved.")

            # ── 3a. Scaler → ONNX ──────────────────────────────────────────
            self.log.emit("Converting scaler → ONNX…")
            onx_scaler = convert_sklearn(
                b['scaler'],
                initial_types=[('input', FloatTensorType([None, n_features]))]
            )
            with open(mobile_dir / 'scaler.onnx', 'wb') as f:
                f.write(onx_scaler.SerializeToString())
            self.log.emit("  scaler.onnx saved.")

            # ── 3b. PCA → ONNX ────────────────────────────────────────────
            self.log.emit("Converting PCA → ONNX…")
            onx_pca = convert_sklearn(
                b['pca_latent'],
                initial_types=[('input', FloatTensorType([None, latent_dim]))]
            )
            with open(mobile_dir / 'pca.onnx', 'wb') as f:
                f.write(onx_pca.SerializeToString())
            self.log.emit("  pca.onnx saved.")

            # ── 4. Classifiers → ONNX (optional) ──────────────────────────────
            if b.get('binary_clf') is not None:
                self.log.emit("Converting binary classifier → ONNX…")
                onx_bin = convert_sklearn(
                    b['binary_clf'],
                    initial_types=[('latent', FloatTensorType([None, latent_dim]))]
                )
                with open(mobile_dir / 'binary_classifier.onnx', 'wb') as f:
                    f.write(onx_bin.SerializeToString())
                self.log.emit("  binary_classifier.onnx saved.")

            if b.get('multi_clf') is not None:
                self.log.emit("Converting multiclass classifier → ONNX…")
                onx_multi = convert_sklearn(
                    b['multi_clf'],
                    initial_types=[('latent', FloatTensorType([None, latent_dim]))]
                )
                with open(mobile_dir / 'multiclass_classifier.onnx', 'wb') as f:
                    f.write(onx_multi.SerializeToString())
                self.log.emit("  multiclass_classifier.onnx saved.")

            # ── 5. Metadata JSON (includes centroids for Java) ─────────────
            self.log.emit("Saving metadata + centroids → JSON…")
            meta = {
                'feature_cols':      b['feature_cols'],
                'n_features':        int(n_features),
                'latent_dim':        int(latent_dim),
                'noise_threshold':   float(b['noise_threshold']),
                'has_binary_clf':    b.get('binary_clf') is not None,
                'has_multi_clf':     b.get('multi_clf')  is not None,
                'centroids':         b['centroids'].tolist(),
                'centroid_labels':   b['centroid_labels'].tolist(),
            }
            with open(mobile_dir / 'metadata.json', 'w') as f:
                json.dump(meta, f, indent=2)
            self.log.emit("  metadata.json saved.")

            self.log.emit(f"Mobile export complete → {mobile_dir}")
            self.log.emit("\nExported files:")
            self.log.emit("  • encoder.onnx       (autoencoder latent encoder)")
            self.log.emit("  • scaler.onnx        (StandardScaler, n_features input)")
            self.log.emit("  • pca.onnx           (PCA, latent_dim input)")
            if b.get('binary_clf') is not None:
                self.log.emit("  • binary_classifier.onnx")
            if b.get('multi_clf') is not None:
                self.log.emit("  • multiclass_classifier.onnx")
            self.log.emit("  • metadata.json      (feature cols, centroids, thresholds)")
            self.finished.emit(str(mobile_dir))

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# APPLY WORKER
# ══════════════════════════════════════════════════════════════════════════════
class ApplyWorker(QObject):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    log      = pyqtSignal(str)

    def __init__(self, model_dir: str, csv_path: str):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.csv_path  = csv_path

    def run(self):
        try:
            import tensorflow as tf
            import io as _io

            d = self.model_dir
            self.log.emit("Loading model_bundle.pkl…")

            with open(d / 'model_bundle.pkl', 'rb') as f:
                bundle = pickle.load(f)

            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as _tmp:
                _tmp.write(bundle['encoder_bytes'])
                _tmp_path = _tmp.name
            encoder    = tf.keras.models.load_model(_tmp_path)
            os.remove(_tmp_path)
            scaler     = bundle['scaler']
            pca_latent = bundle['pca_latent']
            centroids  = bundle['centroids']       # shape (n_clusters, pca_dims)
            cent_labels = bundle['centroid_labels'] # int array
            noise_thr  = bundle['noise_threshold']
            clf_binary = bundle.get('binary_clf')
            clf_multi  = bundle.get('multi_clf')
            used_cols  = bundle.get('feature_cols')
            n_features = bundle['n_features']
            self.log.emit("  Bundle loaded.")

            # ── Load new data ─────────────────────────────────────────────────
            self.log.emit("Loading new data…")
            ext = os.path.splitext(self.csv_path)[1].lower()
            df = pd.read_excel(self.csv_path) if ext in ('.xlsx', '.xls') else pd.read_csv(self.csv_path)

            # Infer feature cols — prefer stored list from bundle
            if used_cols:
                avail = [c for c in used_cols if c in df.columns]
                if len(avail) < n_features:
                    raise ValueError(
                        f"Model expects features {used_cols} but only {avail} found in file."
                    )
                used_cols = avail
            else:
                avail_elem = [c for c in df.columns if c in PERIODIC_TABLE_ELEMENTS]
                used_cols  = avail_elem[:n_features]
                if len(used_cols) < n_features:
                    raise ValueError(
                        f"Model expects {n_features} features but only {len(used_cols)} "
                        f"element columns found in file."
                    )
            self.log.emit(f"  Using {len(used_cols)} features.")

            X   = df[used_cols].apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0).values.astype(float)
            X_s = scaler.transform(X)

            # ── Encode → PCA ──────────────────────────────────────────────────
            self.log.emit("Encoding → PCA → centroid KNN clustering…")
            Z     = encoder.predict(X_s, verbose=0)
            Z_pca = pca_latent.transform(Z)

            # ── Centroid KNN with noise threshold ─────────────────────────────
            if len(centroids) == 0:
                cl = np.full(len(Z_pca), -1, dtype=int)
            else:
                dists = np.linalg.norm(Z_pca[:, None, :] - centroids[None, :, :], axis=2)
                nearest_idx  = np.argmin(dists, axis=1)
                nearest_dist = dists[np.arange(len(Z_pca)), nearest_idx]
                cl = np.where(nearest_dist <= noise_thr, cent_labels[nearest_idx], -1)

            # ── Optional classification ───────────────────────────────────────
            binary_pred = multi_pred = None
            if clf_binary:
                binary_pred = clf_binary.predict(Z)
                self.log.emit("  Binary predictions generated.")
            if clf_multi:
                multi_pred = clf_multi.predict(Z)
                self.log.emit("  Multiclass predictions generated.")

            # ── PCA 2D visualization ──────────────────────────────────────────
            self.log.emit("Generating cluster plot…")
            from sklearn.decomposition import PCA as _PCA
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            pca2 = _PCA(n_components=2, random_state=42)
            Z2 = pca2.fit_transform(Z_pca)

            fig, ax = plt.subplots(figsize=(10, 7))
            colors = plt.cm.tab20.colors
            for i, c in enumerate(np.unique(cl)):
                mask  = cl == c
                label = f'Cluster {c}' if c != -1 else 'Noise'
                color = '#aaaaaa' if c == -1 else colors[i % len(colors)]
                ax.scatter(Z2[mask, 0], Z2[mask, 1], label=label, color=color,
                           s=30, alpha=0.6, edgecolors='black', linewidths=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
            ax.set_title('HDBSCAN Clusters — New Data', fontsize=13, fontweight='bold')
            ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150)
            plt.close(fig)
            buf.seek(0)
            plot_bytes = buf.read()

            # ── Build results DataFrame ───────────────────────────────────────
            result_df = df.copy()
            result_df['Cluster'] = cl
            if binary_pred is not None:
                result_df['Binary_Prediction'] = binary_pred
            if multi_pred is not None:
                result_df['Material_Prediction'] = multi_pred

            cluster_counts = {int(k): int(v) for k, v in
                              zip(*np.unique(cl, return_counts=True))}

            self.log.emit(f"Done. {len(df)} samples processed.")
            self.finished.emit({
                'result_df':      result_df,
                'cluster_counts': cluster_counts,
                'has_binary':     clf_binary is not None,
                'has_multi':      clf_multi  is not None,
                'plot_bytes':     plot_bytes,
            })

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN TAB
# ══════════════════════════════════════════════════════════════════════════════
class TrainTab(QWidget):
    def __init__(self):
        super().__init__()
        self._thread       = self._worker = None
        self._last_bundle  = None
        self._last_out_dir = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ── Left controls ──────────────────────────────────────────────────
        left = QWidget(); left.setFixedWidth(300)
        lv = QVBoxLayout(left); lv.setSpacing(10); lv.setContentsMargins(0,0,0,0)

        # Data file
        fb = _section("1. Data File")
        fl = QVBoxLayout(fb)
        fr = QHBoxLayout()
        self.file_edit = QLineEdit(); self.file_edit.setPlaceholderText("CSV or Excel…")
        self.file_edit.setReadOnly(True)
        bb = QPushButton("Browse"); bb.setFixedWidth(70)
        bb.clicked.connect(self._browse_file)
        fr.addWidget(self.file_edit); fr.addWidget(bb)
        fl.addLayout(fr)

        fl.addWidget(QLabel("Sample-type column (optional):"))
        self.target_combo = QComboBox()
        self.target_combo.setPlaceholderText("None — unlabeled run")
        self.target_combo.setEnabled(False)
        fl.addWidget(self.target_combo)

        fl.addWidget(QLabel("Feature columns to use:"))
        self.feat_combo = QComboBox()
        self.feat_combo.addItem("All detected element columns")
        self.feat_combo.setEnabled(False)
        fl.addWidget(self.feat_combo)
        lv.addWidget(fb)

        # Hyperparameters
        hb = _section("2. Hyperparameters")
        hf = QFormLayout(hb); hf.setSpacing(6)

        self.latent_spin = QSpinBox(); self.latent_spin.setRange(2, 64); self.latent_spin.setValue(8)
        hf.addRow("Latent dim:", self.latent_spin)

        self.epoch_spin = QSpinBox(); self.epoch_spin.setRange(5, 2000); self.epoch_spin.setValue(100)
        hf.addRow("Max epochs:", self.epoch_spin)

        self.batch_spin = QSpinBox(); self.batch_spin.setRange(8, 512); self.batch_spin.setValue(32)
        hf.addRow("Batch size:", self.batch_spin)

        self.test_spin = QDoubleSpinBox(); self.test_spin.setRange(0.05, 0.5)
        self.test_spin.setSingleStep(0.05); self.test_spin.setValue(0.2)
        hf.addRow("Test split:", self.test_spin)

        self.mcs_spin = QSpinBox(); self.mcs_spin.setRange(2, 500); self.mcs_spin.setValue(20)
        hf.addRow("HDBSCAN min cluster:", self.mcs_spin)

        self.ms_spin = QSpinBox(); self.ms_spin.setRange(1, 100); self.ms_spin.setValue(10)
        hf.addRow("HDBSCAN min samples:", self.ms_spin)

        self.noise_sigma_spin = QDoubleSpinBox()
        self.noise_sigma_spin.setRange(0.5, 10.0)
        self.noise_sigma_spin.setSingleStep(0.1)
        self.noise_sigma_spin.setValue(2.0)
        self.noise_sigma_spin.setToolTip(
            "Noise threshold = mean(dist) + σ × std(dist).\n"
            "Higher = more points assigned to clusters, fewer marked noise."
        )
        hf.addRow("Noise threshold (σ):", self.noise_sigma_spin)
        lv.addWidget(hb)

        # Output dir
        ob = _section("3. Output Folder")
        ol = QVBoxLayout(ob)
        or_ = QHBoxLayout()
        self.out_edit = QLineEdit(); self.out_edit.setPlaceholderText("Select output folder…")
        self.out_edit.setReadOnly(True)
        ob2 = QPushButton("Browse"); ob2.setFixedWidth(70)
        ob2.clicked.connect(self._browse_out)
        or_.addWidget(self.out_edit); or_.addWidget(ob2)
        ol.addLayout(or_)
        lv.addWidget(ob)

        # Run button
        self.run_btn = _primary_btn("▶  Run Pipeline")
        self.run_btn.clicked.connect(self._start)
        lv.addWidget(self.run_btn)

        self.progress = QProgressBar(); self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        lv.addWidget(self.progress)

        # Export button (enabled after training completes)
        self.export_btn = _primary_btn("📱  Export for Mobile", color=ACCENT_ALT, hover=ACCENT_ALT_H)
        self.export_btn.setEnabled(False)
        self.export_btn.setToolTip("Export ONNX + centroid files for Android inference.")
        self.export_btn.clicked.connect(self._export_mobile)
        lv.addWidget(self.export_btn)

        self.export_progress = QProgressBar(); self.export_progress.setRange(0, 0)
        self.export_progress.setVisible(False)
        lv.addWidget(self.export_progress)

        lv.addStretch()
        root.addWidget(left)

        # ── Right results ──────────────────────────────────────────────────
        right_split = QSplitter(Qt.Orientation.Vertical)

        # Log + epoch loss
        log_box = _section("Pipeline Log")
        log_layout = QVBoxLayout(log_box)
        self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9)); self.log_edit.setFixedHeight(140)
        log_layout.addWidget(self.log_edit)


        right_split.addWidget(log_box)

        # Metrics + plots
        res_scroll = QScrollArea(); res_scroll.setWidgetResizable(True)
        res_widget = QWidget()
        self.res_layout = QVBoxLayout(res_widget)
        self.res_layout.setSpacing(10)

        self.metrics_label = QLabel("Run the pipeline to see results.")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setFont(QFont("Segoe UI", 10))
        self.res_layout.addWidget(self.metrics_label)

        self.res_layout.addWidget(_bold_label("Cluster Distribution (Training)"))
        self.cluster_table = QTableWidget(0, 2)
        self.cluster_table.setHorizontalHeaderLabels(["Cluster", "Count"])
        self.cluster_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cluster_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.cluster_table.setFixedHeight(160)
        self.res_layout.addWidget(self.cluster_table)

        self.res_layout.addWidget(_bold_label("Training Loss Plot"))
        self.loss_img = QLabel(); self.loss_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loss_img.setStyleSheet(IMG_BORDER_STYLE)
        self.loss_img.setMinimumHeight(220)
        self.res_layout.addWidget(self.loss_img)

        self.res_layout.addWidget(_bold_label("Cluster Visualization"))
        self.cluster_img = QLabel(); self.cluster_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cluster_img.setStyleSheet(IMG_BORDER_STYLE)
        self.cluster_img.setMinimumHeight(280)
        self.res_layout.addWidget(self.cluster_img)

        self.res_layout.addStretch()
        res_scroll.setWidget(res_widget)
        right_split.addWidget(res_scroll)
        right_split.setSizes([200, 600])
        root.addWidget(right_split, 1)

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
            df = pd.read_excel(path, nrows=1) if ext in ('.xlsx', '.xls') \
                 else pd.read_csv(path, nrows=1)
            non_elem = [c for c in df.columns if c not in PERIODIC_TABLE_ELEMENTS]
            elem     = [c for c in df.columns if c in PERIODIC_TABLE_ELEMENTS]
            self.target_combo.clear()
            self.target_combo.addItem("")
            self.target_combo.addItems(non_elem)
            self.target_combo.setEnabled(True)
            self.feat_combo.clear()
            self.feat_combo.addItem("All detected element columns")
            self.feat_combo.addItems(elem)
            self.feat_combo.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.out_edit.setText(path)

    # ── Start ──────────────────────────────────────────────────────────────
    def _start(self):
        csv_path = self.file_edit.text().strip()
        out_dir  = self.out_edit.text().strip()
        if not csv_path:
            QMessageBox.warning(self, "Missing", "Please select a data file."); return
        if not out_dir:
            QMessageBox.warning(self, "Missing", "Please select an output folder."); return

        # Determine feature columns
        ext = os.path.splitext(csv_path)[1].lower()
        try:
            df_head = pd.read_excel(csv_path, nrows=1) if ext in ('.xlsx', '.xls') else pd.read_csv(csv_path, nrows=1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e)); return

        all_elem = [c for c in df_head.columns if c in PERIODIC_TABLE_ELEMENTS]
        if self.feat_combo.currentIndex() == 0 or self.feat_combo.currentText() == "All detected element columns":
            feature_cols = all_elem
        else:
            feature_cols = [self.feat_combo.currentText()]

        target_col = self.target_combo.currentText().strip() or None

        params = {
            'csv_path':              csv_path,
            'out_dir':               out_dir,
            'feature_cols':          feature_cols,
            'target_col':            target_col,
            'latent_dim':            self.latent_spin.value(),
            'epochs':                self.epoch_spin.value(),
            'batch_size':            self.batch_spin.value(),
            'test_size':             self.test_spin.value(),
            'min_cluster_size':      self.mcs_spin.value(),
            'min_samples':           self.ms_spin.value(),
            'noise_threshold_sigma': self.noise_sigma_spin.value(),
        }

        self.log_edit.clear()
        self.metrics_label.setText("Running pipeline…")
        self.run_btn.setEnabled(False); self.progress.setVisible(True)

        self._thread = QThread()
        self._worker = TrainWorker(params)
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

    def _on_finished(self, r):
        # Store bundle + out_dir for later mobile export
        self._last_bundle  = r.get('model_bundle')
        self._last_out_dir = r.get('out_dir')
        self.export_btn.setEnabled(self._last_bundle is not None)

        lines = [
            f"<b>Clusters found:</b> {r['n_clusters']}  |  "
            f"<b>Noise points:</b> {r['n_noise']}  |  "
            f"<b>Train:</b> {r['n_train']}  <b>Test:</b> {r['n_test']}"
        ]
        if r.get('has_labels'):
            if 'silhouette_train' in r:
                lines.append(f"<b>Silhouette (train):</b> {r['silhouette_train']:.4f}  "
                             f"<b>Silhouette (test):</b> {r.get('silhouette_test', 'N/A')}")
            if 'binary_accuracy' in r:
                lines.append(f"<b>Binary RF accuracy:</b> {r['binary_accuracy']:.4f}  "
                             f"<b>F1:</b> {r['binary_f1']:.4f}")
            if 'multi_accuracy' in r:
                lines.append(f"<b>Multiclass RF accuracy:</b> {r['multi_accuracy']:.4f}  "
                             f"<b>F1 (weighted):</b> {r['multi_f1']:.4f}")
        self.metrics_label.setText("<br>".join(lines))

        # Cluster table
        counts = r['cluster_counts_tr']
        self.cluster_table.setRowCount(len(counts))
        for i, (cl, cnt) in enumerate(sorted(counts.items())):
            lbl = "Noise" if cl == -1 else f"Cluster {cl}"
            self.cluster_table.setItem(i, 0, QTableWidgetItem(lbl))
            if cl == -1:
                item = _cell(str(cnt), CELL_WARN_BG, CELL_WARN_FG, center=True)
            else:
                item = _cell(str(cnt), CELL_INFO_BG, CELL_INFO_FG, center=True)
            self.cluster_table.setItem(i, 1, item)

        # Plots
        self._show_image(r['loss_plot'],    self.loss_img,    500, 220)
        self._show_image(r['cluster_plot'], self.cluster_img, 600, 280)

    def _show_image(self, path, label_widget, w, h):
        if path and os.path.exists(path):
            pix = QPixmap(path).scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)
            label_widget.setPixmap(pix)

    def _export_mobile(self):
        if not self._last_bundle or not self._last_out_dir:
            QMessageBox.warning(self, "No Model", "Please run the pipeline first."); return

        self.export_btn.setEnabled(False)
        self.export_progress.setVisible(True)
        self.log_edit.append("\n--- Starting mobile export ---")

        self._exp_thread = QThread()
        self._exp_worker = ExportWorker(self._last_bundle, self._last_out_dir)
        self._exp_worker.moveToThread(self._exp_thread)
        self._exp_thread.started.connect(self._exp_worker.run)
        self._exp_worker.log.connect(self.log_edit.append)
        self._exp_worker.finished.connect(self._on_export_finished)
        self._exp_worker.error.connect(self._on_export_error)
        self._exp_worker.finished.connect(self._exp_thread.quit)
        self._exp_worker.error.connect(self._exp_thread.quit)
        self._exp_thread.finished.connect(lambda: (
            self.export_btn.setEnabled(True),
            self.export_progress.setVisible(False)
        ))
        self._exp_thread.start()

    def _on_export_finished(self, mobile_dir):
        QMessageBox.information(
            self, "Mobile Export Complete",
            f"Files saved to:\n{mobile_dir}\n\n"
            "Contents:\n"
            "  • encoder.onnx\n"
            "  • scaler.onnx\n"
            "  • pca.onnx\n"
            "  • binary_classifier.onnx  (if trained)\n"
            "  • multiclass_classifier.onnx  (if trained)\n"
            "  • metadata.json  (centroids + config)"
        )

    def _on_export_error(self, msg):
        self.log_edit.append(f"EXPORT ERROR:\n{msg}")
        QMessageBox.critical(self, "Export Error", msg[:600])

    def _on_error(self, msg):
        self.log_edit.append(f"ERROR:\n{msg}")
        self.metrics_label.setText("Pipeline failed. See log.")
        QMessageBox.critical(self, "Error", msg[:600])


# ══════════════════════════════════════════════════════════════════════════════
# APPLY TAB
# ══════════════════════════════════════════════════════════════════════════════
class ApplyTab(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = self._worker = None
        self._result_df = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ── Left controls ──────────────────────────────────────────────────
        left = QWidget(); left.setFixedWidth(300)
        lv = QVBoxLayout(left); lv.setSpacing(10); lv.setContentsMargins(0,0,0,0)

        mb = _section("1. Prior Run Output Folder")
        ml = QVBoxLayout(mb)
        ml.addWidget(QLabel("Select the output folder from a previous\ntraining run to load all model files."))
        mr = QHBoxLayout()
        self.model_dir_edit = QLineEdit(); self.model_dir_edit.setPlaceholderText("Select folder…")
        self.model_dir_edit.setReadOnly(True)
        mbtn = QPushButton("Browse"); mbtn.setFixedWidth(70)
        mbtn.clicked.connect(self._browse_model_dir)
        mr.addWidget(self.model_dir_edit); mr.addWidget(mbtn)
        ml.addLayout(mr)
        self.model_status = QLabel("")
        self.model_status.setWordWrap(True)
        self.model_status.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:11px;")
        ml.addWidget(self.model_status)
        lv.addWidget(mb)

        db = _section("2. New Data File")
        dl = QVBoxLayout(db)
        dr = QHBoxLayout()
        self.data_edit = QLineEdit(); self.data_edit.setPlaceholderText("CSV or Excel…")
        self.data_edit.setReadOnly(True)
        dbtn = QPushButton("Browse"); dbtn.setFixedWidth(70)
        dbtn.clicked.connect(self._browse_data)
        dr.addWidget(self.data_edit); dr.addWidget(dbtn)
        dl.addLayout(dr)
        lv.addWidget(db)

        self.apply_btn = _primary_btn("▶  Apply Model", color=GREEN, hover=GREEN_HOVER)
        self.apply_btn.clicked.connect(self._start)
        lv.addWidget(self.apply_btn)

        self.progress = QProgressBar(); self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        lv.addWidget(self.progress)
        lv.addStretch()
        root.addWidget(left)

        # ── Right results ──────────────────────────────────────────────────
        right_split = QSplitter(Qt.Orientation.Vertical)

        log_box = _section("Log")
        log_layout = QVBoxLayout(log_box)
        self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9)); self.log_edit.setFixedHeight(110)
        log_layout.addWidget(self.log_edit)
        right_split.addWidget(log_box)

        res_scroll = QScrollArea(); res_scroll.setWidgetResizable(True)
        res_widget = QWidget()
        rv = QVBoxLayout(res_widget); rv.setSpacing(10)

        self.apply_metrics = QLabel("Apply a model to see results.")
        self.apply_metrics.setWordWrap(True)
        self.apply_metrics.setFont(QFont("Segoe UI", 10))
        rv.addWidget(self.apply_metrics)

        rv.addWidget(_bold_label("Cluster Visualization"))
        self.apply_cluster_img = QLabel()
        self.apply_cluster_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.apply_cluster_img.setStyleSheet(IMG_BORDER_STYLE)
        self.apply_cluster_img.setMinimumHeight(260)
        rv.addWidget(self.apply_cluster_img)

        rv.addWidget(_bold_label("Predictions Table"))
        self.pred_table = QTableWidget(0, 0)
        self.pred_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.pred_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        rv.addWidget(self.pred_table, 1)

        rv.addStretch()
        res_scroll.setWidget(res_widget)
        right_split.addWidget(res_scroll)
        right_split.setSizes([150, 650])
        root.addWidget(right_split, 1)

    def _browse_model_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Prior Output Folder")
        if not path:
            return
        self.model_dir_edit.setText(path)
        p = Path(path)
        if (p / 'model_bundle.pkl').exists():
            # Peek inside bundle to report optional classifiers
            try:
                with open(p / 'model_bundle.pkl', 'rb') as f:
                    b = pickle.load(f)
                status = "✔ model_bundle.pkl found."
                if b.get('binary_clf'):
                    status += "\n+ Binary classifier included."
                if b.get('multi_clf'):
                    status += "\n+ Multiclass classifier included."
            except Exception as e:
                status = f"✔ model_bundle.pkl found (could not inspect: {e})"
        else:
            status = "✘ model_bundle.pkl not found in this folder."
        self.model_status.setText(status)

    def _browse_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open New Data File", "", "Data Files (*.csv *.xlsx *.xls);;All Files (*)"
        )
        if path:
            self.data_edit.setText(path)

    def _start(self):
        model_dir = self.model_dir_edit.text().strip()
        csv_path  = self.data_edit.text().strip()
        if not model_dir:
            QMessageBox.warning(self, "Missing", "Please select a prior output folder."); return
        if not csv_path:
            QMessageBox.warning(self, "Missing", "Please select a data file."); return

        self.log_edit.clear()
        self.apply_btn.setEnabled(False); self.progress.setVisible(True)
        self.apply_metrics.setText("Applying model…")

        self._thread = QThread()
        self._worker = ApplyWorker(model_dir, csv_path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log_edit.append)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(lambda: (
            self.apply_btn.setEnabled(True), self.progress.setVisible(False)
        ))
        self._thread.start()

    def _on_finished(self, r):
        df: pd.DataFrame = r['result_df']
        counts = r['cluster_counts']
        n_clusters = len([k for k in counts if k != -1])
        n_noise    = counts.get(-1, 0)
        lines = [f"<b>Clusters assigned:</b> {n_clusters}  |  <b>Noise:</b> {n_noise}  |  <b>Samples:</b> {len(df)}"]
        if r['has_binary']:
            lines.append("<b>Binary predictions</b> included (Binary_Prediction column).")
        if r['has_multi']:
            lines.append("<b>Multiclass predictions</b> included (Material_Prediction column).")
        self.apply_metrics.setText("<br>".join(lines))

        # Cluster plot from bytes
        pix = QPixmap()
        pix.loadFromData(r['plot_bytes'])
        pix = pix.scaled(600, 280, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
        self.apply_cluster_img.setPixmap(pix)

        # Predictions table
        self.pred_table.setRowCount(len(df))
        self.pred_table.setColumnCount(len(df.columns))
        self.pred_table.setHorizontalHeaderLabels(list(df.columns))
        highlight = {'Cluster', 'Binary_Prediction', 'Material_Prediction'}
        for ci, col in enumerate(df.columns):
            highlighted = col in highlight
            for ri in range(len(df)):
                val  = df.iloc[ri, ci]
                item = _cell(str(val),
                             CELL_INFO_BG if highlighted else None,
                             CELL_INFO_FG)
                self.pred_table.setItem(ri, ci, item)

    def _on_error(self, msg):
        self.log_edit.append(f"ERROR:\n{msg}")
        self.apply_metrics.setText("Failed. See log.")
        QMessageBox.critical(self, "Error", msg[:600])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
class ClusteringPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QWidget()
        header.setStyleSheet(f"QWidget {{ background:{PAGE_HEADER_BG}; }}")
        header.setFixedHeight(56)
        hl = QHBoxLayout(header); hl.setContentsMargins(16, 0, 16, 0)
        title = QLabel("Autoencoder + HDBSCAN  –  Unsupervised pXRF Clustering")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{PAGE_HEADER_FG};")
        hl.addWidget(title)
        layout.addWidget(header)

        tabs = QTabWidget()
        tabs.setFont(QFont("Segoe UI", 10))
        tabs.addTab(TrainTab(), "  Train Pipeline  ")
        tabs.addTab(ApplyTab(), "  Apply to New Data  ")
        layout.addWidget(tabs)