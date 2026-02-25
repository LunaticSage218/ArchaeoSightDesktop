import os
import pickle
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QTabWidget, QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem,
    QScrollArea, QProgressBar, QMessageBox, QHeaderView,
    QRadioButton, QButtonGroup, QTextEdit, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor

# ── Periodic table element symbols used to filter columns ─────────────────────
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


# ── Worker for training in background ─────────────────────────────────────────
class TrainWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, file_path, label_col, params, save_format, model_name, save_dir):
        super().__init__()
        self.file_path = file_path
        self.label_col = label_col
        self.params = params
        self.save_format = save_format
        self.model_name = model_name
        self.save_dir = save_dir

    def run(self):
        print("we ran")
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.metrics import (accuracy_score, classification_report,
                                         confusion_matrix, precision_recall_fscore_support)

            self.log.emit("Loading data…")
            ext = os.path.splitext(self.file_path)[1].lower()
            df = pd.read_excel(self.file_path) if ext in ('.xlsx', '.xls') else pd.read_csv(self.file_path)
            self.log.emit(f"Loaded {len(df)} rows.")

            # Element columns
            element_cols = [c for c in df.columns
                            if c in PERIODIC_TABLE_ELEMENTS and c != self.label_col]
            self.log.emit(f"Element features ({len(element_cols)}): {', '.join(element_cols)}")

            X = df[element_cols].copy().fillna(0)
            X[X < 0] = 0

            y = df[self.label_col].copy().fillna('soil')
            y = y.replace(['blank', 'unknown', ''], 'soil')

            # Encode
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Cross-validation
            self.log.emit("Running 5-fold cross-validation…")
            model_cv = GradientBoostingClassifier(**self.params)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model_cv, X_scaled, y_enc, cv=skf, scoring='accuracy')
            self.log.emit(f"CV scores: {cv_scores.round(4)}  mean={cv_scores.mean():.4f} ±{cv_scores.std()*2:.4f}")

            # Train/test split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

            # Final model
            self.log.emit("Training final model…")
            model = GradientBoostingClassifier(**self.params)
            model.fit(X_tr, y_tr)

            # Evaluate multi-class
            y_pred_enc = model.predict(X_te)
            y_pred = le.inverse_transform(y_pred_enc)
            y_test_labels = le.inverse_transform(y_te)
            acc_multi = accuracy_score(y_test_labels, y_pred)

            # Evaluate binary
            y_proba = model.predict_proba(X_te)
            soil_idx_arr = np.where(le.classes_ == 'soil')[0]
            if len(soil_idx_arr):
                soil_probs = y_proba[:, soil_idx_arr[0]]
            else:
                soil_probs = np.zeros(len(X_te))
            y_bin_pred = np.where(soil_probs > 0.5, 'soil', 'non-soil')
            y_bin_true = np.where(y_test_labels == 'soil', 'soil', 'non-soil')
            acc_binary = accuracy_score(y_bin_true, y_bin_pred)

            # Confusion matrix (multi-class)
            unique_labels = sorted(set(y_test_labels.tolist() + y_pred.tolist()))
            cm = confusion_matrix(y_test_labels, y_pred, labels=unique_labels)

            # Feature importance
            fi = pd.DataFrame({
                'Element': element_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).reset_index(drop=True)

            # Classification report dict
            report = classification_report(y_test_labels, y_pred, output_dict=True, zero_division=0)

            # Save model
            save_obj = {
                'model': model,
                'scaler': scaler,
                'label_encoder': le,
                'element_columns': element_cols,
            }

            if self.save_format == 'pickle':
                out_path = os.path.join(self.save_dir, self.model_name + '.pkl')
                with open(out_path, 'wb') as f:
                    pickle.dump(save_obj, f)
                self.log.emit(f"Model saved (pickle): {out_path}")
            else:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                import onnx

                # ONNX only supports the sklearn model + metadata saved separately
                initial_type = [('float_input', FloatTensorType([None, len(element_cols)]))]
                onnx_model = convert_sklearn(model, initial_types=initial_type)
                onnx_path = os.path.join(self.save_dir, self.model_name + '.onnx')
                with open(onnx_path, 'wb') as f:
                    f.write(onnx_model.SerializeToString())
                # Save metadata as pickle alongside
                meta_path = os.path.join(self.save_dir, self.model_name + '_meta.pkl')
                with open(meta_path, 'wb') as f:
                    pickle.dump({k: v for k, v in save_obj.items() if k != 'model'}, f)
                self.log.emit(f"Model saved (onnx): {onnx_path}")
                self.log.emit(f"Metadata saved: {meta_path}")

            self.finished.emit({
                'acc_multi': acc_multi,
                'acc_binary': acc_binary,
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'cv_scores': cv_scores.tolist(),
                'confusion_matrix': cm.tolist(),
                'confusion_labels': unique_labels,
                'feature_importance': fi,
                'report': report,
            })

        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


# ── Worker for testing in background ──────────────────────────────────────────
class TestWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, model_path, file_path, label_col):
        super().__init__()
        self.model_path = model_path
        self.file_path = file_path
        self.label_col = label_col  # may be None

    def run(self):
        try:
            self.log.emit("Loading model…")
            ext = os.path.splitext(self.model_path)[1].lower()

            if ext == '.pkl':
                with open(self.model_path, 'rb') as f:
                    save_obj = pickle.load(f)
                model = save_obj['model']
                scaler = save_obj['scaler']
                le = save_obj['label_encoder']
                element_cols = save_obj['element_columns']
                use_onnx = False
            elif ext == '.onnx':
                import onnxruntime as rt
                sess = rt.InferenceSession(self.model_path)
                meta_path = self.model_path.replace('.onnx', '_meta.pkl')
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                scaler = meta['scaler']
                le = meta['label_encoder']
                element_cols = meta['element_columns']
                model = sess
                use_onnx = True
            else:
                raise ValueError(f"Unsupported model file: {ext}")

            self.log.emit(f"Model loaded. Features: {', '.join(element_cols)}")

            # Load data
            fext = os.path.splitext(self.file_path)[1].lower()
            df = pd.read_excel(self.file_path) if fext in ('.xlsx', '.xls') else pd.read_csv(self.file_path)
            self.log.emit(f"Data loaded: {len(df)} rows.")

            avail = [c for c in element_cols if c in df.columns]
            missing = [c for c in element_cols if c not in df.columns]
            if missing:
                self.log.emit(f"Warning: missing columns filled with 0: {', '.join(missing)}")

            X = pd.DataFrame(0.0, index=df.index, columns=element_cols)
            for c in avail:
                X[c] = df[c]
            X = X.fillna(0)
            X[X < 0] = 0
            X_scaled = scaler.transform(X)

            if use_onnx:
                inp_name = model.get_inputs()[0].name
                raw = model.run(None, {inp_name: X_scaled.astype(np.float32)})
                y_pred_enc = raw[0]
                # raw[1] is list of dicts for proba in onnxruntime
                proba_list = raw[1]
                classes = le.classes_
                y_proba = np.array([[d[c] for c in classes] for d in proba_list])
            else:
                y_pred_enc = model.predict(X_scaled)
                y_proba = model.predict_proba(X_scaled)

            y_pred = le.inverse_transform(y_pred_enc)

            soil_idx_arr = np.where(le.classes_ == 'soil')[0]
            if len(soil_idx_arr):
                soil_probs = y_proba[:, soil_idx_arr[0]]
            else:
                soil_probs = np.zeros(len(X_scaled))

            confidence = y_proba.max(axis=1)
            binary_pred = np.where(soil_probs > 0.5, 'soil', 'non-soil')

            results_df = df.copy()
            results_df['Predicted_Material'] = y_pred
            results_df['Binary_Prediction'] = binary_pred
            results_df['Confidence'] = confidence.round(4)
            results_df['Soil_Probability'] = soil_probs.round(4)

            metrics = None
            if self.label_col and self.label_col in df.columns:
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                y_true = df[self.label_col].fillna('soil').replace(['blank','unknown',''], 'soil')
                acc_multi = accuracy_score(y_true, y_pred)
                y_bin_true = np.where(y_true == 'soil', 'soil', 'non-soil')
                acc_binary = accuracy_score(y_bin_true, binary_pred)
                unique_labels = sorted(set(y_true.tolist() + y_pred.tolist()))
                cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                metrics = {
                    'acc_multi': acc_multi,
                    'acc_binary': acc_binary,
                    'confusion_matrix': cm.tolist(),
                    'confusion_labels': unique_labels,
                    'report': report,
                }

            self.finished.emit({'results_df': results_df, 'metrics': metrics})

        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


# ── Helpers ────────────────────────────────────────────────────────────────────
def _bold_label(text, size=10):
    lbl = QLabel(text)
    lbl.setFont(QFont("Segoe UI", size, QFont.Weight.Bold))
    return lbl


def _section(title):
    box = QGroupBox(title)
    box.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
    return box


def _h_line():
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


# ── Train Tab ──────────────────────────────────────────────────────────────────
class TrainTab(QWidget):
    def __init__(self):
        super().__init__()
        self._df = None
        self._thread = None
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # ── Left panel: controls ───────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(320)
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # File selection
        file_box = _section("1. Data File")
        file_form = QVBoxLayout(file_box)
        file_row = QHBoxLayout()
        self.train_file_edit = QLineEdit()
        self.train_file_edit.setPlaceholderText("Select CSV or Excel file…")
        self.train_file_edit.setReadOnly(True)
        btn_browse = QPushButton("Browse")
        btn_browse.setFixedWidth(70)
        btn_browse.clicked.connect(self._browse_file)
        file_row.addWidget(self.train_file_edit)
        file_row.addWidget(btn_browse)
        file_form.addLayout(file_row)

        self.label_col_combo = QComboBox()
        self.label_col_combo.setPlaceholderText("Select sample-type column…")
        self.label_col_combo.setEnabled(False)
        file_form.addWidget(QLabel("Sample type column:"))
        file_form.addWidget(self.label_col_combo)
        left_layout.addWidget(file_box)

        # Hyperparameters
        hp_box = _section("2. Hyperparameters")
        hp_form = QFormLayout(hp_box)
        hp_form.setSpacing(6)

        self.n_est_spin = QSpinBox()
        self.n_est_spin.setRange(10, 2000)
        self.n_est_spin.setValue(100)
        hp_form.addRow("n_estimators:", self.n_est_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.001, 1.0)
        self.lr_spin.setSingleStep(0.01)
        self.lr_spin.setDecimals(3)
        self.lr_spin.setValue(0.1)
        hp_form.addRow("learning_rate:", self.lr_spin)

        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 20)
        self.depth_spin.setValue(4)
        hp_form.addRow("max_depth:", self.depth_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        hp_form.addRow("random_state:", self.seed_spin)
        left_layout.addWidget(hp_box)

        # Save options
        save_box = _section("3. Save Model")
        save_layout = QVBoxLayout(save_box)

        fmt_row = QHBoxLayout()
        self.fmt_group = QButtonGroup(self)
        self.rb_pickle = QRadioButton("Pickle (.pkl)")
        self.rb_onnx = QRadioButton("ONNX (.onnx)")
        self.rb_pickle.setChecked(True)
        self.fmt_group.addButton(self.rb_pickle)
        self.fmt_group.addButton(self.rb_onnx)
        fmt_row.addWidget(self.rb_pickle)
        fmt_row.addWidget(self.rb_onnx)
        save_layout.addLayout(fmt_row)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Model name:"))
        self.model_name_edit = QLineEdit("my_gbdt_model")
        name_row.addWidget(self.model_name_edit)
        save_layout.addLayout(name_row)

        dir_row = QHBoxLayout()
        self.save_dir_edit = QLineEdit()
        self.save_dir_edit.setPlaceholderText("Select save directory…")
        self.save_dir_edit.setReadOnly(True)
        btn_dir = QPushButton("Browse")
        btn_dir.setFixedWidth(70)
        btn_dir.clicked.connect(self._browse_save_dir)
        dir_row.addWidget(self.save_dir_edit)
        dir_row.addWidget(btn_dir)
        save_layout.addLayout(dir_row)
        left_layout.addWidget(save_box)

        # Train button + progress
        self.train_btn = QPushButton("▶  Train Model")
        self.train_btn.setFixedHeight(38)
        self.train_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.train_btn.setStyleSheet(
            "QPushButton { background:#2563eb; color:white; border-radius:6px; }"
            "QPushButton:hover { background:#1d4ed8; }"
            "QPushButton:disabled { background:#93c5fd; }"
        )
        self.train_btn.clicked.connect(self._start_training)
        left_layout.addWidget(self.train_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)

        left_layout.addStretch()
        main_layout.addWidget(left)

        # ── Right panel: results ───────────────────────────────────────────
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Log
        log_box = _section("Training Log")
        log_layout = QVBoxLayout(log_box)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9))
        self.log_edit.setFixedHeight(130)
        log_layout.addWidget(self.log_edit)
        right_splitter.addWidget(log_box)

        # Metrics summary
        metrics_box = _section("Results")
        metrics_layout = QVBoxLayout(metrics_box)

        self.metrics_label = QLabel("Train a model to see results.")
        self.metrics_label.setFont(QFont("Segoe UI", 10))
        self.metrics_label.setWordWrap(True)
        metrics_layout.addWidget(self.metrics_label)

        metrics_layout.addWidget(_h_line())

        # Feature importance table
        metrics_layout.addWidget(_bold_label("Feature Importance"))
        self.fi_table = QTableWidget(0, 2)
        self.fi_table.setHorizontalHeaderLabels(["Element", "Importance"])
        self.fi_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.fi_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.fi_table.setFixedHeight(200)
        metrics_layout.addWidget(self.fi_table)

        metrics_layout.addWidget(_bold_label("Confusion Matrix (Multi-class)"))
        self.cm_table = QTableWidget(0, 0)
        self.cm_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.cm_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        metrics_layout.addWidget(self.cm_table)

        metrics_layout.addWidget(_bold_label("Per-class Report"))
        self.report_table = QTableWidget(0, 4)
        self.report_table.setHorizontalHeaderLabels(["Class", "Precision", "Recall", "F1-Score"])
        self.report_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.report_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.report_table.setFixedHeight(200)
        metrics_layout.addWidget(self.report_table)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(metrics_box)
        right_splitter.addWidget(scroll)

        main_layout.addWidget(right_splitter, 1)

    # ── Slots ──────────────────────────────────────────────────────────────
    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "",
            "Data Files (*.csv *.xlsx *.xls);;All Files (*)"
        )
        if not path:
            return
        self.train_file_edit.setText(path)
        self._load_columns(path)

    def _load_columns(self, path):
        try:
            ext = os.path.splitext(path)[1].lower()
            df = pd.read_excel(path, nrows=1) if ext in ('.xlsx', '.xls') else pd.read_csv(path, nrows=1)
            self._df = df
            non_element_cols = [c for c in df.columns if c not in PERIODIC_TABLE_ELEMENTS]
            self.label_col_combo.clear()
            self.label_col_combo.addItems(non_element_cols)
            self.label_col_combo.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read file:\n{e}")

    def _browse_save_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if path:
            self.save_dir_edit.setText(path)

    def _start_training(self):
        file_path = self.train_file_edit.text().strip()
        if not file_path:
            QMessageBox.warning(self, "Missing Input", "Please select a data file.")
            return
        label_col = self.label_col_combo.currentText()
        if not label_col:
            QMessageBox.warning(self, "Missing Input", "Please select the sample-type column.")
            return
        save_dir = self.save_dir_edit.text().strip()
        if not save_dir:
            QMessageBox.warning(self, "Missing Input", "Please select a directory to save the model.")
            return
        model_name = self.model_name_edit.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Missing Input", "Please enter a model name.")
            return

        params = {
            'n_estimators': self.n_est_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'max_depth': self.depth_spin.value(),
            'random_state': self.seed_spin.value(),
        }
        save_format = 'pickle' if self.rb_pickle.isChecked() else 'onnx'

        self.log_edit.clear()
        self.train_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.metrics_label.setText("Training…")

        self._thread = QThread()
        self._worker = TrainWorker(file_path, label_col, params, save_format, model_name, save_dir)
        print("before thread")
        self._worker.moveToThread(self._thread)
        print("after thread")
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_train_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_done)
        self._thread.start()

    def _append_log(self, msg):
        self.log_edit.append(msg)

    def _on_train_finished(self, results):
        acc_m = results['acc_multi']
        acc_b = results['acc_binary']
        cv_m = results['cv_mean']
        cv_s = results['cv_std']
        cv_scores = results['cv_scores']

        self.metrics_label.setText(
            f"<b>Multi-class Accuracy:</b> {acc_m:.4f} ({acc_m*100:.2f}%)    "
            f"<b>Binary Accuracy (Soil/Non-soil):</b> {acc_b:.4f} ({acc_b*100:.2f}%)<br>"
            f"<b>5-Fold CV:</b> {cv_m:.4f} ± {cv_s*2:.4f}    "
            f"Scores: {[round(s,4) for s in cv_scores]}"
        )

        # Feature importance
        fi: pd.DataFrame = results['feature_importance']
        self.fi_table.setRowCount(len(fi))
        for i, row in fi.iterrows():
            self.fi_table.setItem(i, 0, QTableWidgetItem(str(row['Element'])))
            item = QTableWidgetItem(f"{row['Importance']:.6f}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.fi_table.setItem(i, 1, item)

        # Confusion matrix
        cm = results['confusion_matrix']
        labels = results['confusion_labels']
        n = len(labels)
        self.cm_table.setRowCount(n)
        self.cm_table.setColumnCount(n)
        self.cm_table.setHorizontalHeaderLabels(labels)
        self.cm_table.setVerticalHeaderLabels(labels)
        max_val = max(cm[i][j] for i in range(n) for j in range(n)) or 1
        for i in range(n):
            for j in range(n):
                val = cm[i][j]
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if i == j and val > 0:
                    intensity = int(180 + 75 * val / max_val)
                    item.setBackground(QColor(0, min(intensity, 255), 0, 120))
                elif val > 0:
                    item.setBackground(QColor(255, 80, 80, 100))
                self.cm_table.setItem(i, j, item)
        self.cm_table.resizeColumnsToContents()
        self.cm_table.resizeRowsToContents()

        total_height = (
                self.cm_table.horizontalHeader().height() +
                sum(self.cm_table.rowHeight(i) for i in range(n)) +
                10
        )

        self.cm_table.setFixedHeight(total_height)
        # Per-class report
        report = results['report']
        class_keys = [k for k in report if k not in ('accuracy', 'macro avg', 'weighted avg')]
        self.report_table.setRowCount(len(class_keys) + 2)
        for idx, k in enumerate(class_keys):
            self.report_table.setItem(idx, 0, QTableWidgetItem(k))
            self.report_table.setItem(idx, 1, QTableWidgetItem(f"{report[k]['precision']:.4f}"))
            self.report_table.setItem(idx, 2, QTableWidgetItem(f"{report[k]['recall']:.4f}"))
            self.report_table.setItem(idx, 3, QTableWidgetItem(f"{report[k]['f1-score']:.4f}"))
        for idx2, avg_key in enumerate(['macro avg', 'weighted avg']):
            r_idx = len(class_keys) + idx2
            self.report_table.setItem(r_idx, 0, QTableWidgetItem(avg_key))
            self.report_table.setItem(r_idx, 1, QTableWidgetItem(f"{report[avg_key]['precision']:.4f}"))
            self.report_table.setItem(r_idx, 2, QTableWidgetItem(f"{report[avg_key]['recall']:.4f}"))
            self.report_table.setItem(r_idx, 3, QTableWidgetItem(f"{report[avg_key]['f1-score']:.4f}"))

    def _on_error(self, msg):
        self.log_edit.append(f"ERROR:\n{msg}")
        self.metrics_label.setText("Training failed. See log.")
        QMessageBox.critical(self, "Training Error", msg[:500])

    def _on_thread_done(self):
        self.train_btn.setEnabled(True)
        self.progress.setVisible(False)


# ── Test Tab ───────────────────────────────────────────────────────────────────
class TestTab(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = None
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # ── Left controls ──────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(320)
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Model file
        model_box = _section("1. Load Model")
        model_layout = QVBoxLayout(model_box)
        model_row = QHBoxLayout()
        self.model_file_edit = QLineEdit()
        self.model_file_edit.setPlaceholderText("Select .pkl or .onnx model…")
        self.model_file_edit.setReadOnly(True)
        btn_model = QPushButton("Browse")
        btn_model.setFixedWidth(70)
        btn_model.clicked.connect(self._browse_model)
        model_row.addWidget(self.model_file_edit)
        model_row.addWidget(btn_model)
        model_layout.addLayout(model_row)
        left_layout.addWidget(model_box)

        # Data file
        data_box = _section("2. Data File")
        data_layout = QVBoxLayout(data_box)
        data_row = QHBoxLayout()
        self.test_file_edit = QLineEdit()
        self.test_file_edit.setPlaceholderText("Select CSV or Excel file…")
        self.test_file_edit.setReadOnly(True)
        btn_data = QPushButton("Browse")
        btn_data.setFixedWidth(70)
        btn_data.clicked.connect(self._browse_data)
        data_row.addWidget(self.test_file_edit)
        data_row.addWidget(btn_data)
        data_layout.addLayout(data_row)

        data_layout.addWidget(QLabel("True label column (optional):"))
        self.test_label_combo = QComboBox()
        self.test_label_combo.setPlaceholderText("None (no accuracy metrics)")
        self.test_label_combo.setEnabled(False)
        data_layout.addWidget(self.test_label_combo)
        left_layout.addWidget(data_box)

        # Run button
        self.run_btn = QPushButton("▶  Run Model")
        self.run_btn.setFixedHeight(38)
        self.run_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.run_btn.setStyleSheet(
            "QPushButton { background:#059669; color:white; border-radius:6px; }"
            "QPushButton:hover { background:#047857; }"
            "QPushButton:disabled { background:#6ee7b7; }"
        )
        self.run_btn.clicked.connect(self._start_testing)
        left_layout.addWidget(self.run_btn)

        self.test_progress = QProgressBar()
        self.test_progress.setRange(0, 0)
        self.test_progress.setVisible(False)
        left_layout.addWidget(self.test_progress)

        left_layout.addStretch()
        main_layout.addWidget(left)

        # ── Right results ──────────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # Log
        log_box = _section("Log")
        log_layout = QVBoxLayout(log_box)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9))
        self.log_edit.setFixedHeight(100)
        log_layout.addWidget(self.log_edit)
        right_layout.addWidget(log_box)

        # Metrics (shown only when true labels given)
        self.metrics_box = _section("Accuracy Metrics")
        metrics_layout = QVBoxLayout(self.metrics_box)
        self.test_metrics_label = QLabel("")
        self.test_metrics_label.setFont(QFont("Segoe UI", 10))
        self.test_metrics_label.setWordWrap(True)
        metrics_layout.addWidget(self.test_metrics_label)

        self.test_cm_table = QTableWidget(0, 0)
        self.test_cm_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.test_cm_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        metrics_layout.addWidget(_bold_label("Confusion Matrix"))
        metrics_layout.addWidget(self.test_cm_table)

        self.metrics_box.setVisible(False)
        right_layout.addWidget(self.metrics_box)

        # Predictions table
        pred_box = _section("Predictions")
        pred_layout = QVBoxLayout(pred_box)
        self.pred_table = QTableWidget(0, 0)
        self.pred_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.pred_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        pred_layout.addWidget(self.pred_table)
        right_layout.addWidget(pred_box, 1)

        main_layout.addWidget(right, 1)

    # ── Slots ──────────────────────────────────────────────────────────────
    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Model File", "",
            "Model Files (*.pkl *.onnx);;All Files (*)"
        )
        if path:
            self.model_file_edit.setText(path)

    def _browse_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "",
            "Data Files (*.csv *.xlsx *.xls);;All Files (*)"
        )
        if not path:
            return
        self.test_file_edit.setText(path)
        try:
            ext = os.path.splitext(path)[1].lower()
            df = pd.read_excel(path, nrows=1) if ext in ('.xlsx', '.xls') else pd.read_csv(path, nrows=1)
            non_element_cols = [c for c in df.columns if c not in PERIODIC_TABLE_ELEMENTS]
            self.test_label_combo.clear()
            self.test_label_combo.addItem("")  # blank = no labels
            self.test_label_combo.addItems(non_element_cols)
            self.test_label_combo.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read file:\n{e}")

    def _start_testing(self):
        model_path = self.model_file_edit.text().strip()
        if not model_path:
            QMessageBox.warning(self, "Missing Input", "Please select a model file.")
            return
        file_path = self.test_file_edit.text().strip()
        if not file_path:
            QMessageBox.warning(self, "Missing Input", "Please select a data file.")
            return

        label_col = self.test_label_combo.currentText().strip() or None

        self.log_edit.clear()
        self.run_btn.setEnabled(False)
        self.test_progress.setVisible(True)
        self.metrics_box.setVisible(False)
        self.pred_table.setRowCount(0)
        self.pred_table.setColumnCount(0)

        self._thread = QThread()
        self._worker = TestWorker(model_path, file_path, label_col)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_test_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_done)
        self._thread.start()

    def _append_log(self, msg):
        self.log_edit.append(msg)

    def _on_test_finished(self, results):
        df: pd.DataFrame = results['results_df']
        metrics = results.get('metrics')

        # Populate predictions table (show all cols, highlight prediction cols)
        self.pred_table.setRowCount(len(df))
        self.pred_table.setColumnCount(len(df.columns))
        self.pred_table.setHorizontalHeaderLabels(list(df.columns))
        highlight_cols = {'Predicted_Material', 'Binary_Prediction', 'Confidence', 'Soil_Probability'}
        for col_idx, col_name in enumerate(df.columns):
            for row_idx in range(len(df)):
                val = df.iloc[row_idx, col_idx]
                item = QTableWidgetItem(str(val))
                if col_name in highlight_cols:
                    item.setBackground(QColor(219, 234, 254))  # light blue
                self.pred_table.setItem(row_idx, col_idx, item)
        self._append_log(f"Done. {len(df)} predictions generated.")

        # Metrics
        if metrics:
            acc_m = metrics['acc_multi']
            acc_b = metrics['acc_binary']
            self.test_metrics_label.setText(
                f"<b>Multi-class Accuracy:</b> {acc_m:.4f} ({acc_m*100:.2f}%)    "
                f"<b>Binary Accuracy:</b> {acc_b:.4f} ({acc_b*100:.2f}%)"
            )
            cm = metrics['confusion_matrix']
            labels = metrics['confusion_labels']
            n = len(labels)
            self.test_cm_table.setRowCount(n)
            self.test_cm_table.setColumnCount(n)
            self.test_cm_table.setHorizontalHeaderLabels(labels)
            self.test_cm_table.setVerticalHeaderLabels(labels)
            max_val = max(cm[i][j] for i in range(n) for j in range(n)) or 1
            for i in range(n):
                for j in range(n):
                    val = cm[i][j]
                    item = QTableWidgetItem(str(val))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    if i == j and val > 0:
                        item.setBackground(QColor(0, min(180 + 75 * val // max_val, 255), 0, 120))
                    elif val > 0:
                        item.setBackground(QColor(255, 80, 80, 100))
                    self.test_cm_table.setItem(i, j, item)
            self.test_cm_table.resizeColumnsToContents()
            self.test_cm_table.resizeRowsToContents()

            total_height = (
                    self.test_cm_table.horizontalHeader().height() +
                    sum(self.test_cm_table.rowHeight(i) for i in range(n)) +
                    10
            )
            self.test_cm_table.setFixedHeight(total_height)

            self.metrics_box.setVisible(True)

    def _on_error(self, msg):
        self.log_edit.append(f"ERROR:\n{msg}")
        QMessageBox.critical(self, "Error", msg[:500])

    def _on_thread_done(self):
        self.run_btn.setEnabled(True)
        self.test_progress.setVisible(False)


# ── Main Page ──────────────────────────────────────────────────────────────────
class GradientBoostedDecisionTreePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QWidget()
        header.setStyleSheet("background:#1e3a5f;")
        header.setFixedHeight(56)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(16, 0, 16, 0)
        title = QLabel("Gradient Boosted Decision Tree  –  pXRF Material Classifier")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color:white;")
        h_layout.addWidget(title)
        layout.addWidget(header)

        tabs = QTabWidget()
        tabs.setFont(QFont("Segoe UI", 10))
        tabs.addTab(TrainTab(), "  Train Model  ")
        tabs.addTab(TestTab(),  "  Test Model  ")
        layout.addWidget(tabs)