# ArchaeoSight Desktop

A desktop app for archaeologists working with **pXRF** (portable X-ray fluorescence) data. You feed it a spreadsheet of element readings taken at mapped points across a site, and it helps you answer two questions:

1. **What is this?** — classify each reading as soil vs. pottery / slag / metal / bone, either with a supervised model you train on labelled samples, or with unsupervised clustering when you have no labels.
2. **Where should I dig next?** — interpolate the results across the whole site and rank the unexcavated spots most worth sampling, balancing "looks promising" against "we know nothing about that corner yet."

It's a single PyQt6 window with four tabs. Everything runs locally on your machine; nothing is uploaded anywhere.

---

## The four tabs

### 1. Gradient Boosted Decision Tree — supervised classifier

Train a scikit-learn gradient-boosting model to predict material type from element concentrations.

- **Train Model** — pick your CSV/Excel file, choose the label column (usually `material`), tune `n_estimators` / `learning_rate` / `max_depth`, and train. Optionally set a **group column** (auto-detects `BAG` / `CNTXT`) so repeat shots of the same object never land on both sides of a train/test split — grouped scores are the honest ones and will look lower than plain cross-validation. Saves as a pickle (`.pkl`) or ONNX (`.onnx` + a sibling `_meta.pkl`).
- **Test Model** — load a saved model, run it on new readings, see accuracy metrics and a prediction table. The button **→ Send Results to Next Dig** hands the classified points straight to the Next Dig tab in memory, carrying a `NonSoil_Probability` column (1 − probability of soil) that becomes the interpolation target.

### 2. Clustering with HDBSCAN + Autoencoders — unsupervised

For when you don't have labels. A TensorFlow autoencoder compresses the element readings into a small latent space, PCA reduces it further, and HDBSCAN finds natural groupings — plus points it flags as noise.

- **Train Pipeline** — set latent dim, epochs, batch size, HDBSCAN `min cluster size` / `min samples`, and the noise threshold σ. Writes everything to an output folder keyed on `model_bundle.pkl`, alongside `training_loss.png`, `latent_pca_hdbscan_train.png`, `cluster_assignments_*.csv`, and the raw latent vectors.
- **Apply to New Data** — point it at a previous run's output folder and a new file; new points are assigned to the nearest cluster centroid, or marked noise if they're beyond the stored distance threshold.
- **📱 Export for Mobile** — writes `encoder.onnx`, `scaler.onnx`, `pca.onnx`, optional classifiers, and `metadata.json` (feature columns, centroids, thresholds) for the Android companion app.

### 3. Kriging — spatial interpolation

Ordinary or universal kriging (via pykrige) of one numeric column across the site. Choose the variogram model (linear, power, gaussian, spherical, exponential, hole-effect), lag count, and grid resolution. Produces an interpolated surface, a variance/uncertainty surface, the fitted variogram, and optional cross-validation — saved as `interpolation_<col>.png`, `variance_<col>.png`, `variogram_<col>.png`, and a full grid `kriging_<col>.csv`.

This tab is the standalone "just show me where the copper is" view.

### 4. Next Dig — adaptive sampling recommendations

The payoff tab. It krigs your target (an element, or `NonSoil_Probability` from the classifier) and blends the predicted **value** with the predicted **uncertainty** into a single priority surface:

```
priority = (1 - w) * value + w * sigma
```

The **explore vs. exploit** slider is `w`. All the way left, it sends you where the signal is strongest; all the way right, where the map is least certain; the middle is the usual UCB-style compromise.

It then masks out an avoid-radius around points you've already sampled, optionally clips to the convex hull of the surveyed area (so it doesn't send you off the edge of the map into extrapolation nonsense), and greedily picks the top *N* sites subject to a minimum spacing.

**Site image overlay** — you can drop in an overhead photo of the site (PNG/JPG/TIF) and the recommendations get drawn on top of it. If a world file sits beside the image (`.jgw`, `.pgw`, `.tfw`, `.wld`), the extent fields auto-fill and the georeferencing is exact. Without one, the image is centered on your data with pixel aspect preserved — approximate placement, never stretched.

Outputs: `next_dig_<col>.csv` (the ranked picks), `priority_<col>.png`, and `overlay_<col>.png`.

---

## What your data needs to look like

CSV or Excel. One row per pXRF reading. Columns:

| Column | Required | Notes |
| --- | --- | --- |
| Element columns | yes | Named with bare periodic-table symbols: `Fe`, `Cu`, `Sr`, `Ca`, `Zn`… These are auto-detected. Values in ppm or whatever unit you like — negatives (non-detects) get clipped to 0. |
| `X_Coord`, `Y_Coord` | for Kriging / Next Dig | Projected coordinates in **meters** (UTM), not lat/lon. |
| `material` | for supervised training | The class label. Blank or `unknown` is coerced to `soil`. |
| `BAG` / `CNTXT` | optional | Grouping keys, used to keep repeat shots of one object out of both train and test. |
| anything else | optional | Carried along, ignored by the models. |

A real example lives in `examples/AllPXRF_FINAL_14Oct.csv` (~4,500 rows) — use it to check the app works before pointing it at your own data.

---

## Installation from scratch (Windows)

**Use Anaconda, not a plain venv.** hdbscan and pykrige ship compiled extensions, and `pip install` on Windows will happily try to build them from source and fail unless you have a full MSVC toolchain. Conda ships prebuilt binaries. This is the path of least suffering.

Once conda is installed, the whole setup is a single command — skip to step 3 if you already have Miniconda and the code.

### 1. Install Miniconda

Download and run the Windows 64-bit installer from <https://www.anaconda.com/download/success> (Miniconda is the small one and is all you need; full Anaconda works too). Accept the defaults.

When it finishes, open **Anaconda Prompt** from the Start menu. Do *not* use plain `cmd` or PowerShell for the next steps unless you've run `conda init` — and don't use the `python` that's already on your PATH, that's the Microsoft Store stub and it doesn't work.

### 2. Get the code

If you have Git:

```bat
git clone <your-repo-url> ArchaeoSightDesktop
cd ArchaeoSightDesktop
```

Otherwise download the ZIP, extract it, and `cd` into the folder.

### 3. Create the environment — one command

```bat
conda env create -f environment.yml
```

That's the entire install. `environment.yml` pins every dependency and handles the conda/pip split for you: hdbscan and pykrige come from conda-forge as prebuilt binaries, TensorFlow and the ONNX stack install from PyPI into the same environment. Give it ten minutes or so — TensorFlow is a large download.

> **On TensorFlow:** on Windows, TF 2.11+ is CPU-only — GPU support requires WSL2. This app's autoencoder is small enough that CPU is fine; expect the Clustering tab to take a couple of minutes on a few thousand rows.

### 4. Run it

```bat
conda activate ArchaeoSightDesktop
python main.py
```

The window should open on the Gradient Boosted Decision Tree tab. If it does, you're done.

### Running it later

Every new terminal session, you need to reactivate the env first:

```bat
conda activate ArchaeoSightDesktop
cd path\to\ArchaeoSightDesktop
python main.py
```

Or skip the activation entirely by calling the env's interpreter directly:

```bat
C:\Users\<you>\anaconda3\envs\ArchaeoSightDesktop\python.exe main.py
```

### Updating or rebuilding the environment

After `environment.yml` changes:

```bat
conda env update -f environment.yml --prune
```

To start over from scratch:

```bat
conda env remove -n ArchaeoSightDesktop
conda env create -f environment.yml
```

### A note on `requirements.txt`

`environment.yml` is the supported install. `requirements.txt` is kept alongside it as a reference pin list — it's pip-valid, but `pip install -r requirements.txt` on Windows will fail trying to compile hdbscan and pykrige from source unless you have the MSVC build tools. Use it to check or reproduce an exact version, not to install.

---

## Other scripts

### Generating synthetic training data

`montecarlo_samples.py` fabricates extra samples per class when a material type is under-represented. It uses a Gaussian copula fitted per class, so each element's marginal distribution is preserved exactly (values can never fall outside the observed range — no negative concentrations) while the correlations *between* elements are reproduced. Modelling the columns independently would destroy those correlations, which is the whole reason this exists.

```bat
python montecarlo_samples.py examples\AllPXRF_FINAL_14Oct.csv --label-col material
```

Useful flags:

```bat
--multiplier 2                       generate 2x each class's original count
--n-per-class 500                    fixed count per class instead
--exclude X_Coord,Y_Coord,ID         bootstrap these instead of modelling them
--include-original --output combined.csv   prepend the real rows, tagged synthetic=False
```

It prints a `corr-diff` per class — the mean absolute difference between the real and synthetic correlation matrices. Near 0 means the synthetic data kept the element relationships intact.

### Building a standalone .exe THIS DOESN'T WORK RN

```bat
python build_windows.py                # one-folder build (recommended)
python build_windows.py --onefile      # single .exe — large and slow to start with TF
python build_windows.py --clean        # wipe build/, dist/ and the .spec first
python build_windows.py --icon app.ico
```

Output lands in `dist\ArchaeoSight\ArchaeoSight.exe`. Run this with the same interpreter that has the dependencies, so the bundle matches what you've been testing against. The script force-collects the packages whose stock PyInstaller hooks are incomplete (TensorFlow, hdbscan, pykrige, onnx, sklearn, matplotlib), and `pyi_rth_native_dlls.py` is a runtime hook that puts TF/onnxruntime on the Windows DLL search path inside the frozen app.

Stick with one-folder builds. `--onefile` with TensorFlow produces something enormous that unpacks to a temp dir on every launch, and on some machines it fails outright.

---

## Troubleshooting

**`'conda' is not recognized`** — you're not in the Anaconda Prompt. Open it from the Start menu, or run `conda init cmd.exe` once and restart your terminal.

**`python main.py` opens the Microsoft Store** — you're using the PATH stub, not the env. Run `conda activate ArchaeoSightDesktop` first, or call the env's `python.exe` by its full path.

**`ModuleNotFoundError: No module named 'PyQt6'`** (or tensorflow, hdbscan…) — the env isn't active, or you installed into a different one. Check with `conda env list` and `where python`.

**hdbscan or pykrige fails to build during install** — you're pip-installing them. Use `conda env create -f environment.yml`, which pulls both from conda-forge as prebuilt binaries.

**`DLL load failed while importing QtCore/QtWidgets: The specified procedure could not be found`** — don't "fix" this by upgrading PyQt6; it's why PyQt6 is pinned to 6.8.1. Note the message says *procedure*, not *module*: a DLL was found and loaded, it just doesn't export a symbol the caller wants. PyQt6 6.10.2's `Qt6Core.dll` imports version-stamped symbols from `icuuc.dll`, and on this machine the only ICU the loader finds is the wrong build — the failure happens below Python entirely and reproduces with a bare `ctypes.WinDLL` load of `Qt6Core.dll`. 6.8.1 has no such dependency.

If you hit this again after changing versions, `diagnose_qt.py` in the project root will name the exact DLL and missing exports:

```bat
pip install pefile
python diagnose_qt.py
```

Things that look like the cause but aren't, all ruled out on this machine: mismatched `PyQt6` / `PyQt6-Qt6` / `PyQt6-sip` pins, an outdated Visual C++ redistributable (System32 was already current), the stale `msvcp140.dll` bundled inside `PyQt6\Qt6\bin`, and a conda-installed Qt shadowing the pip one.

**`conda env create` fails to solve** — usually a channel issue. Confirm conda-forge is reachable (`conda config --show channels`) and try again; if a specific pin is unavailable for your platform, loosen it in `environment.yml` (e.g. `pandas=3.0.*` → `pandas`) and re-run.

**The app opens but a tab throws on Run** — every tab has a log panel showing the full traceback. Start there; the computation runs on a background thread and the error is reported verbatim.

**Kriging predictions fall outside [0, 1]** — kriging is unbounded, so bounded targets need a transform. Pick **Logit** for probability columns like `NonSoil_Probability`, and **Log** for right-skewed element concentrations.

**Next Dig recommends spots off the edge of the site** — tick "Restrict to surveyed area (convex hull)". Interpolation beyond your sample coverage is extrapolation, and the uncertainty term will always find it attractive.

---

## Project layout

```
main.py                    single window, wires the four tabs together
styles.py                  dark theme, colour tokens, shared widget helpers
pages/
  GradientBoostedDecisionTreePage.py    supervised classifier (Train / Test)
  ClusteringPage.py                     autoencoder + HDBSCAN (Train / Apply)
  KrigingPage.py                        spatial interpolation
  NextDigPage.py                        adaptive sampling recommendations
montecarlo_samples.py      synthetic sample generator (CLI)
environment.yml            the full environment — the one-command install
requirements.txt           reference pin list (not the install path)
build_windows.py           PyInstaller driver
pyi_rth_native_dlls.py     runtime hook for native DLLs in frozen builds
examples/                  reference datasets
models/                    saved model artifacts
```

Each page follows the same shape: a fixed-width control panel on the left, results on the right (log, tables, plots), and all the actual computation on a background `QThread` so the UI never freezes. Heavy imports (TensorFlow, sklearn, pykrige, matplotlib) are deliberately deferred until a worker actually runs, which is why the app starts in about a second despite the stack behind it.

## Citation

If you use this project in your research, please cite it as:

> Griffith, L. (2026). *ArchaeoSightDesktop*. GitHub.  
> https://github.com/LunaticSage218/ArchaeoSightDesktop

**DOI:** https://doi.org/10.5281/zenodo.22091615
