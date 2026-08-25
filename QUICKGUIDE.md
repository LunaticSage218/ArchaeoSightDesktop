# ArchaeoSight Desktop — Quick Start

A step-by-step guide to installing and running ArchaeoSight on a Windows PC,
followed by a worked example using the sample data that comes with the program.

Follow the steps in order. Type the commands exactly as written, then press
**Enter**. Nothing in this guide sends your data anywhere — everything runs on
your own computer.

---

## Step 1 — Install Miniconda (once, ~5 minutes)

ArchaeoSight needs a program called **conda** to install its many parts.

1. Go to <https://www.anaconda.com/download/success>
2. Under **Miniconda Installers**, download the **Windows 64-Bit** installer.
   (Miniconda is the small version. Full Anaconda also works if you already
   have it.)
3. Run the installer and click through with the default settings.
4. When it finishes, click the Windows **Start** button, type
   `Anaconda Prompt`, and open it.

You will see a black window with a line ending in `>`. This is where every
command in this guide gets typed.

> **Important:** use **Anaconda Prompt** for everything below. Do not use
> PowerShell or the regular Command Prompt — conda will not be found there.

---

## Step 2 — Get the program files

Pick **one** of the two options below. They both end with the code in a folder
on your computer. Option B is easier if you have never used Git.

### Option A — With Git

Use this if Git is already installed (type `git --version`; if you get a
version number, you have it).

```bat
cd some-directory-you-can-find
git clone https://github.com/LunaticSage218/ArchaeoSightDesktop.git
cd ArchaeoSightDesktop
```

The folder is now at `C:\Users\<your name>\some-directory-you-can-find\ArchaeoSightDesktop`.

To get later updates, come back to this folder and run:

```bat
git pull
```

### Option B — With the ZIP file (no Git needed)

1. Go to <https://github.com/LunaticSage218/ArchaeoSightDesktop>
2. Click the green **Code** button, then **Download ZIP**.
3. Open your **Downloads** folder, right-click `ArchaeoSightDesktop-main.zip`,
   and choose **Extract All…**
4. In the box that appears, set the destination to
   `C:\Users\<your name>\some-directory-you-can-find` and click **Extract**.
5. Back in Anaconda Prompt, type:

```bat
cd %USERPROFILE%\some-directory-you-can-find\ArchaeoSightDesktop-main
```

> **Note:** the ZIP creates a folder named `ArchaeoSightDesktop-main` (with
> `-main` on the end). If you rename it to `ArchaeoSightDesktop`, use that name
> in the command instead.

To check you are in the right place, type `dir`. You should see `main.py` and
`environment.yml` in the list.

---

## Step 3 — Build the environment (once, ~10 minutes)

Still in Anaconda Prompt, inside the project folder:

```bat
conda env create -f environment.yml
```

This downloads and installs everything the program needs. It is a large
download and will take several minutes, printing many lines. Leave it alone
until you get the `>` prompt back.

When it finishes you should see a message that ends with something like
`To activate this environment, use: conda activate ArchaeoSightDesktop`.

---

## Step 4 — Run the program

```bat
conda activate ArchaeoSightDesktop
python main.py
```

The ArchaeoSight window opens on the **Gradient Boosted Decision Tree** tab.
That's it — the installation is done.

Leave the black Anaconda Prompt window open while you use the program. Closing
it closes ArchaeoSight.

---

## Starting it again on another day

The installation only happens once. From then on, open **Anaconda Prompt** and
type these three lines:

```bat
conda activate ArchaeoSightDesktop
cd %USERPROFILE%\some-directory-you-can-find\ArchaeoSightDesktop
python main.py
```

(Use `ArchaeoSightDesktop-main` on the second line if you installed from the
ZIP.)

**Tip:** you can save yourself the typing. Open Notepad, paste the three lines
above, and save the file to your Desktop as `ArchaeoSight.bat` — set
"Save as type" to **All Files** so it does not become a `.txt`. Double-clicking
that file from then on starts the program.

---

## Worked example — the Gradient Boosted Decision Tree

The Gradient Boosted Decision Tree (**GBDT**) is the supervised classifier. You
show it pXRF readings you have already identified, it learns which element
patterns go with which material, and then it labels new readings for you.

The program ships with a real dataset to practise on:

```
examples\AllPXRF_FINAL_14Oct.csv
```

It holds 4,530 pXRF readings from a site. Each row is one shot, with 18 element
readings (Fe, Cu, Ca, Sr, and so on), the coordinates it was taken at
(`X_Coord`, `Y_Coord`), the bag and context it came from (`BAG`, `CNTXT`), and
the material it was identified as (`material`). The `material` column contains
3,729 pottery, 339 slag, 332 soil, 79 metal, and 51 blank rows — blanks are
treated as soil.

### Part 1 — Train a model

In the ArchaeoSight window, stay on the **Gradient Boosted Decision Tree** tab
and make sure the inner tab **Train Model** is selected.

**1. Data File** — click **Browse**, navigate into the project folder, open the
`examples` folder, and choose `AllPXRF_FINAL_14Oct_synthetic.csv`. This contains a dataset with real and synthetic data.

The two dropdowns below wake up once the file loads.

- **Sample type column:** choose `material`.
  This is the column holding the answers you are teaching it.
- **Group column for splits (optional):** choose `BAG`.
  This matters. Several pXRF shots are often taken of the *same* object, and
  those readings are near-identical. Without grouping, some end up in the
  practice set and some in the exam set, and the model scores well by
  recognising an object it has already seen. Setting `BAG` keeps every reading
  from one bag on the same side of the split. The resulting score is lower —
  and it is the honest one. (`CNTXT` groups by context instead, which is
  stricter still.)

**2. Hyperparameters** — leave all four at their defaults for your first run, you can change them later on if you please: 

| Setting | Default | What it does |
| --- | --- | --- |
| `n_estimators` | 100 | How many small decision trees are built and stacked. More is slower, and past a point stops helping. |
| `learning_rate` | 0.1 | How much each new tree is allowed to correct the ones before it. Lower is more cautious. |
| `max_depth` | 4 | How many questions deep each tree may go. Deeper trees memorise rather than generalise. |
| `random_state` | 42 | Fixes the shuffling, so re-running gives identical numbers. Only change this if you want to see how much luck was involved. |

In a real scenario, you want to train the tree slowly and deeply, but for now the defaults are for illustrating what it does. 

**3. Save Model**

- Leave the format on **Pickle (.pkl)**. (ONNX is only needed for exporting to
  the Android app.)
- **Model name:** type `example_gbdt`
- Click **Browse** and choose the `models` folder inside the project. If there isn't a models folder, go ahead and make one. 

Now click **▶ Train Model**.

A progress bar runs along the bottom and the **Training Log** on the right
fills in. On a normal laptop this takes roughly half a minute.

**What you should see when it finishes:**

- **Results** — a line reading *Multi-class Accuracy*, *Binary Accuracy
  (Soil/Non-soil)*, and a *5-Fold CV (grouped)* figure with a ± range. Multi-class
  accuracy is how often it named the exact material; binary accuracy is how
  often it merely got soil-vs-not-soil right, and is always the higher number.
  Because you set a group column, the CV line says **grouped** — expect a lower
  figure here than you would get without it.
- **Feature Importance** — the element columns ranked by how much the model
  leaned on each one. This is often the most archaeologically interesting panel:
  it tells you which elements are actually separating your materials.
- **Confusion Matrix** — rows are the true material, columns are what the model
  guessed. The green diagonal is the correct calls; anything red and off the
  diagonal is a mistake, and shows you *which* material it was confused with.
- **Per-class Report** — precision, recall and F1 for each material. Watch the
  `metal` row: with only 79 examples in the data, its scores will be shaky, and
  that is a shortage of samples, not a broken program.

Your trained model is now saved at `models\example_gbdt.pkl`.

### Part 2 — Run the model on data

Click the inner tab **Test Model**.

**1. Load Model** — **Browse** to `models\example_gbdt.pkl`.

**2. Data File** — **Browse** to `examples\AllPXRF_FINAL_14Oct.csv` again.

> In real work this second file would be your *new, unidentified* readings.
> We are reusing the training file here only so the example is self-contained.
> The scores it prints will look flattering for exactly that reason — the model
> has seen these rows before. Judge a model by the training tab's grouped CV
> number, not by this.

**3. True label column (optional)** — choose `material`.

Leave this blank when your new readings have no known answers; you will still
get predictions, just no accuracy scores.

Click **▶ Run Model**.

**What you should see:**

- **Accuracy Metrics** with a confusion matrix (this panel only appears because
  you supplied a true label column).
- **Predictions** — a table of every reading with the material the model
  predicted, and a `NonSoil_Probability` column: the model's confidence that the
  point is *something other than plain soil*, from 0 to 1. A 0.95 means "almost
  certainly cultural material," a 0.10 means "almost certainly just dirt."

### Part 3 — Turn it into dig recommendations

Click **→ Send Results to Next Dig** (it only becomes clickable after a run).

The program jumps to the **Next Dig** tab with your classified points already
loaded. Now:

1. Set the target column to `NonSoil_Probability`.
2. Set the transform to **Logit** — `NonSoil_Probability` is a probability, and
   kriging left alone will happily predict values below 0 or above 1, which are
   meaningless. Logit keeps the predictions inside 0–1.
3. Tick **Restrict to surveyed area (convex hull)** so it doesn't recommend
   spots beyond where you actually sampled.
4. Set the **explore vs. exploit** slider. All the way left sends you to the
   strongest signal; all the way right sends you to the least-known ground;
   the middle balances the two, and is the usual starting point.
5. Run it.

You get a ranked list of recommended dig locations, written out as
`next_dig_NonSoil_Probability.csv`, plus `priority_*.png` and `overlay_*.png`
map images.

---

## If something goes wrong

**`'conda' is not recognized`**
You are in the wrong window. Close it, and open **Anaconda Prompt** from the
Start menu instead.

**`python main.py` opens the Microsoft Store**
Windows is using its own placeholder Python. Run
`conda activate ArchaeoSightDesktop` first, then try again.

**`ModuleNotFoundError: No module named 'PyQt6'` (or tensorflow, hdbscan…)**
The environment is not switched on. Run `conda activate ArchaeoSightDesktop`
and try again. To confirm the environment exists at all, run `conda env list` —
you should see `ArchaeoSightDesktop` in the list.

**`can't open file 'main.py'`**
You are in the wrong folder. `cd` back into the project folder and check with
`dir` that `main.py` is listed.

**hdbscan or pykrige fails to build during install**
You installed with `pip` instead of conda. Use
`conda env create -f environment.yml`. (`requirements.txt` in the project is a
version reference only — do not install from it.)

**`conda env create` fails to solve**
Usually a network or channel problem. Check your internet connection and run
the same command again.

**A tab shows an error when you click Run**
Every tab has a **Log** panel on the right with the full error text. Read it
there, or copy it into a bug report — it says exactly what failed.

**Starting over**
To wipe the environment and rebuild it from nothing:

```bat
conda env remove -n ArchaeoSightDesktop
conda env create -f environment.yml
```
