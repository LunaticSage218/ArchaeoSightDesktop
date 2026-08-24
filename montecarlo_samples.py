#!/usr/bin/env python
"""
Monte Carlo generation of synthetic samples, conditioned on a class label.

Given a CSV/Excel file whose rows are labelled by a class column (e.g.
material = pottery / soil / slag / metal), this fabricates new "fake" samples
*per class* that stay statistically consistent with the real samples of that
class.

How it keeps data consistent
-----------------------------
For each class the numeric columns are modelled JOINTLY with a Gaussian copula:
  * each feature's marginal distribution is preserved EXACTLY by sampling from
    its empirical quantiles (so medians/percentiles and skew match, and values
    never fall outside the observed range -> no negative concentrations),
  * the correlations BETWEEN elements are reproduced via the copula fitted on
    normal-scores (modelling columns independently would destroy these).
The fit is NaN-robust (pairwise), so scattered missing element readings don't
shrink the per-class sample. Non-numeric / carry-through columns are filled by
bootstrap-resampling from the same class, keeping the schema and categorical
mix class-consistent.

Usage
-----
    python montecarlo_samples.py data.csv
    python montecarlo_samples.py data.csv --label-col material --multiplier 2
    python montecarlo_samples.py data.csv --n-per-class 500 --exclude X_Coord,Y_Coord,ID
    python montecarlo_samples.py data.csv --include-original --output combined.csv

Defaults: generate the same number of samples per class as the original
(scaled by --multiplier), preserve all columns, write <input>_synthetic.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, on_bad_lines="skip", low_memory=False)


def _pick_label_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise SystemExit(f"ERROR: label column '{requested}' not found. "
                             f"Columns: {list(df.columns)}")
        return requested
    if "material" in df.columns:
        return "material"
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if len(non_numeric) == 1:
        return non_numeric[0]
    raise SystemExit(
        "ERROR: could not auto-detect a label column. Pass --label-col. "
        f"Non-numeric candidates: {non_numeric}"
    )


def _psd_correlation(R: np.ndarray) -> np.ndarray:
    """Project a (possibly non-PSD, pairwise) correlation matrix to the nearest PSD
    correlation matrix (clip eigenvalues, renormalise the diagonal to 1)."""
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    R = (R + R.T) / 2.0
    w, V = np.linalg.eigh(R)
    R = (V * np.clip(w, 1e-8, None)) @ V.T
    d = np.sqrt(np.clip(np.diag(R), 1e-12, None))
    return R / np.outer(d, d)


def _fit_sample_class(
    sub: pd.DataFrame, feats: list[str], m: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Draw m synthetic feature rows for one class via a Gaussian copula.

    Each feature's marginal is preserved EXACTLY by mapping back through its
    empirical quantiles, while inter-feature correlations are reproduced through
    the Gaussian copula fitted on normal-scores. NaN-robust (pairwise), needs no
    distributional assumption, and cannot produce values outside the observed
    range (so no negative concentrations).
    """
    from scipy.stats import norm

    Xdf = sub[feats].apply(pd.to_numeric, errors="coerce")
    result = pd.DataFrame(np.nan, index=range(m), columns=feats, dtype=float)

    usable = [f for f in feats if Xdf[f].notna().sum() >= 2]
    if not usable:
        return result

    # Normal-scores transform (rank -> uniform -> standard normal), per feature.
    Z = np.full((len(Xdf), len(usable)), np.nan)
    observed = {}
    for j, f in enumerate(usable):
        col = Xdf[f]
        observed[f] = np.sort(col.dropna().to_numpy())
        n = len(observed[f])
        u = (col.rank(method="average") - 0.5) / n            # NaN ranks stay NaN
        Z[:, j] = norm.ppf(np.clip(u.to_numpy(), 1e-6, 1 - 1e-6))

    R = _psd_correlation(pd.DataFrame(Z, columns=usable).corr().to_numpy())

    g = rng.multivariate_normal(np.zeros(len(usable)), R, size=m)
    u_sim = norm.cdf(g)
    for j, f in enumerate(usable):
        # Inverse empirical CDF: sampled values stay within the observed range.
        result[f] = np.quantile(observed[f], np.clip(u_sim[:, j], 0.0, 1.0))
    return result


def _corr_drift(orig: pd.DataFrame, syn: pd.DataFrame) -> float:
    """Mean absolute difference between the (pairwise) correlation matrices.

    0 = identical correlation structure; ~1 = maximally different. Pairwise so
    scattered NaNs don't force listwise deletion.
    """
    co = orig.corr().to_numpy()
    cs = syn.corr().to_numpy()
    diff = np.abs(np.nan_to_num(co) - np.nan_to_num(cs))
    iu = np.triu_indices_from(diff, k=1)
    return float(diff[iu].mean()) if iu[0].size else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="Monte Carlo per-class synthetic sample generator.")
    ap.add_argument("input", help="Input CSV or Excel file.")
    ap.add_argument("--label-col", default=None,
                    help="Class column (default: 'material', else the lone non-numeric column).")
    ap.add_argument("--n-per-class", type=int, default=None,
                    help="Fixed number of synthetic samples per class. "
                         "Default: match each class's original count.")
    ap.add_argument("--multiplier", type=float, default=1.0,
                    help="Scale the matched per-class counts (ignored if --n-per-class set).")
    ap.add_argument("--features", default=None,
                    help="Comma-separated numeric columns to model. Default: all numeric.")
    ap.add_argument("--exclude", default=None,
                    help="Comma-separated numeric columns to NOT model "
                         "(they are bootstrapped instead), e.g. X_Coord,Y_Coord,ID.")
    ap.add_argument("--include-original", action="store_true",
                    help="Prepend the original rows (with synthetic=False) to the output.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--output", default=None, help="Output CSV path.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"ERROR: input not found: {in_path}")

    df = _read(in_path)
    label_col = _pick_label_col(df, args.label_col)
    rng = np.random.default_rng(args.seed)

    # Feature columns: numeric, excluding the label and any --exclude list.
    excluded = {c.strip() for c in args.exclude.split(",")} if args.exclude else set()
    if args.features:
        feats = [c.strip() for c in args.features.split(",")]
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise SystemExit(f"ERROR: --features not in file: {missing}")
    else:
        feats = [c for c in df.columns
                 if c != label_col and c not in excluded
                 and pd.api.types.is_numeric_dtype(df[c])]
    if not feats:
        raise SystemExit("ERROR: no numeric feature columns selected.")

    carry = [c for c in df.columns if c != label_col and c not in feats]
    int_feats = [c for c in feats if pd.api.types.is_integer_dtype(df[c])]

    # Drop rows with no class label.
    n_nan_label = int(df[label_col].isna().sum())
    work = df[df[label_col].notna()].copy()
    classes = work[label_col].value_counts()

    print(f"Input: {in_path}  ({len(df)} rows)")
    print(f"Label column: '{label_col}'  |  classes: {dict(classes)}"
          + (f"  |  dropped {n_nan_label} unlabelled rows" if n_nan_label else ""))
    print(f"Modelled features ({len(feats)}): {feats}")
    if carry:
        print(f"Carry-through (bootstrapped): {carry}")
    print()

    parts = []
    for cls, n_orig in classes.items():
        sub = work[work[label_col] == cls]
        m = args.n_per_class if args.n_per_class is not None else int(round(n_orig * args.multiplier))
        if m <= 0:
            continue

        gen = _fit_sample_class(sub, feats, m, rng)

        # Bootstrap carry-through columns from this class's observed values.
        for c in carry:
            vals = sub[c].dropna().to_numpy()
            gen[c] = rng.choice(vals, size=m) if len(vals) else np.nan

        # Round integer-typed feature columns back to whole numbers.
        for c in int_feats:
            gen[c] = np.rint(gen[c])

        gen[label_col] = cls
        parts.append(gen)

        drift = _corr_drift(
            sub[feats].apply(pd.to_numeric, errors="coerce"),
            gen[feats],
        )
        print(f"  {str(cls):10s}  orig={n_orig:5d}  ->  generated={m:5d}"
              + (f"   corr-diff={drift:.3f}" if drift == drift else ""))

    synthetic = pd.concat(parts, ignore_index=True)
    synthetic["synthetic"] = True

    if args.include_original:
        orig = df.copy()
        orig["synthetic"] = False
        result = pd.concat([orig, synthetic], ignore_index=True)
    else:
        result = synthetic

    # Preserve original column order, with 'synthetic' last.
    cols = [c for c in df.columns if c in result.columns] + ["synthetic"]
    result = result[cols]

    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_synthetic.csv")
    result.to_csv(out_path, index=False)
    print(f"\nWrote {len(synthetic)} synthetic rows "
          f"({len(result)} total) to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
