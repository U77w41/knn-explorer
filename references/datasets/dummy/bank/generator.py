#!/usr/bin/env python3
"""Reference generator for the KNN Explorer "bank accounts" dummy dataset.

SYNTHETIC. Every value comes from the random distributions below. Nothing is
derived from, fitted to, or calibrated against real bank or customer data, and
the class rules are invented for teaching, not taken from any detection system.

Design reference only, not app code. NumPy only, fixed seed, so every run
writes byte-identical CSVs.

    python3 generator.py              -> data.csv         (normal / suspicious / mule)
    python3 generator.py --binary     -> data-binary.csv  (mule / not_mule)
    python3 generator.py --out x.csv  -> write somewhere else

The binary file has the same rows in the same order; only the label column is
collapsed. A short summary (class counts, which rows carry label noise or an
outlier) goes to stderr.

Feature roles (see ../README.md for the full column dictionary):
    monthly_inflow_inr  F1  headline x-axis, large scale (thousands to lakhs)
    avg_hold_days       F2  headline y-axis, small scale (0.1-9.9 days): how
                            long money sits before it leaves; separates the
                            classes but is drowned out by inflow unless scaled
    txn_count_30d       F3  informative, moderate scale; 4% missing (MCAR)
    marketing_score     F4  pure noise: a 0-1 score from an unrelated
                            marketing model, same for every class
    account_age_days    F5  right-skewed (log-normal) plus a few extreme
                            outliers (decades-old accounts)
    primary_channel     C1  categorical, 4 levels, partially informative;
                            4% missing
    account_class       target, last column
"""

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------- parameters
SEED = 9    # first seed in 1-40 that passed every target with margin (see ../README.md)

CLASSES = ["normal", "suspicious", "mule"]
CLASS_COUNTS = [90, 70, 40]           # 45% / 35% / 20%, n = 200
BINARY_POSITIVE = "mule"              # --binary: "mule" vs "not_mule"

# F1 monthly_inflow_inr ~ Normal(mean, sd) per class, clipped, rounded to Rs 500
INFLOW = {"normal": (40000, 14000), "suspicious": (58000, 16000), "mule": (68000, 18000)}
INFLOW_CLIP = (3000, 300000)

# F2 avg_hold_days ~ Normal(mean, sd) per class, clipped, rounded to 0.1 day
HOLD = {"normal": (6.3, 1.3), "suspicious": (4.0, 1.2), "mule": (1.1, 0.5)}
HOLD_CLIP = (0.1, 9.9)

# F3 txn_count_30d ~ Normal(mean, sd) per class, clipped, rounded to an integer
TXN = {"normal": (22, 8), "suspicious": (34, 12), "mule": (60, 16)}
TXN_CLIP = (1, 200)

# F4 marketing_score ~ Uniform(0, 1) for every class (no signal), 2 dp
SCORE_RANGE = (0.0, 1.0)

# F5 account_age_days ~ LogNormal(log(median), sigma) per class, rounded to a
# day, then N_AGE_OUTLIERS random rows become very old: Uniform(AGE_OUTLIER_RANGE)
AGE_MEDIAN = {"normal": 1500.0, "suspicious": 700.0, "mule": 220.0}
AGE_SIGMA = 0.5
AGE_CLIP = (7, 20000)
N_AGE_OUTLIERS = 5
AGE_OUTLIER_RANGE = (15000, 20000)

# C1 primary_channel: per-class probabilities over the levels
CHANNELS = ["UPI", "NEFT", "ATM", "branch"]
CHANNEL_P = {
    "normal":     [0.40, 0.10, 0.05, 0.45],
    "suspicious": [0.30, 0.55, 0.12, 0.03],
    "mule":       [0.40, 0.05, 0.54, 0.01],
}

# Label noise: for each (A, B) pair, one random row of class A and one of
# class B swap labels. Class counts stay exact; 2 x len(LABEL_SWAPS) rows end up
# mislabelled (12 rows = 6%). 3 of the 6 swaps involve mule, so the --binary
# file keeps 6 noisy rows (3%).
LABEL_SWAPS = [("normal", "suspicious"), ("normal", "mule"), ("suspicious", "mule"),
               ("normal", "suspicious"), ("normal", "suspicious"), ("suspicious", "mule")]

# Missing completely at random: exact number of blank cells per column
N_MISSING = {"txn_count_30d": 8, "primary_channel": 8}   # 4% each

COLUMNS = ["monthly_inflow_inr", "avg_hold_days", "txn_count_30d",
           "marketing_score", "account_age_days", "primary_channel", "account_class"]
DECIMALS = {"monthly_inflow_inr": 0, "avg_hold_days": 1, "txn_count_30d": 0,
            "marketing_score": 2, "account_age_days": 0}


# ---------------------------------------------------------------- generator
def generate(binary=False):
    """Return (columns, rows, info). rows are lists; None marks a missing cell."""
    rng = np.random.default_rng(SEED)
    blocks = {c: [] for c in COLUMNS}

    for cls, n in zip(CLASSES, CLASS_COUNTS):
        inflow = rng.normal(*INFLOW[cls], n)
        hold = rng.normal(*HOLD[cls], n)
        txn = rng.normal(*TXN[cls], n)
        score = rng.uniform(*SCORE_RANGE, n)
        age = rng.lognormal(np.log(AGE_MEDIAN[cls]), AGE_SIGMA, n)
        channel = rng.choice(len(CHANNELS), size=n, p=CHANNEL_P[cls])

        blocks["monthly_inflow_inr"].append(np.round(np.clip(inflow, *INFLOW_CLIP) / 500) * 500)
        blocks["avg_hold_days"].append(np.clip(hold, *HOLD_CLIP))
        blocks["txn_count_30d"].append(np.clip(txn, *TXN_CLIP))
        blocks["marketing_score"].append(score)
        blocks["account_age_days"].append(np.clip(age, *AGE_CLIP))
        blocks["primary_channel"].append(channel)
        blocks["account_class"].append(np.full(n, CLASSES.index(cls)))

    data = {c: np.concatenate(v) for c, v in blocks.items()}
    n_rows = len(data["account_class"])

    # outliers in F5
    out_idx = rng.choice(n_rows, N_AGE_OUTLIERS, replace=False)
    data["account_age_days"][out_idx] = rng.uniform(*AGE_OUTLIER_RANGE, N_AGE_OUTLIERS)

    # label noise: swap labels between one row of class A and one of class B
    y = data["account_class"]
    noise_idx = []
    for a, b in LABEL_SWAPS:
        free = np.setdiff1d(np.arange(n_rows), noise_idx)
        i = int(rng.choice(free[y[free] == CLASSES.index(a)]))
        j = int(rng.choice(free[y[free] == CLASSES.index(b)]))
        y[i], y[j] = y[j], y[i]
        noise_idx += [i, j]

    # shuffle rows so classes are interleaved
    perm = rng.permutation(n_rows)
    data = {c: v[perm] for c, v in data.items()}
    inv = np.argsort(perm)
    out_rows = sorted(int(inv[i]) + 1 for i in out_idx)       # 1-based data rows
    noise_rows = sorted(int(inv[i]) + 1 for i in noise_idx)

    # round
    for c, d in DECIMALS.items():
        data[c] = np.round(data[c], d)

    # missing completely at random
    missing = {}
    for c, k in N_MISSING.items():
        missing[c] = set(rng.choice(n_rows, k, replace=False).tolist())

    rows = []
    for i in range(n_rows):
        label = CLASSES[data["account_class"][i]]
        if binary:
            label = label if label == BINARY_POSITIVE else "not_" + BINARY_POSITIVE
        row = []
        for c in COLUMNS[:-1]:
            if i in missing.get(c, ()):
                row.append(None)
            elif c == "primary_channel":
                row.append(CHANNELS[data[c][i]])
            else:
                row.append(float(data[c][i]))
        row.append(label)
        rows.append(row)

    info = {"label_noise_rows": noise_rows, "outlier_rows": out_rows}
    return COLUMNS, rows, info


def fmt(col, v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    d = DECIMALS[col]
    return str(int(v)) if d == 0 else f"{v:.{d}f}"


def write_csv(path, columns, rows):
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(fmt(c, v) if c in DECIMALS else (v or "")
                             for c, v in zip(columns, row)) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--binary", action="store_true",
                    help=f"collapse the target to {BINARY_POSITIVE} / not_{BINARY_POSITIVE}")
    ap.add_argument("--out", help="output CSV path (default: next to this script)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    path = args.out or os.path.join(here, "data-binary.csv" if args.binary else "data.csv")
    columns, rows, info = generate(binary=args.binary)
    write_csv(path, columns, rows)

    labels = [r[-1] for r in rows]
    counts = {c: labels.count(c) for c in sorted(set(labels))}
    print(f"wrote {path}: {len(rows)} rows, {counts}", file=sys.stderr)
    print(f"label-noise rows (1-based): {info['label_noise_rows']}", file=sys.stderr)
    print(f"account_age_days outlier rows (1-based): {info['outlier_rows']}", file=sys.stderr)


if __name__ == "__main__":
    main()
