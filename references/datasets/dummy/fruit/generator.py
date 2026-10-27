#!/usr/bin/env python3
"""Reference generator for the KNN Explorer "fruit sorter" dummy dataset.

Design reference only, not app code. NumPy only, fixed seed, so every run
writes byte-identical CSVs.

    python3 generator.py              -> data.csv         (apple / orange / pear)
    python3 generator.py --binary     -> data-binary.csv  (apple / not_apple)
    python3 generator.py --out x.csv  -> write somewhere else

The binary file has the same rows in the same order; only the label column is
collapsed. A short summary (class counts, which rows carry label noise or an
outlier) goes to stderr.

Feature roles (see ../README.md for the full column dictionary):
    weight_g        F1  headline x-axis, large scale (~120-250 g)
    diameter_cm     F2  headline y-axis, small scale (~6-9 cm); separates the
                        classes but is drowned out by weight_g unless scaled
    sweetness_brix  F3  informative, moderate scale; 4% missing (MCAR)
    days_in_store   F4  pure noise, integer 0-14, same for every class
    stem_length_mm  F5  right-skewed (log-normal) plus a few extreme outliers
                        (fruit picked with a twig still attached)
    skin_colour     C1  categorical, 4 levels, partially informative; 4% missing
    fruit           target, last column
"""

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------- parameters
SEED = 7    # first seed in 1-40 that passed every target with margin (see ../README.md)

CLASSES = ["apple", "orange", "pear"]
CLASS_COUNTS = [90, 70, 40]           # 45% / 35% / 20%, n = 200
BINARY_POSITIVE = "apple"             # --binary: "apple" vs "not_apple"

# F1 weight_g ~ Normal(mean, sd) per class, clipped, rounded to 1 g
WEIGHT = {"apple": (172, 22), "orange": (196, 26), "pear": (178, 24)}
WEIGHT_CLIP = (90, 320)

# F2 diameter_cm ~ Normal(mean, sd) per class, correlated with weight inside
# each class (bigger fruit is wider), clipped, rounded to 0.1 cm
DIAMETER = {"apple": (7.4, 0.40), "orange": (8.2, 0.42), "pear": (6.6, 0.38)}
DIAMETER_WEIGHT_CORR = 0.4
DIAMETER_CLIP = (5.0, 9.9)

# F3 sweetness_brix ~ Normal(mean, sd) per class, rounded to 0.1 deg Bx
BRIX = {"apple": (13.0, 1.5), "orange": (10.8, 1.5), "pear": (12.2, 1.5)}
BRIX_CLIP = (6.0, 18.0)

# F4 days_in_store ~ UniformInteger[0, 14] for every class (no signal)
DAYS_RANGE = (0, 14)

# F5 stem_length_mm ~ LogNormal(log(median), sigma) per class, rounded to 1 mm,
# then N_STEM_OUTLIERS random rows get a twig: Uniform(STEM_OUTLIER_RANGE)
STEM_MEDIAN = {"apple": 18.0, "orange": 9.0, "pear": 27.0}
STEM_SIGMA = 0.45
STEM_CLIP = (2, 400)
N_STEM_OUTLIERS = 5
STEM_OUTLIER_RANGE = (150, 280)

# C1 skin_colour: per-class probabilities over the levels
COLOURS = ["red", "green", "yellow", "orange"]
COLOUR_P = {
    "apple":  [0.55, 0.28, 0.14, 0.03],
    "orange": [0.02, 0.08, 0.27, 0.63],
    "pear":   [0.10, 0.52, 0.36, 0.02],
}

# Label noise: for each (A, B) pair, one random row of class A and one of
# class B swap labels. Class counts stay exact; 2 x len(LABEL_SWAPS) rows end up
# mislabelled (10 rows = 5%). 4 of the 5 swaps involve apple, so the --binary
# file keeps 8 noisy rows too.
LABEL_SWAPS = [("apple", "orange"), ("apple", "pear"), ("orange", "pear"),
               ("apple", "orange"), ("apple", "pear")]

# Missing completely at random: exact number of blank cells per column
N_MISSING = {"sweetness_brix": 8, "skin_colour": 8}   # 4% each

COLUMNS = ["weight_g", "diameter_cm", "sweetness_brix", "days_in_store",
           "stem_length_mm", "skin_colour", "fruit"]
DECIMALS = {"weight_g": 0, "diameter_cm": 1, "sweetness_brix": 1,
            "days_in_store": 0, "stem_length_mm": 0}


# ---------------------------------------------------------------- generator
def generate(binary=False):
    """Return (columns, rows, info). rows are lists; None marks a missing cell."""
    rng = np.random.default_rng(SEED)
    blocks = {c: [] for c in COLUMNS}

    for cls, n in zip(CLASSES, CLASS_COUNTS):
        w_mu, w_sd = WEIGHT[cls]
        z_w = rng.standard_normal(n)
        weight = w_mu + w_sd * z_w

        d_mu, d_sd = DIAMETER[cls]
        r = DIAMETER_WEIGHT_CORR
        diameter = d_mu + d_sd * (r * z_w + np.sqrt(1 - r * r) * rng.standard_normal(n))

        b_mu, b_sd = BRIX[cls]
        brix = rng.normal(b_mu, b_sd, n)

        days = rng.integers(DAYS_RANGE[0], DAYS_RANGE[1] + 1, n)

        stem = rng.lognormal(np.log(STEM_MEDIAN[cls]), STEM_SIGMA, n)

        colour = rng.choice(len(COLOURS), size=n, p=COLOUR_P[cls])

        blocks["weight_g"].append(np.clip(weight, *WEIGHT_CLIP))
        blocks["diameter_cm"].append(np.clip(diameter, *DIAMETER_CLIP))
        blocks["sweetness_brix"].append(np.clip(brix, *BRIX_CLIP))
        blocks["days_in_store"].append(days.astype(float))
        blocks["stem_length_mm"].append(np.clip(stem, *STEM_CLIP))
        blocks["skin_colour"].append(colour)
        blocks["fruit"].append(np.full(n, CLASSES.index(cls)))

    data = {c: np.concatenate(v) for c, v in blocks.items()}
    n_rows = len(data["fruit"])

    # outliers in F5
    out_idx = rng.choice(n_rows, N_STEM_OUTLIERS, replace=False)
    data["stem_length_mm"][out_idx] = rng.uniform(*STEM_OUTLIER_RANGE, N_STEM_OUTLIERS)

    # label noise: swap labels between one row of class A and one of class B
    y = data["fruit"]
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
        label = CLASSES[data["fruit"][i]]
        if binary:
            label = label if label == BINARY_POSITIVE else "not_" + BINARY_POSITIVE
        row = []
        for c in COLUMNS[:-1]:
            if i in missing.get(c, ()):
                row.append(None)
            elif c == "skin_colour":
                row.append(COLOURS[data[c][i]])
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
    print(f"stem_length_mm outlier rows (1-based): {info['outlier_rows']}", file=sys.stderr)


if __name__ == "__main__":
    main()
