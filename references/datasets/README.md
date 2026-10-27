# Datasets

Six small classification CSVs for the KNN explorer. Each has a header row, the target as the **last column**, UTF-8, `\n` line endings, and missing values as empty cells. All files are under 210 KB.

Stats below come from `pandas` 3.0.5 and `scikit-learn` 1.9.1, run on the files in this folder (2026-10-27). They are not copied from the source pages. "CV" means 5-fold stratified cross-validation, shuffled with seed 0. When a row says "standardised", the `StandardScaler` was fitted inside each fold.

| File | Rows × cols | Features | Target (classes) | Missing | Licence |
|---|---|---|---|---|---|
| `toy-2d-binary.csv` | 62 × 3 | 2 numeric | `label` (2) | none | generated here, no restrictions |
| `iris.csv` | 150 × 5 | 4 numeric | `species` (3) | none | CC BY 4.0 |
| `palmer-penguins.csv` | 344 × 8 | 5 numeric, 2 categorical | `species` (3) | 19 cells in 11 rows | CC0 |
| `wine.csv` | 178 × 14 | 13 numeric | `cultivar` (3) | none | CC BY 4.0 |
| `breast-cancer-wisconsin.csv` | 569 × 31 | 30 numeric | `diagnosis` (2) | none | CC BY 4.0 |
| `german-credit.csv` | 1000 × 21 | 7 numeric, 13 categorical | `credit_risk` (2) | none | CC BY 4.0 |

---

## toy-2d-binary.csv

- **Source:** generated for this project. No licence restrictions.
- **Shape:** 62 × 3. Columns `x1`, `x2` (float, 2 dp) and `label` (`A`/`B`).
- **Balance:** A 31 (50%), B 31 (50%). **Missing:** none.
- **Ranges:** x1 −0.27 to 5.63, x2 0.61 to 6.02. Both axes share a scale, so the 2-D plot shows the true distances.
- **Generator** (NumPy 2.5.3; re-run it to reproduce the file exactly):

```python
import numpy as np, pandas as pd
rng = np.random.default_rng(seed=7)
n = 30
a = rng.normal(loc=[2.0, 2.0], scale=0.9, size=(n, 2))   # class A blob
b = rng.normal(loc=[4.5, 4.0], scale=0.9, size=(n, 2))   # class B blob
X = np.vstack([a, b]); y = np.array(["A"] * n + ["B"] * n)
# two deliberate label-noise points inside the other class's blob
X = np.vstack([X, [[4.6, 4.2], [1.9, 1.8]]]); y = np.concatenate([y, ["A", "B"]])
pd.DataFrame({"x1": X[:, 0].round(2), "x2": X[:, 1].round(2), "label": y}).to_csv("toy-2d-binary.csv", index=False)
```

- **CV accuracy (unscaled):** k=1 0.937, k=5 0.953, k=15 0.953.
- **Teaches:** the core loop with nothing else in the way. Use it for the distance → sort → vote animation, k=1 islands around the two noisy points (rows 61–62), the ε-ball/unit-ball shapes for L1, L2, L∞ and Minkowski p, Voronoi cells at k=1, and tie-breaking at even k.

## iris.csv

- **Source:** UCI ML Repository, dataset 53: https://archive.ics.uci.edu/dataset/53/iris. Download: `https://archive.ics.uci.edu/static/public/53/iris.zip`.
- **Licence:** CC BY 4.0, as stated on the UCI page.
- **Citation:** Fisher, R. (1936). *Iris* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76
- **Processing:** taken from `bezdekIris.data`, UCI's corrected copy. `iris.names` notes that samples 35 and 38 in `iris.data` disagree with Fisher's paper. The `Iris-` prefix was removed from class names.
- **Shape:** 150 × 5: `sepal_length_cm`, `sepal_width_cm`, `petal_length_cm`, `petal_width_cm`, `species`.
- **Balance:** setosa, versicolor and virginica, 50 each (33.3%). **Missing:** none.
- **Ranges (cm):** sepal length 4.3–7.9, sepal width 2.0–4.4, petal length 1.0–6.9, petal width 0.1–2.5. The largest span is only 2.5× the smallest.
- **CV accuracy, k=5 unscaled:** all 4 features 0.953, petal pair 0.960, sepal pair 0.753.
- **Teaches:** choosing features for the 2-D view. The petal pair separates the classes almost completely, while the sepal pair leaves versicolor and virginica overlapping. Setosa splits off at any k. Also a good first look at a 4-D → 2-D projection with PCA or a scatter-plot matrix.

## palmer-penguins.csv

- **Source:** `palmerpenguins` R package, https://github.com/allisonhorst/palmerpenguins. Raw file: `https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv`.
- **Licence:** CC0, per the repo README, under the Palmer Station LTER Data Policy.
- **Citation:** Horst AM, Hill AP, Gorman KB (2020). *palmerpenguins: Palmer Archipelago (Antarctica) penguin data*. R package version 0.1.0. doi:10.5281/zenodo.3960218. The data come from Gorman KB, Williams TD, Fraser WR (2014), *PLoS ONE* 9(3):e90081.
- **Processing:** `NA` strings became empty cells, integer columns are written without `.0`, and `species` was moved to the last column. Nothing else changed.
- **Shape:** 344 × 8. Numeric: `bill_length_mm` (32.1–59.6), `bill_depth_mm` (13.1–21.5), `flipper_length_mm` (172–231), `body_mass_g` (2700–6300), `year` (2007–2009). Categorical: `island` (3 levels), `sex` (2 levels).
- **Balance:** Adelie 152 (44.2%), Gentoo 124 (36.0%), Chinstrap 68 (19.8%).
- **Missing:** 19 cells in 11 rows. Two rows lack all four measurements and sex. Nine more lack only `sex`. After dropping incomplete rows, 333 remain.
- **CV accuracy, k=5 (333 complete rows, `year` dropped):** 4 numeric unscaled 0.766, 4 numeric standardised 0.988, numeric plus one-hot `island`/`sex` standardised 0.988.
- **Teaches:** the full preprocessing pipeline: dropping or imputing missing values, one-hot encoding `island` and `sex` (10 columns after encoding), dropping a meaningless column (`year`), and scaling. `body_mass_g` spans 3600 units against a bill depth span of 8.4, so unscaled distances are mostly body mass.

## wine.csv

- **Source:** UCI ML Repository, dataset 109: https://archive.ics.uci.edu/dataset/109/wine. Download: `https://archive.ics.uci.edu/static/public/109/wine.zip`.
- **Licence:** CC BY 4.0, as stated on the UCI page.
- **Citation:** Aeberhard, S. & Forina, M. (1992). *Wine* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J
- **Processing:** column names come from `wine.names`. The class column is renamed `cultivar` with values `cultivar_1/2/3` so it can't be read as a numeric feature, and it is moved to the last column.
- **Shape:** 178 × 14, with 13 continuous chemical measurements.
- **Balance:** cultivar_2 71 (39.9%), cultivar_1 59 (33.1%), cultivar_3 48 (27.0%). **Missing:** none.
- **Scale spread:** `proline` runs 278–1680 (span 1402) and `magnesium` 70–162, while `nonflavanoid_phenols` runs 0.13–0.66 (span 0.53) and `hue` 0.48–1.71. The largest span is 2650× the smallest. `wine.names` itself says to "standardise variables for … classifiers which are NOT scale invariant".
- **CV accuracy, k=5:** unscaled 0.663, unscaled without `proline` 0.843, standardised 0.961. Standardised with Minkowski p=1, 2 and 4 gives 0.961, 0.961 and 0.955.
- **Teaches:** feature scaling. Show the distance equation term by term: unscaled, the `proline` term dominates the sum. Compare z-score and min-max scaling side by side.

## breast-cancer-wisconsin.csv

- **Source:** UCI ML Repository, dataset 17: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic. Download: `https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip`.
- **Licence:** CC BY 4.0, as stated on the UCI page.
- **Citation:** Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
- **Processing:** the patient `id` column was dropped (an identifier, not a feature). The 30 features are named `<feature>_<mean|se|worst>`, following `wdbc.names`: radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension. `M/B` became `malignant/benign`, and the target is in the last column.
- **Shape:** 569 × 31, all 30 features continuous.
- **Balance:** benign 357 (62.7%), malignant 212 (37.3%). **Missing:** none.
- **Scale spread:** `area_worst` spans 4069 units and `fractal_dimension_se` spans 0.029, a ratio of about 141,000×.
- **CV accuracy, k=5:** 30 features unscaled 0.932, standardised 0.965, 10 `*_worst` features 0.960, `radius_mean` and `texture_mean` only 0.882. Adding 100 random N(0,1) noise columns to the 30 standardised features drops it to 0.917.
- **Distance concentration:** median relative contrast (d_max − d_min)/d_min per point, standardised: 2 features 82.9, 30 features 9.6, 30 + 100 noise 1.1. As dimensions are added, a point's nearest and farthest neighbours end up at similar distances.
- **Teaches:** the curse of dimensionality and feature selection. Plot relative contrast against the number of features, add noise features to watch accuracy fall, and compare a few features with all of them. It also suits a binary confusion matrix, since a missed malignant case is the costly error.

## german-credit.csv

- **Source:** UCI ML Repository, dataset 144: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data. Download: `https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip`. This file is built from `german.data`.
- **Licence:** CC BY 4.0, as stated on the UCI page.
- **Citation:** Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77
- **Processing:** symbolic codes such as `A11` and `A43` were decoded to short labels, following `german.doc`. `A47` ("vacation – does not exist?") is labelled `vacation` but never occurs in the data, and neither does `A95` (female: single). Target `1/2` became `good/bad`. Values contain no commas, so no field needs quoting.
- **Shape:** 1000 × 21.
  - Numeric: `duration_months` (4–72), `credit_amount_dm` (250–18424), `installment_rate_pct` (1–4), `residence_since` (1–4), `age_years` (19–75), `existing_credits` (1–4), `num_dependents` (1–2).
  - Categorical, with level counts: `checking_status` 4, `credit_history` 5, `purpose` 10, `savings` 5, `employment_since` 5, `personal_status_sex` 4, `other_debtors` 3, `property` 4, `other_installment_plans` 3, `housing` 3, `job` 4, `telephone` 2, `foreign_worker` 2.
  - One-hot encoding gives 61 columns: 54 dummies plus the 7 numeric.
- **Balance:** good 700 (70%), bad 300 (30%). **Missing:** none.
- **Cost matrix (from `german.doc`):** predicting good when the truth is bad costs 5. Predicting bad when the truth is good costs 1.
- **CV accuracy, k=5:** majority-class baseline 0.700. Numeric only: unscaled 0.649, standardised 0.666. One-hot plus standardised: 0.739 at k=5, 0.724 at k=15.
- **Teaches:** one-hot encoding at scale (13 categorical columns become 54 dummies) and why accuracy misleads on imbalanced data: numeric-only KNN scores below "always predict good". It also covers cost-sensitive evaluation and thresholds on the vote share. Several of these ordinal codes, like `savings` and `employment_since`, could be integer-encoded instead, which makes a good comparison. Note that `personal_status_sex`, `age_years` and `foreign_worker` are protected or sensitive attributes, so the app can use this dataset to raise fairness questions.
