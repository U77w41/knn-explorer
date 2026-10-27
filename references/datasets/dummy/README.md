# Dummy datasets (built-in default)

Two candidate datasets for the app to use before anyone uploads a CSV. Both are synthetic, 200 rows, with five numeric features, one categorical feature and the target in the last column. Each comes with a binary variant that keeps the same rows and changes only the label. `generator.py` in each folder rebuilds both CSVs byte for byte. These generators are for design reference; the app doesn't run them.

**Recommendation: fruit as the default.** Reasons are at the end.

## Shared design

| Role | What it's for | Fruit column | Bank column |
|---|---|---|---|
| F1 | headline x-axis, large scale | `weight_g` | `monthly_inflow_inr` |
| F2 | headline y-axis, small scale; separates classes once scaled | `diameter_cm` | `avg_hold_days` |
| F3 | informative, moderate scale, 4% missing | `sweetness_brix` | `txn_count_30d` |
| F4 | pure noise | `days_in_store` | `marketing_score` |
| F5 | skewed, 5 extreme outliers | `stem_length_mm` | `account_age_days` |
| C1 | categorical, 4 levels, partly informative, 4% missing | `skin_colour` | `primary_channel` |
| target | 3 classes, 90 / 70 / 40 (45 / 35 / 20%) | `fruit` | `account_class` |

- **Why 200 rows.** One row is 0.5%, so percentages stay round. A stratified 80/20 split gives 160 training and 40 test rows (18 / 14 / 8 per class). Eight minority-class test rows is about the least a confusion matrix can work with; at 150 rows it would be six. One query's distance pass touches 199 other rows, short enough to animate one point at a time, and the 2-D scatters in this folder still show individual markers. 5-fold CV folds are 40 rows each.
- **Missing values.** Exactly 8 cells in F3 and 8 in C1, picked uniformly at random (MCAR), in 16 different rows. Cells are empty in the CSV.
- **Label noise.** Each generator swaps labels between pairs of rows from different classes, so class counts stay exact. Fruit swaps 5 pairs (10 rows, 5%), bank 6 pairs (12 rows, 6%). This noise, plus real overlap between classes, is what makes k = 1 overfit.
- **Rounding.** Values are rounded the way a person would record them (weight to 1 g, diameter to 0.1 cm, inflow to ₹500, hold time to 0.1 day), so some distances tie exactly. Counts are below.
- **Binary variant.** `--binary` maps the labels to `apple` / `not_apple` (45 / 55%) or `mule` / `not_mule` (20 / 80%).
- **Regenerate.** `python3 generator.py` writes `data.csv`; `python3 generator.py --binary` writes `data-binary.csv`. Both also print the label-noise and outlier row numbers to stderr. NumPy only.
- **Seed choice.** Each `SEED` is the first seed from 1 to 40 that passed every target with margin (fruit 7, bank 9). The pass rates over all 40 seeds are under "Seed robustness".

### How the numbers were produced

scikit-learn 1.9.1, NumPy 2.5.3. Accuracy is the mean over repeated stratified 5-fold CV (5 repeats, `random_state=0`, 25 fits). Median imputation and the scaler sit inside a `Pipeline`, so they're fitted on each training fold only. Unless a row says otherwise: k = 7, uniform weights, Euclidean distance, all five numeric features. One-hot columns are 0/1 and not scaled. A "flip" is an out-of-fold prediction that changes between two setups, counted over all 5 × 200 predictions. The validation scripts were throwaway and aren't in the repo.

Pass rules:

| Target | Passes when |
|---|---|
| T1 | z-scored F1+F2 beats unscaled F1+F2 by at least 10 points (k = 7) |
| T2 | best k on 1–40 is between 5 and 15, k = 1 is at least 3 points below it, and k = 160 (the whole training fold) scores the majority-class rate |
| T3 | min-max puts 95% of F5 inside 0–0.25 while robust spreads the same rows over at least 2 units, and at least 5% of predictions flip |
| T4 | one-hot C1 flips at least 5% of predictions (the accuracy change is reported) |
| T5 | 50 extra N(0, 1) noise columns cost at least 10 points (mean of 5 noise draws) |
| T6 | the k = 7 neighbour sets differ for at least 10% of queries between L1 and L2, and between L2 and L∞ |
| T7 | raw F1 range is at least 20× the F2 range; on z-scored F1×F2 every pair of class centroids is at least 1 within-class SD apart and nearest-centroid error is 5–35%; and the PNGs look right on inspection |

---

## Fruit sorter (`fruit/`)

### Columns

| Column | Unit | Role | How it's generated | Range in file | Missing |
|---|---|---|---|---|---|
| `weight_g` | g, integer | F1 | Normal per class: apple 172 ± 22, orange 196 ± 26, pear 178 ± 24 | 117–245 | 0 |
| `diameter_cm` | cm, 0.1 | F2 | Normal per class: apple 7.4 ± 0.40, orange 8.2 ± 0.42, pear 6.6 ± 0.38; within-class correlation 0.4 with weight | 5.9–9.0 | 0 |
| `sweetness_brix` | °Bx, 0.1 | F3 | Normal, sd 1.5: apple 13.0, orange 10.8, pear 12.2 | 8.1–16.0 | 8 (4%) |
| `days_in_store` | days, integer | F4 | Uniform integer 0–14, same for every class | 0–14 | 0 |
| `stem_length_mm` | mm, integer | F5 | Log-normal, σ 0.45, medians apple 18, orange 9, pear 27; 5 random rows replaced by 150–280 (picked with a twig). Skewness 5.5 | 4–265 | 0 |
| `skin_colour` | category | C1 | red / green / yellow / orange with P = apple .55/.28/.14/.03, orange .02/.08/.27/.63, pear .10/.52/.36/.02 | 4 levels | 8 (4%) |
| `fruit` | label | target | apple 90 (45%), orange 70 (35%), pear 40 (20%) | | 0 |

Binary: apple 90, not_apple 110. Label-noise rows (1-based data rows): 63, 84, 110, 116, 126, 160, 172, 188, 189, 190. F5 outlier rows: 43, 77, 140, 141, 152.

### Targets

| Target | Result | Pass |
|---|---|---|
| T1 multiclass | z-score 0.751 vs none 0.517: **+23.4 pts**. F1 alone 0.478, F2 alone 0.692, so unscaled F1+F2 is close to F1 alone | ✅ |
| T1 binary | 0.748 vs 0.628: **+12.0 pts** (F1 alone 0.629) | ✅ |
| T2 multiclass | best **k = 13** (0.730); k = 1 0.629, **10.1 pts lower**; k = 40 0.669; k ≥ 135 0.450 = majority rate | ✅ |
| T2 binary | best **k = 13** (0.757); k = 1 0.667, **9.0 pts lower**; k ≥ 145 0.550 = majority rate | ✅ |
| T3 | F5 after min-max: 95% of rows in 0–0.16, outliers 0.18–1. After robust: 95% in −0.92 to 2.23, outliers up to 19.2. (z-score: −0.53 to 0.61, outliers up to 6.7.) Accuracy min-max 0.690, robust 0.729 (**+3.9 pts**); **21.5%** of predictions flip | ✅ |
| T4 | numeric only 0.709, plus one-hot `skin_colour` 0.760 (**+5.1 pts**); **17.9%** flip. Binary: 0.713 → 0.787, 19.0% flip | ✅ |
| T5 | 0.709 → +10 noise 0.632 (−7.7) → +50 noise 0.550 (**−15.9 pts**) | ✅ |
| T6 | neighbour sets differ L1 vs L2 **88.5%**, L2 vs L∞ **92.0%**, L1 vs L∞ 99.0% of queries. On z-scored F1×F2 alone: 66.5 / 60.0 / 89.5%. CV accuracy L1 0.733, L2 0.709, L∞ 0.691; 10–11% of predictions flip between metrics | ✅ |
| T7 | raw range 128 g vs 3.1 cm, **41 : 1** at equal aspect (a flat band). Z-scored centroid gaps 1.27 (apple–pear) to 2.59 SD (orange–pear); nearest-centroid error 25%. Orange sits upper right, pear at the bottom, apple in the middle overlapping both | ✅ |

Scaler comparison (k = 7, all numeric): none 0.600, min-max 0.690, max-abs 0.663, z-score 0.709, robust 0.729, quantile 0.748, log1p(F5) + z-score 0.753, Yeo-Johnson 0.759.

### Ties

Leave-one-out: each of the 200 rows is queried against the other 199. A boundary tie means the 5th and 6th nearest distances are equal, so which row counts as the 5th neighbour depends on file order. Raw distances are exact. After scaling I matched distances within 1e-9 (relative), because z-scoring adds float noise.

| Space | L1 | L2 | L∞ |
|---|---|---|---|
| raw F1×F2 | 39 | 25 | 114 |
| z-scored F1×F2 | 38 | 31 | 89 |
| z-scored, all 5 numeric | 0 | 0 | 45 |

Vote ties on z-scored F1×F2: 33 queries at k = 4 (44 in the binary file), 22 at k = 6. Ten rows share an exact (weight, diameter) pair with another row. Even-k vote ties show up on the validation curve: k = 2 scores 0.608, below k = 1 (0.629), because scikit-learn gives tied votes to the label that sorts first.

### Pictures

`scatter-raw-equal-aspect.png`, `scatter-zscored.png`, `boundary-k1.png`, `boundary-k15.png` (z-scored F1×F2, fitted on all 200 rows), `validation-curve.png`, `scaler-comparison.png`.

---

## Bank accounts (`bank/`)

**Synthetic.** Every value is drawn from the distributions in `bank/generator.py`. Nothing is derived from, fitted to, or calibrated against real bank, customer or transaction data, and the class differences were made up for teaching. They aren't detection rules.

### Columns

| Column | Unit | Role | How it's generated | Range in file | Missing |
|---|---|---|---|---|---|
| `monthly_inflow_inr` | ₹, nearest 500 | F1 | Normal per class: normal 40,000 ± 14,000, suspicious 58,000 ± 16,000, mule 68,000 ± 18,000 | 11,500–1,05,500 | 0 |
| `avg_hold_days` | days, 0.1 | F2 | Normal, clipped to 0.1–9.9: normal 6.3 ± 1.3, suspicious 4.0 ± 1.2, mule 1.1 ± 0.5 | 0.1–9.2 | 0 |
| `txn_count_30d` | count | F3 | Normal, rounded, at least 1: normal 22 ± 8, suspicious 34 ± 12, mule 60 ± 16 | 1–93 | 8 (4%) |
| `marketing_score` | 0–1, 0.01 | F4 | Uniform 0–1, same for every class (an unrelated marketing model's output) | 0.00–1.00 | 0 |
| `account_age_days` | days, integer | F5 | Log-normal, σ 0.5, medians normal 1,500, suspicious 700, mule 220; 5 random rows replaced by 15,000–20,000 (decades-old accounts). Skewness 5.3 | 63–19,425 | 0 |
| `primary_channel` | category | C1 | UPI / NEFT / ATM / branch with P = normal .40/.10/.05/.45, suspicious .30/.55/.12/.03, mule .40/.05/.54/.01 | 4 levels | 8 (4%) |
| `account_class` | label | target | normal 90 (45%), suspicious 70 (35%), mule 40 (20%) | | 0 |

Binary: mule 40, not_mule 160. Label-noise rows: 16, 23, 40, 43, 58, 72, 103, 117, 160, 168, 191, 198 (6 of them change the binary label). F5 outlier rows: 78, 82, 105, 109, 122.

### Targets

| Target | Result | Pass |
|---|---|---|
| T1 multiclass | z-score 0.815 vs none 0.567: **+24.8 pts** (F1 alone 0.538, F2 alone 0.780) | ✅ |
| T1 binary | 0.950 vs 0.811: **+13.9 pts** (majority rate 0.800, F1 alone 0.798) | ✅ |
| T2 multiclass | best **k = 8** (0.833); k = 1 0.779, **5.4 pts lower**; k = 40 0.763; k ≥ 140 0.450 = majority rate | ✅ |
| T2 binary | best **k = 6** (0.950); k = 1 0.901, **4.9 pts lower**; k ≥ 65 0.800 = majority rate | ✅ |
| T3 | F5 after min-max: 95% of rows in 0–0.17, outliers 0.17–1. After robust: 95% in −0.81 to 2.27, outliers up to 17.6. (z-score: −0.53 to 0.67, outliers up to 6.6.) Accuracy min-max 0.808, robust 0.849 (**+4.1 pts**); **12.3%** flip | ✅ |
| T4 | 0.827 → 0.864 with one-hot `primary_channel` (**+3.7 pts**); **10.8%** flip. Binary only flips 3.8% (+0.6 pts) | ✅ |
| T5 | 0.827 → +10 noise 0.678 (−14.9) → +50 noise 0.592 (**−23.5 pts**) | ✅ |
| T6 | neighbour sets differ L1 vs L2 **85.5%**, L2 vs L∞ **88.0%**, L1 vs L∞ 100% of queries. On z-scored F1×F2: 62.0 / 55.0 / 80.5%. CV accuracy L1 0.844, L2 0.827, L∞ 0.779 | ✅ |
| T7 | raw range ₹94,000 vs 9.1 days, **10,330 : 1**: at equal aspect every account sits on one line. Z-scored centroid gaps 1.73 (mule–suspicious) to 3.43 SD (mule–normal); nearest-centroid error 21% | ✅ |

Scaler comparison (k = 7, all numeric): none 0.563, min-max 0.808, max-abs 0.818, z-score 0.827, robust 0.849, log1p(F5) + z-score 0.856, quantile 0.858, Yeo-Johnson 0.867.

### Ties

Same method as fruit.

| Space | L1 | L2 | L∞ |
|---|---|---|---|
| raw F1×F2 | 8 | 10 | 131 |
| z-scored F1×F2 | 5 | 5 | 57 |
| z-scored, all 5 numeric | 0 | 0 | 25 |

Vote ties on z-scored F1×F2: 19 at k = 4 (3 in the binary file), 9 at k = 6. Three rows share an exact (inflow, hold) pair with another row. k = 2 scores 0.736, below k = 1 (0.779).

### Pictures

Same six PNGs as fruit.

---

## Seed robustness

Same parameters, seeds 1–40, counting how many seeds pass. T5–T7 weren't swept; they pass with wide margins on both shipped seeds.

| Target | Fruit | Bank |
|---|---|---|
| T1, multiclass and binary | 35 / 40 | 37 / 40 |
| T2, multiclass | 25 / 40 | 26 / 40 |
| T2, multiclass and binary | 19 / 40 | 6 / 40 |
| T3, at least 5% flips | 40 / 40 | 40 / 40 |
| T3, robust beats min-max on accuracy | 33 / 40 | 36 / 40 |
| T4, at least 5% flips | 40 / 40 | 40 / 40 |
| T4, one-hot adds accuracy | 38 / 40 | 29 / 40 |
| Every target, with margin* | 8 / 40 | 1 / 40 |

\* T1 ≥ 12 pts in both modes; best k 6–14 in both modes with k = 1 at least 4 (multiclass) or 3 (binary) points below; robust beats min-max by ≥ 2 pts with ≥ 5% flips; one-hot adds ≥ 1 pt with ≥ 5% flips.

The weak spot is T2. With 200 rows the validation curve is nearly flat from about k = 5 to k = 20, so where the maximum lands depends on the draw. The bank binary file is easy (0.95), which leaves k = 1 little room to be worse.

## Recommendation

Use **fruit** as the default and keep **bank** as an optional second built-in.

1. **Fruit is more robust.** It passes every target with margin on 8 of 40 seeds, bank on 1. T2 passes in both modes on 19 fruit seeds against 6 bank seeds, and one-hot adds accuracy on 38 against 29.
2. **Fruit's squash can be drawn.** At 41 : 1 the equal-aspect plot is a thin band you can still see. Bank's 10,330 : 1 is a single line, which makes the point but hides the data.
3. **Anyone can read the units.** Grams, centimetres and °Brix need no explanation. Hold time and inflow do.
4. **It's harder, so mistakes are visible.** Best accuracy is 0.73 (bank 0.83), so the confusion matrix, the k = 1 islands and the vote chapter all have plenty of errors to show.
5. **Bank labels carry risk on a public site.** "Mule" and "suspicious" next to made-up thresholds could be read as real detection rules, however clearly the README says otherwise.

Where bank does better: its binary file is 20 / 80, so it shows the imbalance lesson (majority baseline 0.80, mule recall) more sharply than fruit's 45 / 55. It also has the cleaner z-scored clusters. Offer it in chapter 10 and to readers who want a finance example.

### Caveats

- L1 and L2 ties disappear with all five numeric features (0 boundary ties). Ties only show up in 2-D views or under L∞, so the tie demos in chapters 4–5 should use F1×F2.
- The five outliers inflate the F5 class means. Use medians when describing that column.
- The shipped seeds were picked. Other seeds with the same parameters fail T2 about half the time (table above), so if the generator is ever re-seeded, re-check T2 first.
