# 05 · Preprocessing: scaling, encoding, missing values

See also: `10-feature-scaling-deep-dive.md`

## Summary
Distances add up raw per-feature differences, so a feature measured in large units dominates unless the features are rescaled. Encoding choices (one-hot vs ordinal) and missing-value handling also change distances directly. Every scaling, encoding or imputation statistic must be learned from the training split only.

## Math

Let $x_{ij}$ be feature $j$ of row $i$. All statistics below are computed on **training rows only** and then applied unchanged to validation, test and query rows.

**Min–max** (scikit-learn `MinMaxScaler`, default range [0,1]):

$$x'_{ij}=\frac{x_{ij}-\min_j}{\max_j-\min_j},\qquad\text{with feature\_range }(a,b):\ x''_{ij}=x'_{ij}\,(b-a)+a .$$

$\min_j,\max_j$ are the training min and max of feature $j$.

**Z-score** (scikit-learn `StandardScaler`):

$$z_{ij}=\frac{x_{ij}-u_j}{s_j}.$$

$u_j$ is the training mean, and $s_j$ the training standard deviation with the biased estimator ($\text{ddof}=0$, `numpy.std`). If $s_j=0$, scikit-learn uses a scale of 1 (the feature is left un-scaled).

**Robust** (scikit-learn `RobustScaler`):

$$r_{ij}=\frac{x_{ij}-\mathrm{median}_j}{Q_{75,j}-Q_{25,j}}.$$

The default `quantile_range` is (25, 75), i.e. the IQR. Outliers barely affect the median and IQR.

**Worked example: one unscaled feature dominates.** Features (age in years, income in currency units). Query $\mathbf{q}=(30,\ 50000)$. $A=(31,\ 52000)$ is similar in age, slightly different in income. $B=(60,\ 50500)$ is 30 years older, nearly the same income. Euclidean:

| | raw $\delta$ | share of $\delta^2$ from income | z-scored $\delta$ (assume training sd: age 10, income 5000) |
|---|---|---|---|
| $\mathbf{q}\to A$ | $\sqrt{1^2+2000^2}=2000.0$ | 99.99998% | $\sqrt{0.1^2+0.4^2}=0.412$ |
| $\mathbf{q}\to B$ | $\sqrt{30^2+500^2}=500.9$ | 99.6% | $\sqrt{3.0^2+0.1^2}=3.002$ |

Raw: $B$ is nearest, because age is effectively ignored. Scaled: $A$ is nearest. Means cancel in differences, so only the $s_j$ matter for distances.

**One-hot encoding.** A categorical feature with $K$ levels becomes $K$ binary columns (scikit-learn `OneHotEncoder`), exactly one of them equal to 1. For two rows with **different** categories, exactly 2 columns differ, each by 1:

$$\Delta L_1=|1-0|+|0-1|=2,\qquad \Delta L_2^2=1^2+1^2=2\ \ (\text{so }\Delta L_2=\sqrt2\text{ if nothing else differs}),\qquad \Delta L_\infty=1 .$$

The same category contributes 0. Every mismatch costs the same, whatever $K$ is and whichever two categories they are.

With `drop='first'` ($K-1$ columns), a mismatch involving the dropped category flips only 1 column ($\Delta L_1=1$), while a mismatch between two kept categories flips 2 ($\Delta L_1=2$). Distances between categories become asymmetric for no reason.

**Scaling one-hot columns.** A 0/1 column where category $c$ has training frequency $q_c$ has mean $q_c$ and sd $\sqrt{q_c(1-q_c)}$.
- Min–max leaves 0/1 columns unchanged (if both values occur).
- Z-scoring divides each column by $\sqrt{q_c(1-q_c)}$, so a mismatch between categories $a$ and $b$ contributes

$$\Delta L_2^2=\frac{1}{q_a(1-q_a)}+\frac{1}{q_b(1-q_b)} .$$

| $q$ | sd | one column's difference after z-scoring | squared |
|---|---|---|---|
| 0.5 | 0.5 | 2.0 | 4.0 |
| 0.2 | 0.4 | 2.5 | 6.25 |
| 0.1 | 0.3 | 3.33 | 11.1 |
| 0.01 | 0.0995 | 10.05 | 101 |

So rare categories become very far from everything. For comparison, a z-scored numeric feature typically differs by about 1 to 2 units between random rows.

**Ordinal encoding** maps categories to $0,1,\dots,K-1$. Distance then becomes $|{\rm code}_a-{\rm code}_b|$, which imposes an order and equal spacing. scikit-learn warns that estimators "would interpret the categories as being ordered, which is often not desired".

**Missing values.**
- *Drop rows:* fewer training points, and the remaining sample may be biased if missingness is related to the label.
- *Mean/median imputation:* replaces $x_{ij}$ with the training mean/median. On that axis the row sits at the centre, so its difference to any other row along feature $j$ is just that row's deviation from the centre, which says nothing about the imputed row itself.
- *Distance that skips missing coordinates* (scikit-learn `nan_euclidean`, Dixon 1979):
  $$\delta(\mathbf{x},\mathbf{y})=\sqrt{\frac{d}{|P|}\sum_{j\in P}(x_j-y_j)^2},$$
  where $P$ is the set of coordinates present in both rows and $d$ the total number of coordinates. Example from the docs: $[3,\text{na},\text{na},6]$ vs $[1,\text{na},4,5]$ gives $\sqrt{\tfrac42((3-1)^2+(6-5)^2)}$.
- *Gower:* set $w_{ijk}=0$ for a missing variable (note 02). That this was Gower's original treatment is **unverified**; the weight mechanism itself is in the formula.

**Leakage rule** (scikit-learn "Common pitfalls"): "never call `fit` on the test data". Learn $u_j,s_j,\min_j,\max_j$, medians, IQRs, category lists and imputation values from training rows only. Inside cross-validation, refit them in each fold (scikit-learn recommends a `Pipeline`).

**Target labels.** scikit-learn's `KNeighborsClassifier` accepts string labels and encodes them internally with `np.unique(y, return_inverse=True)`. `classes_` is sorted, and that order decides vote ties (note 04) and the column order of `predict_proba`. No separate label encoder is needed.

## Geometric meaning
- **Scaling** stretches or squashes axes. In 2D, an unscaled income axis makes the neighbourhood circle look like a thin vertical sliver in "age" units: neighbours are chosen almost only by income. After z-scoring, the circle is round in standard-deviation units.
- **Min–max** maps the data into the unit square. **Robust** centres the bulk at 0 but leaves outliers far out, so they don't compress everyone else.
- **One-hot** places each category at a vertex of a simplex. All pairs of distinct vertices sit at the same L2 distance $\sqrt2$, which is why one-hot distances are equal for every category pair.
- **Ordinal codes** put categories on a line, so "red–blue" is twice "red–green".
- **Mean imputation** collapses all rows with a missing feature onto the line (in 2D) or hyperplane $x_j=u_j$, which shows up as an unnatural stripe of points.

## Effect on predictions
- Switching raw → z-score → min–max → robust can change neighbour sets and predictions with $k$ and the metric held fixed (example above).
- Z-scoring one-hot columns makes category mismatches dominate when categories are rare.
- More one-hot columns push $d$ up (note 07) and steer scikit-learn's `algorithm='auto'` to brute force once $D>15$ (note 09).
- Leakage makes CV and test scores optimistic. With small $n$ the effect on the chosen $k$ can be visible.

## Pitfalls & edge cases
- **Constant feature:** z-score has $s=0$ and min–max has $\max=\min$ (divide by zero). scikit-learn uses scale 1 for zero variance in `StandardScaler`; pick an explicit rule for min–max.
- **Test values outside the training range** go outside [0,1] after min–max. That is fine, but don't clip silently.
- **IQR = 0** (more than half the rows share one value) breaks robust scaling the same way.
- **Unknown category at test time:** scikit-learn can encode it as all zeros (the docs say so for `handle_unknown='infrequent_if_exist'` when no infrequent category is enabled, and for `'ignore'` together with `drop`). Its distance to any known category is then 1 in L1 rather than 2.
- **Numeric-looking categories** (zip codes, IDs) treated as numbers give meaningless distances. Let the user mark column types.
- **Label parsing:** "1","2","10" as strings sort to ["1","10","2"]; as numbers they sort to [1,2,10]. Tie outcomes differ.
- **Scaling the target** never applies in classification. Scale features only.

## Visualization ideas
- **Axis-dominance demo:** a 2D scatter with raw axes drawn to *true* scale (so the age range is a thin strip), and a toggle (raw / min–max / z / robust) that animates the axes rescaling. The query's neighbourhood circle and its $k$ highlighted neighbours update live.
- **Per-feature contribution bar** for one query–neighbour pair: a stacked bar of $(x_j-y_j)^2$ per feature, coloured by feature and linked to the matching term in $\sum_j(\cdot)^2$. Raw income takes ~100% of the bar; after scaling the bar evens out.
- **One-hot cube:** for a 3-level category, show the three one-hot points as vertices of a triangle (simplex) in 3D, with all edges labelled $\sqrt2$. A "drop first" toggle moves one vertex to the origin, and the edge lengths change to 1, 1, √2.
- **Rare-category inflation:** a slider for category frequency $q$ that shows the per-column contribution $1/\sqrt{q(1-q)}$ growing as $q\to0$, next to a typical z-scored numeric difference for scale.
- **Leakage toggle:** "fit scaler on all data vs train only". Show the two sets of $(u_j,s_j)$ side by side and the CV accuracy difference.
- **Missing-value stripe:** a toggle between drop / mean-impute / nan-euclidean, with imputed points drawn hollow. Under mean-impute they line up on $x_j=u_j$.

## Sources
- scikit-learn preprocessing guide, https://scikit-learn.org/stable/modules/preprocessing.html. Supports: MinMaxScaler formula, fit on training then re-apply on test, OneHotEncoder ($K$ binary columns, `drop`, `handle_unknown` all-zeros), ordinal caution quote, outlier note.
- scikit-learn StandardScaler API, https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html. Supports: $z=(x-u)/s$, biased sd (ddof=0), zero variance gives scale 1.
- scikit-learn RobustScaler API, https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html. Supports: removes median, scales by quantile range, default IQR (25, 75).
- scikit-learn "Common pitfalls", https://scikit-learn.org/stable/common_pitfalls.html. Supports: leakage definition, "never call fit on the test data", Pipeline recommendation.
- scikit-learn `nan_euclidean_distances` API, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.nan_euclidean_distances.html. Supports: formula, example, Dixon 1979 reference.
- scikit-learn source `_base.py`, https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/neighbors/_base.py. Supports: internal `np.unique` label encoding.
- KNeighborsClassifier API. Supports: classes ordered lexicographically in `predict_proba`.
- UWO CS4442 slides, https://www.csd.uwo.ca/~oveksler/Courses/Winter2015/CS4442_9542b/L3-ML-knn.pdf. Supports: a second unnormalised-feature misclassification example (feature ranges 1–2 vs 100–200).
- Wikipedia "k-nearest neighbors algorithm". Supports: normalising features with different units can greatly improve accuracy.
- Wikipedia "Gower's distance". Supports: per-variable weights $w_{ijk}$.
- Worked numbers (age/income table, one-hot frequency table): computed locally, not from a source.
