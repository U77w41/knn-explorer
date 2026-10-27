# 10 · Feature scaling deep dive

## Summary
Every per-feature affine scaler (min–max, z-score, robust, max-abs, manual weights) is the same thing seen differently. It is Minkowski distance in raw units with each coordinate divided by a scale $s_j$. Centring never matters for Minkowski distances, and multiplying every $s_j$ by one constant changes nothing except fixed-bandwidth kernels and fixed radii. So scalers differ only in the *ratios* $s_j/s_{j'}$. Nonlinear transforms (quantile, power, log) also change spacing within a feature, so they can reorder neighbours even in 1-D. The per-sample `Normalizer` is a different kind of operation: it rescales rows, not columns. Mahalanobis distance ignores every affine scaler, while cosine does not even ignore centring. On the five bundled datasets, scaling vs no scaling swaps most of each query's neighbours (wine: 53% of 5-NN sets fully disjoint). The choice *among* reasonable scalers mostly moves accuracy by about 1–4 points. One surprise: scaling makes iris slightly worse. Note 05 covers the basic formulas, one-hot cost and leakage rules, and this note builds on it.

## Math

**Setup.** A transform $T$ is fitted on **training rows only** and then applied unchanged to training, validation, test and query rows. A *column* (per-feature) transform acts as $x'_j=T_j(x_j)$. A *row* transform (`Normalizer`) uses the whole row. KNN then runs unchanged on $T(\mathbf{x})$. Library facts below are for scikit-learn 1.9.1.

| Transform | $x'_j$ | Learned from TRAIN | scikit-learn (defaults) | Kind |
|---|---|---|---|---|
| None | $x_j$ | nothing | `"passthrough"` | identity |
| Min–max | $a+(b-a)\dfrac{x_j-\min_j}{\max_j-\min_j}$ | $\min_j,\max_j$ | `MinMaxScaler(feature_range=(0,1), clip=False)` | affine, $s_j=\frac{\max_j-\min_j}{b-a}$ |
| Z-score | $(x_j-\bar x_j)/\sigma_j$, $\sigma_j$ with ddof=0 | mean, sd | `StandardScaler()` | affine, $s_j=\sigma_j$ |
| Robust | $\dfrac{x_j-\mathrm{med}_j}{Q_{hi,j}-Q_{lo,j}}$ | median, two percentiles | `RobustScaler(quantile_range=(25,75), unit_variance=False)` | affine, $s_j=$ IQR |
| Max-abs | $x_j/\max_i\lvert x_{ij}\rvert$ | max-abs | `MaxAbsScaler()` | linear, $c_j=0$, $s_j=\max\lvert x_j\rvert$ |
| Quantile | $G^{-1}\big(\hat F_j(x_j)\big)$ | $\hat F_j$ (landmark quantiles) | `QuantileTransformer(n_quantiles=1000, output_distribution='uniform', subsample=10_000)` | monotone, nonlinear |
| Power | $\psi_{\lambda_j}(x_j)$, then z-score | $\lambda_j$ (MLE), then mean, sd | `PowerTransformer(method='yeo-johnson', standardize=True)` | monotone, nonlinear |
| log / log1p | $\ln x_j$ / $\ln(1+x_j)$ | nothing | `FunctionTransformer(np.log1p)` | monotone, nonlinear, stateless |
| L2 Normalizer | $x_j/\lVert\mathbf{x}\rVert_2$ (whole row) | nothing | `Normalizer(norm='l2')` | **per row**, nonlinear |
| Manual weights | $w_j\,x_j$ | user-set | (none needed) | linear, $s_j=1/w_j$ |

Transform details:
- **Min–max.** Test values outside $[\min_j,\max_j]$ map outside $[a,b]$. `clip=True` clamps them, and the docs warn this "can distort the distribution of test data".
- **Robust.** `quantile_range=(q_lo, q_hi)` with $0<q_{lo}<q_{hi}<100$. `unit_variance=True` additionally divides the scale by the IQR of $N(0,1)$ ($\approx1.349$, computed locally), so normal features end up with variance 1.
- **Max-abs** does not centre. It maps positive-only data to $[0,1]$ and mixed-sign data to $[-1,1]$, and is the scaler meant for sparse data.
- **Quantile.**
  - $\hat F_j$ is built from `n_quantiles` landmarks (default 1000, capped at the number of training rows) estimated on at most `subsample` rows.
  - In the installed source, `transform` takes $\tfrac12$(forward `np.interp` + backward `np.interp`) so that tied values get the middle of their quantile run. A value equal to the training min or max is set exactly to the bound.
  - Uniform output: $x'=\hat F_j(x)\in[0,1]$.
  - Normal output: $x'=\Phi^{-1}(\hat F_j(x))$, clipped to $\pm\Phi^{-1}(1-10^{-7})\approx\pm5.199$ (source constant `BOUNDS_THRESHOLD = 1e-7`; the ±5.199 values were observed).
  - Test values beyond the training range go to the bounds (docs).
- **Power.** $\lambda_j$ is fitted by maximum likelihood per feature, then the output is z-scored (`standardize=True`). With $\psi$ from the scikit-learn guide:

$$\text{Yeo–Johnson: }\psi_\lambda(x)=\begin{cases}\big((x+1)^\lambda-1\big)/\lambda & \lambda\ne0,\ x\ge0\\ \ln(x+1) & \lambda=0,\ x\ge0\\ -\big((1-x)^{2-\lambda}-1\big)/(2-\lambda) & \lambda\ne2,\ x<0\\ -\ln(1-x) & \lambda=2,\ x<0\end{cases}\qquad \text{Box–Cox: }\psi_\lambda(x)=\begin{cases}(x^\lambda-1)/\lambda & \lambda\ne0\\ \ln x & \lambda=0\end{cases}$$

  Box–Cox needs $x>0$. Yeo–Johnson takes any sign.
- **log / log1p.** `log1p` is Yeo–Johnson with $\lambda=0$ on $x\ge0$, and `log` is Box–Cox with $\lambda=0$. Neither learns anything, so neither can leak.
- **Normalizer.** $\mathbf{x}'=\mathbf{x}/\lVert\mathbf{x}\rVert_2$, one factor per row. `fit` does nothing (the docs call it stateless), and zero rows stay zero (observed). Consequences:
  - No single $s_j$ exists, so the identity below does not apply.
  - On unit rows, $\lVert\mathbf{u}-\mathbf{v}\rVert_2^2=2D_C(\mathbf{u},\mathbf{v})$, so Euclidean KNN on normalised rows ranks like cosine (note 02). This was checked on wine to machine precision.
  - On raw data the row norm is dominated by the biggest-unit feature (wine: `proline` supplies a median 97.7% of $\lVert\mathbf{x}\rVert^2$), so `Normalizer` then behaves almost like "divide every feature by `proline`".

**The key identity.** For any per-feature affine map $x'_j=(x_j-c_j)/s_j$ with $s_j>0$:

$$d'_p(\mathbf{x},\mathbf{y})=\Big(\sum_j\Big|\tfrac{x_j-c_j}{s_j}-\tfrac{y_j-c_j}{s_j}\Big|^p\Big)^{1/p}=\Big(\sum_j\frac{|x_j-y_j|^p}{s_j^{\,p}}\Big)^{1/p},\qquad d'_\infty=\max_j\frac{|x_j-y_j|}{s_j}.$$

So "scale, then plain Minkowski" equals "weighted Minkowski in raw units" with weight $s_j^{-p}$ on $|x_j-y_j|^p$. Checked on wine (70/30 split) for z-score, min–max, min–max(−1,1), max-abs and robust with $p\in\{1,2,3,\infty\}$: max absolute difference $\le7\times10^{-15}$.

**scikit-learn gotcha:** `metric='minkowski', metric_params={'w': w}` computes $(\sum_j w_j|x_j-y_j|^p)^{1/p}$. So reproducing a scaler needs `w = s**-p`, not `1/s`. Verified: `w=s**-p` matches the `StandardScaler` pipeline exactly for p = 1, 2, 3, while `w=1/s` gives different predictions for p = 2 and 3.

Consequences (all verified numerically, script results in brackets):
1. **Centring never changes Minkowski distances or KNN predictions.** $c_j$ cancels. [`StandardScaler()` vs `StandardScaler(with_mean=False)` and `RobustScaler()` vs `with_centering=False`: identical `predict_proba` for p = 1, 2, ∞.]
2. **Min–max, max-abs, z-score and robust differ only through $s_j$**: range, max|x|, sd or IQR. Neighbour order depends only on the ratios $s_j/s_{j'}$. Min–max and z-score give identical neighbours exactly when range$_j/\sigma_j$ is the same for every feature. [That ratio varies by 1.20× across penguins' features (min–max vs z 5-NN Jaccard 0.886), 1.77× on wine (0.717) and 2.96× on breast cancer (0.719).]
3. **A common constant changes neither neighbour sets nor $1/d$ vote proportions.** If $s_j\to c\,s_j$ for all $j$, every distance is divided by $c$, so the ranking is unchanged and every $1/d$ weight is multiplied by $c$. A **fixed-bandwidth Gaussian** weight $e^{-d^2/2h^2}$ becomes $e^{-d^2/2(ch)^2}$, i.e. the bandwidth effectively changes. [Wine, z-scored, $k=15$: $c=0.1$ and $c=10$ leave neighbour sets and uniform/$1/d$ probabilities identical. Gaussian with $h=1$ flips 2/54 and 1/54 test predictions. Gaussian with adaptive $h=d_k$ (note 04) stays identical.] The same applies to a fixed radius (radius neighbours). So `feature_range`, `ddof`, and `RobustScaler(unit_variance=True)` are all global constants. [min–max (0,1) vs (−1,1) and robust with or without `unit_variance`: identical probabilities.]
4. **Nonlinear transforms can change neighbours even in 1-D.** Affine maps divide all 1-D distances by the same $s$, so the order never changes. Example: training $a=10$ (class A), $b=100$ (B), query 40. Raw $|40-10|=30<|40-100|=60$, so 1-NN is A. After log, $1.386>0.916$, so 1-NN is B. A quantile-uniform fitted on $\{1,2,3,4,10,100\}$ maps 10, 40, 100 to 0.8, 0.867, 1.0, so 1-NN is A again (0.067 < 0.133). Three monotone transforms, two different answers.

**Manual weights** fit the same identity: $x'_j=w_jx_j$ means $s_j=1/w_j$, and $w_j=0$ deletes the feature. A weight on top of a scaler gives an effective scale of $s_j/w_j$. So "pick a scaler" and "set feature weights" are one control: in 2-D, the five affine scalers are just five settings of one number, $\log(s_2/s_1)$.

**Mahalanobis is invariant to any invertible linear (hence affine) map.** With $\mathbf{x}\mapsto A\mathbf{x}+\mathbf{b}$, the training covariance becomes $A\Sigma A^\top$, so

$$(A\Delta)^\top(A\Sigma A^\top)^{-1}(A\Delta)=\Delta^\top A^\top A^{-\top}\Sigma^{-1}A^{-1}A\Delta=\Delta^\top\Sigma^{-1}\Delta .$$

Per-feature scaling is $A=\mathrm{diag}(1/s_j)$. [Wine: z-score and min–max give a max relative difference of $1.6\times10^{-15}$. A random 13×13 $A$ (condition number 133) plus a shift gives $1.1\times10^{-7}$, which is float error. In 5×5 CV with $\Sigma$ refitted per fold, Mahalanobis accuracy is identical under none, min–max, z, robust and max-abs: wine 0.925, breast cancer 0.809, iris 0.889.] Nonlinear transforms do change it: wine 5-NN Jaccard after quantile vs raw Mahalanobis is 0.449.

**Cosine is not invariant to centring** (nor to unequal per-feature scaling). It is invariant only to one global constant and to per-row scale. Example: $\mathbf{q}=(3,1)$, $A=(6,2)$, $B=(3,1.5)$. Raw: $D_C(\mathbf{q},A)=0$, $D_C(\mathbf{q},B)=0.010$, so $A$ is nearest. After subtracting $(2,2)$ from all three: $0.293$ vs $0.051$, so $B$ is nearest. With cosine, the scaler's centring term ($\min_j$ for min–max, the mean for z, 0 for max-abs) moves the origin that angles are measured from. [Iris 5-NN cosine, 5×5 CV: none 0.961, max-abs 0.960, z 0.873, min–max 0.807, quantile-uniform 0.747.]

**Outliers.** Take one feature with values $1,\dots,9$ plus one outlier $1000$:

| Scaler | learned centre / scale | inliers 1..9 map to | inlier span | outlier → |
|---|---|---|---|---|
| min–max | min 1, range 999 | [0, 0.008] | 0.008 | 1 |
| max-abs | 1000 | [0.001, 0.009] | 0.008 | 1 |
| z-score | mean 104.5, sd 298.5 | [−0.347, −0.320] | 0.027 | 3.0 |
| robust | median 5.5, IQR 4.5 | [−1.0, 0.778] | 1.78 | 221 |
| quantile-uniform | ranks | [0, 0.889] | 0.889 | 1 (clipped) |
| Yeo–Johnson | λ by MLE | [−1.86, 0.56] | 2.42 | 2.11 |

Add a clean second feature $g\in\{0..9\}$ and query $(2,7)$. Raw and robust both pick the 3-NN $(3,8),(1,5),(5,9)$. Min–max picks $(7,7),(3,8),(9,6)$, because the first feature now contributes a median 0.000 of each inlier's squared distance (0.50 raw/robust). One outlier switched a feature off.

**Leakage and exact cross-validation.** Every learned quantity in the table (min/max, mean/sd, median/IQR, quantile landmarks, $\lambda_j$, imputation values) must come from the training fold. Exact $F$-fold CV refits $T$ in each fold, and exact LOOCV refits it $n$ times. The fast "one neighbour search for all $k$" LOOCV of note 08 fits the scaler once and is an approximation.

For affine scalers the identity keeps exact LOOCV at $O(n^2d)$. For held-out row $i$, compute the leave-one-out scales $s_j^{(-i)}$ and use weighted raw distances $\big(\sum_j|x_{ij}-x_{mj}|^p/(s_j^{(-i)})^p\big)^{1/p}$:
- sd: rank-one downdate in $O(d)$
- min/max: keep the second-smallest and second-largest
- median/IQR: shift one order statistic

[Checked on wine row 0: the weighted form matches refit-and-transform to $1.8\times10^{-15}$.] Nonlinear transforms need the training rows re-transformed per fold.

**Edge cases (scikit-learn behaviour, observed in 1.9.1 unless docs are cited).**

| Situation | What happens |
|---|---|
| Constant column | `_handle_zeros_in_scale` sets any scale $<10\,\varepsilon$ to 1. The column becomes all 0 (Standard, MinMax, Robust, Quantile-uniform, Yeo–Johnson), all 1 (MaxAbs on a non-zero constant) or all −5.199 (Quantile-normal). It contributes 0 to every distance. Harmless, but worth flagging. |
| IQR = 0 but not constant | `RobustScaler` uses scale 1, so the column **stays in raw units**: $[1,2,1,1,9]\to[0,1,0,0,8]$ (min–max gives $[0,.125,0,0,1]$). German `num_dependents` (84.5% = 1) has IQR 0. |
| Binary / one-hot column (see note 05 for z-scoring) | Min–max, max-abs and quantile-uniform keep 0/1. Robust usually keeps it, since the IQR is 0 or 1, but percentile interpolation gave IQR 0.5 at $n=199$, frequency ≈ 0.25. Z-score and Yeo–Johnson give a mismatch $1/\sqrt{q(1-q)}$ (2.0, 2.5, 4.59 at $q=$ .5, .2, .05). **Quantile-normal gives 10.40 at every frequency**, because min → −5.199 and max → +5.199. |
| Discrete numeric with ties at the ends | Quantile-normal pushes the min and max values to ±5.199. A 1..4 column with shares like German `installment_rate_pct` (14/23/15/48%) maps to −5.199, −0.66, −0.14, +5.199. Quantile-uniform maps it to 0, 0.255, 0.445, 1. |
| Test value outside the training range | Min–max goes outside [0,1] (train {0,5,10}, test −5, 12, 20, 50 → −0.5, 1.2, 2, 5). Z and robust are unbounded. **Quantile clamps** (12, 20, 50 → 1.0), so all three become identical on that feature. |
| Missing values | All scalers here ignore NaN in `fit` and keep it in `transform` (docs for Standard, MinMax, MaxAbs, Quantile and Power; observed for Robust). `KNeighborsClassifier` then raises "Input X contains NaN", so impute first. |
| Impute-then-scale vs scale-then-impute | Mean-imputing first shrinks $\sigma_j$ by $\sqrt{n_{obs}/n}$ (observed 0.837 = $\sqrt{140/200}$), which raises that feature's weight by 1/0.837. Scaling first (NaN ignored) keeps $\sigma_j$ of the observed values. Either way the imputed rows sit at the centre (note 05). |
| `n_quantiles` > training rows | Capped at $n$ with a warning. Small folds give a coarse, step-like $\hat F_j$. |
| Box–Cox or log on $x\le0$ | Box–Cox raises an error, and log gives −∞/NaN. Use Yeo–Johnson or log1p ($x>-1$). |

## Geometric meaning
- **Rubber sheet.** An affine scaler stretches the raw plot by $1/s_j$ along axis $j$ (and slides it by $c_j$, which changes nothing), then KNN draws ordinary balls on the stretched sheet. Pulled back to raw units, the radius-$r$ ball is an axis-aligned shape with semi-axes $r\,s_j$:
  - an ellipse for $p=2$
  - a diamond with vertices $\pm r s_j$ for $p=1$
  - a box with half-widths $r s_j$ for $p=\infty$
  - the superellipse $\sum_j|u_j/(r s_j)|^p\le1$ in general

  A large $s_j$ means a long axis, so the feature is cheap to move along and counts for little.
- **Only aspect ratios matter.** A global constant zooms every ball equally. In 2-D, none, min–max, z, robust and max-abs are five aspect ratios $s_2/s_1$ of one ellipse.
- **Boundaries tilt but stay straight under affine scaling.** The scaled-L2 bisector of $\mathbf{a},\mathbf{b}$ is $\{\mathbf{x}:\sum_j(x_j-a_j)^2/s_j^2=\sum_j(x_j-b_j)^2/s_j^2\}$, a hyperplane with normal $S^{-2}(\mathbf{b}-\mathbf{a})$. In raw units that normal is generally not parallel to $\mathbf{b}-\mathbf{a}$. Example: $\mathbf{a}=(1,1)$, $\mathbf{b}=(3,2)$ gives raw bisector $x_2=5.5-2x_1$, and $s=(1,2)$ gives $x_2=17.5-8x_1$ (checked). For L1/L∞, the 45° pieces become slopes $\pm s_2/s_1$ (derived, not checked).
- **Nonlinear transforms bend the sheet unevenly.** Locally the ball is an ellipse with semi-axes $r/|T_j'(x_j)|$, which changes with position:
  - log: semi-axis $\propto x_j$, so balls grow with the value
  - quantile-uniform: $\propto 1/\hat f_j(x_j)$, small where data are dense and large where they are sparse, which gives roughly equal-count neighbourhoods
  - in raw units the neighbourhood is lopsided around the query, and 1-NN boundaries curve (for $\mathbf{a}=(1,1)$, $\mathbf{b}=(4,3)$ under log, the bisector's secant slope runs from −5.9 to −0.12 across $x_1\in[0.5,6]$; computed locally)

  This local-ellipse view is derived, not taken from a source.
- **Normalizer** slides every point along its ray from the origin onto the unit sphere. Neighbourhoods become cones/wedges, as with cosine (note 02).
- **Mahalanobis** rotates and stretches by $\Sigma^{-1/2}$. Any per-feature stretch applied first is absorbed, so the raw-unit ellipse is the same whichever affine scaler is on.
- **Outliers:** under min–max, max-abs and z, one far point stretches the sheet so much that the inliers along that axis collapse to a line. Robust keeps the inlier spacing and leaves the outlier far away. Quantile pulls the outlier back to the edge.

## Effect on predictions

**Experiment setup.**
- scikit-learn 1.9.1 on the bundled CSVs (target = last column).
- `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)`.
- Imputer, one-hot encoder and scaler sit inside a `Pipeline`, so they are refitted per fold.
- `KNeighborsClassifier(n_neighbors=k)` with uniform weights and Minkowski p=2.
- Quantile and power transformers use their defaults (`n_quantiles` capped at the fold size).
- Penguins: `year` dropped. The numeric run drops the 2 rows with missing measurements. The one-hot run uses median imputation for numeric columns, most-frequent imputation for `island`/`sex`, then `OneHotEncoder(handle_unknown='ignore')`.
- "OHE" columns scale the numeric columns only and pass the 0/1 dummies through. "scale all" applies the scaler after encoding, dummies included.
- Bold marks the best value per column.
- The mean accuracy over 25 folds is shown. As a rough guide, differences under about 0.01–0.02 are within noise (fold sd shown under each table).
- Scripts were run in a scratch directory and are not in the repo.

**iris (150 × 4)**

| scaler | k=1 | k=5 | k=15 |
|---|---|---|---|
| none | 0.956 | **0.963** | 0.964 |
| min-max | 0.956 | 0.955 | 0.965 |
| z-score | 0.945 | 0.953 | 0.959 |
| robust | 0.931 | 0.935 | 0.915 |
| max-abs | 0.951 | 0.957 | 0.956 |
| quantile-uniform | 0.955 | 0.943 | 0.953 |
| quantile-normal | 0.949 | 0.943 | 0.963 |
| yeo-johnson | 0.933 | 0.951 | 0.957 |
| L2 Normalizer | **0.963** | 0.961 | **0.972** |

Fold-level sd 0.029–0.047.

**wine (178 × 13)**

| scaler | k=1 | k=5 | k=15 |
|---|---|---|---|
| none | 0.734 | 0.699 | 0.700 |
| min-max | 0.952 | **0.969** | **0.979** |
| z-score | 0.954 | 0.962 | 0.969 |
| robust | 0.956 | 0.953 | 0.960 |
| max-abs | **0.958** | 0.952 | 0.940 |
| quantile-uniform | 0.949 | 0.961 | 0.958 |
| quantile-normal | 0.951 | 0.943 | 0.931 |
| yeo-johnson | 0.948 | 0.955 | 0.949 |
| L2 Normalizer | 0.837 | 0.815 | 0.771 |

Fold-level sd 0.016–0.073.

**breast-cancer-wisconsin (569 × 30)**

| scaler | k=1 | k=5 | k=15 |
|---|---|---|---|
| none | 0.917 | 0.932 | 0.930 |
| min-max | 0.949 | 0.967 | **0.967** |
| z-score | 0.949 | 0.967 | 0.960 |
| robust | 0.947 | 0.963 | 0.959 |
| max-abs | **0.955** | 0.966 | 0.965 |
| quantile-uniform | 0.950 | 0.968 | 0.964 |
| quantile-normal | 0.947 | 0.968 | 0.960 |
| yeo-johnson | 0.952 | **0.969** | 0.962 |
| L2 Normalizer | 0.909 | 0.923 | 0.921 |

Fold-level sd 0.015–0.028.

**palmer-penguins: numeric (342 × 4) and one-hot island/sex + imputation (344 × 9 after encoding)**

| scaler | num k=1 | num k=5 | num k=15 | OHE k=1 | OHE k=5 | OHE k=15 | OHE, scale all, k=5 |
|---|---|---|---|---|---|---|---|
| none | 0.850 | 0.781 | 0.763 | 0.837 | 0.780 | 0.752 | 0.780 |
| min-max | 0.980 | **0.987** | 0.980 | **0.991** | 0.990 | **0.994** | 0.990 |
| z-score | 0.981 | 0.986 | **0.981** | 0.990 | 0.991 | 0.993 | 0.991 |
| robust | 0.982 | **0.987** | **0.981** | 0.989 | 0.990 | 0.992 | 0.990 |
| max-abs | **0.984** | 0.985 | 0.981 | 0.990 | 0.991 | 0.992 | 0.991 |
| quantile-uniform | 0.977 | 0.982 | 0.977 | 0.989 | **0.992** | 0.991 | **0.992** |
| quantile-normal | 0.973 | 0.971 | 0.970 | 0.982 | 0.985 | 0.981 | 0.988 |
| yeo-johnson | 0.978 | 0.982 | 0.978 | 0.987 | 0.991 | 0.993 | 0.991 |
| L2 Normalizer | 0.967 | 0.960 | 0.894 | 0.964 | 0.943 | 0.886 | 0.953 |

Fold-level sd 0.008–0.044.

**german-credit: numeric (1000 × 7) and one-hot (1000 × 61)**. Majority-class baseline 0.700.

| scaler | num k=1 | num k=5 | num k=15 | OHE k=1 | OHE k=5 | OHE k=15 | OHE, scale all, k=5 | OHE balanced acc k=5 |
|---|---|---|---|---|---|---|---|---|
| none | 0.608 | 0.650 | 0.692 | 0.608 | 0.651 | 0.692 | 0.651 | 0.524 |
| min-max | 0.627 | 0.653 | 0.690 | **0.718** | 0.726 | 0.730 | 0.726 | 0.622 |
| z-score | 0.626 | 0.666 | 0.692 | 0.693 | 0.732 | 0.731 | 0.727 | 0.619 |
| robust | **0.642** | 0.674 | **0.700** | 0.706 | **0.740** | **0.751** | **0.740** | **0.641** |
| max-abs | 0.625 | 0.653 | 0.685 | 0.710 | 0.729 | 0.731 | 0.729 | 0.624 |
| quantile-uniform | 0.619 | 0.658 | 0.676 | 0.708 | 0.732 | 0.734 | 0.732 | 0.629 |
| quantile-normal | 0.637 | 0.657 | 0.684 | 0.668 | 0.700 | 0.707 | 0.724 | 0.570 |
| yeo-johnson | 0.633 | 0.661 | 0.681 | 0.695 | 0.733 | 0.728 | 0.718 | 0.626 |
| L2 Normalizer | 0.608 | **0.683** | 0.697 | 0.691 | 0.723 | 0.730 | 0.683 | 0.618 |

Fold-level sd 0.013–0.033. Min–max, max-abs, robust and quantile-uniform give the same numbers in "OHE" and "scale all" because they leave 0/1 columns as 0/1 (see the edge-case table).

**Who dominates the distance.** This is the median, over all row pairs, of the largest single feature's share of $\lVert\mathbf{x}-\mathbf{y}\rVert_2^2$. The scaler was fitted on all rows, since this is descriptive only.

| dataset | none (top feature, % of pairs) | z-score | min-max | robust | quantile-u | L2 Normalizer |
|---|---|---|---|---|---|---|
| wine | 0.997 (`proline`, 97%) | 0.288 | 0.300 | 0.304 | 0.257 | 0.867 (`magnesium`) |
| breast cancer | 0.689 (`area_worst`, 84%) | 0.168 | 0.171 | 0.179 | 0.123 | 0.412 |
| iris | 0.679 (`petal_length`, 68%) | 0.494 | 0.489 | 0.583 | 0.487 | 0.575 |
| penguins (num) | 1.000 (`body_mass_g`, 99%) | 0.528 | 0.531 | 0.534 | 0.524 | 0.912 |
| german (num) | 1.000 (`credit_amount_dm`, 99%) | 0.492 | 0.589 | 0.498 | 0.465 | 0.873 |

**Neighbour sets, k=5.** The table pools every test query over the 25 folds (wine 890 queries, breast cancer 2845). Jaccard = $|N_a\cap N_b|/|N_a\cup N_b|$, and "flips" counts predictions that change.

| scaler | wine J vs none | wine J vs z | wine flips vs none | BC J vs none | BC J vs z | BC flips vs none |
|---|---|---|---|---|---|---|
| min-max | 0.077 | 0.717 | 30.3% | 0.054 | 0.719 | 7.9% |
| z-score | 0.076 | 1 | 29.8% | 0.053 | 1 | 7.7% |
| robust | 0.072 | 0.813 | 30.1% | 0.055 | 0.744 | 7.6% |
| max-abs | 0.080 | 0.555 | 30.2% | 0.055 | 0.643 | 8.2% |
| quantile-uniform | 0.072 | 0.540 | 30.7% | 0.047 | 0.435 | 7.8% |
| quantile-normal | 0.071 | 0.552 | 30.6% | 0.053 | 0.502 | 7.5% |
| yeo-johnson | 0.078 | 0.670 | 30.0% | 0.049 | 0.535 | 7.5% |
| L2 Normalizer | 0.166 | 0.103 | 27.4% | 0.143 | 0.068 | 5.3% |

None → z-score, wine: 265 flips (249 fixed, 15 broken, 1 wrong-to-wrong). No query kept its neighbour set, and 53.0% of queries share none of their 5 neighbours. Breast cancer: 218 flips (159 fixed, 59 broken), and 65.0% of sets are disjoint. Between z-score and the other affine scalers, prediction flips stay at 1.0–3.5% even though the Jaccard overlap is only 0.56–0.81. Neighbour sets are much less stable than predictions.

Other datasets, none → z: iris 13 flips (3 fixed, 10 broken), penguins 359 (355 fixed, 4 broken), german 1283 (682 fixed, 601 broken).

**Metric × scaler, k=5** (Mahalanobis = whitening with the fold's covariance, then L2):

| scaler | wine L1 | wine L2 | wine L∞ | wine cos | wine Mahal. | BC L1 | BC L2 | BC L∞ | BC cos | BC Mahal. |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.759 | 0.699 | 0.681 | 0.815 | 0.925 | 0.938 | 0.932 | 0.927 | 0.923 | 0.809 |
| min-max | **0.970** | **0.969** | **0.947** | 0.949 | **0.925** | **0.969** | 0.967 | 0.948 | 0.915 | 0.809 |
| z-score | 0.962 | 0.962 | 0.923 | 0.960 | **0.925** | 0.967 | 0.967 | 0.940 | **0.962** | 0.809 |
| robust | 0.966 | 0.953 | 0.924 | 0.944 | **0.925** | 0.964 | 0.963 | 0.942 | 0.954 | 0.809 |
| max-abs | 0.958 | 0.952 | 0.922 | 0.942 | **0.925** | 0.966 | 0.966 | 0.944 | 0.944 | 0.809 |
| quantile-uniform | 0.965 | 0.961 | 0.941 | 0.921 | 0.912 | 0.967 | 0.968 | **0.951** | 0.850 | **0.911** |
| yeo-johnson | 0.954 | 0.955 | 0.933 | **0.964** | 0.901 | 0.966 | **0.969** | **0.951** | 0.955 | 0.892 |

Unscaled, the metric choice matters (wine L1 0.759, L∞ 0.681, cosine 0.815). Scaled, L1 and L2 land within 0.013 of each other, and L∞ is 1.7–3.9 points below L2. Mahalanobis rows are identical for all affine scalers, which confirms the invariance in CV.

**Best k (odd 1–31, 5×5 CV) per scaler:**

| dataset | none | min-max | z-score | robust | quantile-u | yeo-johnson |
|---|---|---|---|---|---|---|
| iris | 13 (0.967) | 15 (0.965) | 13 (0.964) | 3 (0.940) | 9 (0.959) | 13 (0.963) |
| wine | 1 (0.734) | 15 (0.979) | 25 (0.976) | 23 (0.970) | 11 (0.969) | 7 (0.958) |
| breast cancer | 11 (0.933) | 7 (0.971) | 5 (0.967) | 11 (0.965) | 7 (0.969) | 5 (0.969) |
| penguins (num) | 1 (0.850) | 5 (0.987) | 5 (0.986) | 5 (0.987) | 5 (0.982) | 3 (0.984) |
| german (num) | 31 (0.705) | 31 (0.700) | 27 (0.699) | 19 (0.703) | 31 (0.698) | 29 (0.697) |
| german (OHE) | 31 (0.705) | 13 (0.733) | 7 (0.737) | 9 (0.756) | 9 (0.737) | 5 (0.733) |

The validation curve has to be recomputed whenever the scaler changes. Wine's best $k$ goes from 1 to 25.

**Leakage** (scaler fitted on all rows before CV, minus fitted inside each fold, k=5, accuracy points):

| dataset | min-max | z-score | robust | quantile-u | yeo-johnson |
|---|---|---|---|---|---|
| iris | +0.13 | +0.00 | −0.13 | −0.67 | −0.13 |
| wine | −0.45 | +0.23 | −0.12 | −0.23 | −0.23 |
| breast cancer | +0.04 | +0.07 | −0.11 | −0.25 | +0.07 |
| penguins (num) | +0.00 | −0.06 | +0.06 | −0.12 | −0.17 |
| german (num) | −0.00 | +0.04 | +0.02 | −0.36 | +0.14 |

- **LOOCV, k=5, leaky vs exact** (refit on $n-1$ rows): at most 1 prediction differs per dataset and scaler (iris, wine, penguins, breast cancer × z, min–max, quantile-uniform). Under z-score, 5 of wine's 178 rows get a different neighbour set but none a different prediction.
- **30-row wine subsamples** (40 draws): the mean leaky − proper difference is −0.08 to +0.75 points by scaler, and single draws range from −6.7 to +8.3.
- So unsupervised scaling leaks only a little here, and in both directions. It is still worth teaching, because the bias is real at small $n$ and much larger for supervised steps such as feature selection (scikit-learn pitfalls page).

**Per chapter, what the global scaler changes:**
- **Neighbours:** the set itself.
- **Vote:** uniform votes change only through the set. $1/d$ votes also change through distance *ratios*, but not through a global constant. A fixed-$h$ Gaussian changes even under `feature_range` or `unit_variance`.
- **Choosing k:** the whole curve and the best $k$ (table above).
- **Metric:** L1/L2/L∞ rankings, cosine (centring-dependent) and Mahalanobis (immune to affine scalers).
- **Boundaries:** they tilt (affine) or bend (nonlinear) in raw units.
- **Evaluation:** per-fold refitting. Accuracy differences between reasonable scalers are often inside fold noise.

**Surprises worth designing around:**
1. **Column scaling slightly hurts iris** (k=5: none 0.963, z 0.953, robust 0.935), and of 13 none→z flips, 10 are new errors. Total spread $\sigma_j$ includes between-class spread, so dividing by it down-weights the features that separate the classes. Petal length has total/within-class sd ratio 4.13, while sepal width has 1.29. Relative to petal length, z-score gives sepal width 4.05× the weight and robust 7.0× (IQR 0.5 vs 3.5).
2. **The L2 Normalizer is erratic.** It is the best on iris (0.972 at k=15) and on german numeric at k=5 (0.683), but poor on wine (0.815) and below no scaling on breast cancer (0.923 vs 0.932). It is a row operation: on raw data it mostly divides by the largest-unit feature.
3. **Quantile-normal is the weakest column scaler at k=5** on wine, penguins and german-OHE (0.700 vs 0.732 for z). Its ±5.199 clipping turns ties at a column's min/max, including all binary columns, into a jump of about 10.4.
4. **Unscaled, wine's best k is 1.** Every larger $k$ averages over neighbours chosen by `proline` alone.
5. **German credit:** on numeric features nothing clears the 0.700 majority baseline by more than half a point (best is none at k=31, 0.705). With one-hot features, robust scaling is best (0.740, balanced 0.641), and the uniform-vote best $k$ shifts from 31 (none) to 5–13 depending on the scaler.
6. **Mahalanobis on breast cancer** is poor (0.809, below unscaled L2's 0.932), and only nonlinear transforms help it (quantile-uniform 0.911). The 30 features are highly collinear (radius, perimeter and area), so whitening amplifies noise directions. This explanation is a hypothesis, not checked.

**Literature agrees in direction.**
- de Amorim et al. (82 binary KEEL datasets, KNN k=5, scaler fitted per fold) report mean KNN F1 of 0.51 unscaled vs 0.53–0.56 for five scalers, with quantile best. Their Friedman test across scalers is significant for KNN on F1 over all datasets (p = 0.018) but not on G-mean (p = 0.073). They note KNN "is only sensitive to scale when the difference among the features significantly changes an instance's set of neighbors".
- Ahsan et al. (one 303-row heart-disease dataset, k unreported, fitting protocol unreported) report KNN accuracy of 0.75 unscaled vs 0.79–0.87 scaled.

## Pitfalls & edge cases
- **Mixing up `w` in scikit-learn's Minkowski:** it multiplies $|\Delta|^p$, so emulating a scaler needs $w_j=s_j^{-p}$. If the app shows "scaler = weights", it must show which exponent it uses.
- **Changing `feature_range` or `unit_variance` looks like a new scaler** but changes nothing for neighbours. It does change fixed-$h$ Gaussian weights and fixed radii. Tie $h$ or $r$ to the data (e.g. $h=d_k$) or say so.
- **Centring is invisible for Minkowski but changes cosine.** A "centre" toggle should be greyed out for L1/L2/L∞ and live for cosine.
- **Mahalanobis + affine scaler:** the scaler setting does nothing. Say so in the UI rather than letting users think it was applied.
- **Outliers:**
  - A single extreme training value switches a feature off under min–max, max-abs and (less so) z.
  - Robust keeps the inliers but leaves huge test values huge.
  - Quantile clamps, so distinct extreme test values become equal.
- **IQR = 0 under RobustScaler** silently leaves a column in raw units. Count-like or mostly-constant columns (`num_dependents`) can then dominate or vanish depending on their units.
- **Quantile-normal on binary, one-hot or tied columns** inflates mismatches to about 10.4 per column. Prefer quantile-uniform, or exclude such columns from nonlinear transforms.
- **Normalizer is not a column scaler:** with no column scaler before it, it inherits the raw-unit dominance it was supposed to fix. If it is offered, place it after a column scaler and label it "row direction".
- **Leakage:** refit per fold, and for exact LOOCV refit per row (cheap for affine scalers via the identity). Label the fast LOOCV path "approximate".
- **Missing values:**
  - Decide impute→scale or scale→impute; they give different $\sigma_j$ (×0.837 at 30% missing).
  - Never feed NaN to KNN.
  - `nan_euclidean` (note 05) is the alternative.
- **Constant columns** contribute nothing and are harmless, but the app should flag them. MaxAbs maps a non-zero constant to 1, not 0.
- **Small training folds** make quantile and power fits unstable (few landmarks, noisy $\lambda$). The CV spread across folds should be shown.
- **Box–Cox/log need positive inputs.** Check the training *and* query values, since a query can be ≤ 0 even when no training value is.
- **Neighbour sets ≠ predictions:** on wine and breast cancer, column scalers overlap z-score's 5-NN sets with Jaccard of only 0.44–0.81, yet flip only 1.0–4.6% of its predictions. Show both, or users will over-read the churn.

## Visualization ideas
- **Rubber-sheet morph:**
  - A slider $t\in[0,1]$ animates every point from raw to scaled coordinates. For affine scalers, interpolate $\log s_j$ so the stretch is smooth. For nonlinear ones, use $(1-t)x+tT(x)$ with tick labels that warp along.
  - Neighbour links to the query are recomputed every frame. A link that appears or disappears flashes, and a counter shows "neighbours changed so far: m/k".
  - The query's radius-$d_k$ ball is drawn in raw units as the ellipse with semi-axes $d_k s_j$. It visibly becomes the circle at $t=1$.
- **Per-feature contribution bars:**
  - For the query and each neighbour, a stacked bar of $|x_j-y_j|^p/s_j^p$, raw and scaled side by side.
  - Show the top 5 features plus "other", colour-linked to the matching term in the identity.
  - Wine raw opens at a median 99.7% `proline`.
- **Decision-map small multiples:**
  - One panel per scaler (9), same two features, same $k$, all drawn in **raw** coordinates so the tilt (affine) and curvature (nonlinear) of boundaries is visible. Each panel carries its CV accuracy badge.
  - Hovering a cell highlights the same cell in all panels.
  - Reuse cached per-cell neighbour lists per scaler (note 09: the cost is 9× one map, so compute lazily).
- **Neighbours lost/gained diff:**
  - Around the query, compare scaler A with scaler B: kept neighbours grey, lost ones hollow red, gained ones solid green.
  - Show the Jaccard value, vote bars before and after, and a "prediction flipped?" chip.
  - Default to none vs z-score on wine, where 53% of sets are disjoint.
- **Per-scaler accuracy chart:**
  - Horizontal bars (9 scalers), sorted, with ±fold-sd whiskers.
  - Dashed reference lines for "none" and the majority-class baseline.
  - A $k$ selector and an accuracy/balanced-accuracy switch (needed for German credit).
- **"Scaling = feature weights" sliders:**
  - One log-scale weight slider per feature, $w_j=1/s_j$.
  - Buttons snap the sliders to the z, min–max, robust or max-abs weights (normalised so the largest is 1). A badge then shows "identical neighbours and predictions to StandardScaler".
  - In 2-D a single aspect-ratio slider $\log(s_2/s_1)$ is enough, with the five affine scalers as tick marks.
  - The live formula pane shows $(\sum_j|\Delta_j|^p/s_j^p)^{1/p}$ with the current $s_j$, plus a small note that scikit-learn's `w` equals $s_j^{-p}$.
- **Outlier drag demo:**
  - The user drags one training point far along one axis. Under min–max, max-abs and z the other points' scaled coordinates collapse onto a line, and an "inlier span" readout drops (e.g. 1 → 0.008). Neighbour links reorganise around the other feature.
  - Robust stays put, and quantile pins the dragged point to the edge.
  - Reset button.
- **Leakage toggle:**
  - "Fit scaler on: training fold only / all rows". It shows the two $(c_j,s_j)$ vectors side by side, per-fold $s_j$ as small bars, the CV accuracy difference, and the count of LOOCV predictions that differ.
  - The honest message is usually "tiny here (≤0.7 points), larger with small $n$".
- **Extra, cheap:**
  - A **1-D strip** with the log-vs-affine 1-NN flip (10, 40, 100).
  - A **centring toggle** that does nothing for L2 and flips neighbours for cosine.
  - A **"no effect" lock** on the scaler selector when the metric is Mahalanobis.
  - A **warning chip** when quantile-normal meets binary or tied columns.

## Sources
- scikit-learn 1.9.1 preprocessing guide, https://scikit-learn.org/stable/modules/preprocessing.html. Supports: MaxAbsScaler for centred/sparse data; RobustScaler for outliers; quantile transform $G^{-1}(F(X))$, rank preservation, "distort correlations and distances within and across features"; Yeo–Johnson and Box–Cox formulas, MLE λ, positivity for Box–Cox, default standardisation; Normalizer as per-sample unit norm and stateless; `FunctionTransformer(np.log1p)`; fit on train then re-apply.
- scikit-learn API pages (1.9.1), all under https://scikit-learn.org/stable/modules/generated/:
  - `sklearn.preprocessing.StandardScaler.html`: formula, ddof=0, zero variance → scale 1, NaN handling, outlier sensitivity.
  - `sklearn.preprocessing.MinMaxScaler.html`: formula, `feature_range`, `clip` and its warning, NaN handling.
  - `sklearn.preprocessing.MaxAbsScaler.html`: no centring, sparse input, NaN handling.
  - `sklearn.preprocessing.RobustScaler.html`: `quantile_range`, `unit_variance`, `with_centering`. The page is silent on NaN and IQR = 0, so both were observed instead.
  - `sklearn.preprocessing.QuantileTransformer.html`: per-feature CDF mapping, out-of-range values to the bounds, `n_quantiles`, `subsample`, NaN handling, the non-linear distortion note.
  - `sklearn.preprocessing.PowerTransformer.html`: methods, MLE, `standardize`, NaN handling.
  - `sklearn.preprocessing.Normalizer.html`: rows rescaled independently, `norm`, stateless, cosine remark.
- scikit-learn example "Importance of Feature Scaling", https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html. Supports: KNN (k=20) on `proline`/`hue` gives "completely different models" scaled vs unscaled; caveat that scaling non-predictive small-scale features can hurt.
- scikit-learn example "Compare the effect of different scalers on data with outliers", https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html. Supports: MinMax "compresses all inliers into the narrow range [0, 0.005]" on California housing; StandardScaler and MinMax "very sensitive" to outliers; RobustScaler not influenced by a few large outliers; quantile collapses outliers to the bounds ("saturation artifacts"); Normalizer maps samples onto the unit circle.
- scikit-learn "Common pitfalls", https://scikit-learn.org/stable/common_pitfalls.html. Supports: leakage definition, "never call `fit` on the test data", Pipeline in CV, and that the risk covers `StandardScaler` and `SimpleImputer`.
- scikit-learn 1.9.1 installed source `sklearn/preprocessing/_data.py` (read locally). Supports: `_handle_zeros_in_scale` (scale < 10ε → 1), `BOUNDS_THRESHOLD = 1e-7`, `np.interp` forward/backward averaging and exact bound assignment in `QuantileTransformer._transform_col`.
- de Amorim, Cavalcanti & Cruz, arXiv:2212.12343 (local PDF, see `../papers/`). Supports: KNN F1/G-mean table, Friedman p-values, per-fold fitting, the KNN sensitivity quote, Singh & Singh summary.
- Ahsan et al. 2021, MDPI *Technologies* 9(3):52, HTML page read (PDF download refused). Supports: KNN accuracy row (Table 5) and the unreported protocol details.
- Notes 02 (cosine/Normalizer equivalence, Mahalanobis), 04 (Gaussian weights, $h=d_k$), 05 (basic scalers, one-hot cost, leakage rule) and 08 (fast LOOCV).
- Everything else was computed locally with scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.5 and SciPy 1.18.1 on the bundled CSVs: all identity checks, worked examples, edge-case behaviours marked "observed", and all experiment tables. Statements marked "derived" (L1/L∞ boundary slopes, the local-ellipse view of nonlinear transforms) and the Mahalanobis-on-breast-cancer explanation are not from a source and were not separately tested.
