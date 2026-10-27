# 08 · Evaluation

## Summary
k-NN memorises the training set, so scoring it on the rows it was fit on is meaningless: every row finds itself at distance 0. Use a held-out test split (stratified for classification) or cross-validation, and report per-class precision/recall/F1 next to accuracy, because majority votes favour large classes.

## Math

**Hold-out.** Split rows into train/test. A stratified split keeps class proportions approximately equal in both parts.

**k-fold CV** (scikit-learn docs). Split the data into $F$ folds. For each fold $f$, fit on the other $F-1$ folds and score on fold $f$, then average:

$$\mathrm{CV}=\frac1F\sum_{f=1}^{F}\mathrm{score}_f .$$

- LOOCV is $F=n$.
- `StratifiedKFold` makes "each set contain approximately the same percentage of samples of each target class as the complete set".
- scikit-learn notes that LOO "often results in high variance as an estimator for the test error" and that "5 or 10-fold cross validation should be preferred to LOO".

**Why training-set evaluation is broken for k-NN.** If $\mathbf{x}_q=\mathbf{x}_i$ is a training row, then $\delta(\mathbf{x}_q,\mathbf{x}_i)=0$ and row $i$ is always in $N_k(\mathbf{x}_q)$.
- $k=1$: $\hat y=y_i$, so training accuracy is 100% (barring duplicates with conflicting labels).
- `weights='distance'` (scikit-learn): the zero-distance rule (note 04) gives row $i$ weight 1 and all others 0, so training accuracy is 100% **for every $k$**. Checked on random labels with $k=1,5,15$: 1.0 every time.
- Uniform weights with $k>1$: the row still casts 1 of $k$ votes for itself, which biases the estimate upward.

scikit-learn's `predict(X=None)` / `kneighbors(X=None)` query the training points with themselves excluded ("points are not considered their own neighbors"). In the source, it asks for $k+1$ neighbours and drops the row's own index, so exact duplicates of the row stay in. On random labels, $k=1$: training accuracy 1.0, `predict(None)` accuracy 0.46.

**Fast LOOCV for every $k$ at once.** One search for the $k_{\max}+1$ nearest neighbours of each training row, with self removed, yields the ordered neighbour list. The $k$-NN prediction for any $k\le k_{\max}$ uses the first $k$ entries of that list. So a full LOOCV accuracy-vs-$k$ curve costs one neighbour search: $O(n^2 d)$ brute force. Caveats: if the scaler was fitted on all rows, this is slightly leaky (note 05); and ties at rank $k$ follow row order.

**Confusion matrix** (scikit-learn convention): $C_{ij}$ = number of samples with true class $i$ predicted as class $j$. Rows are true labels, columns are predictions (other references may transpose).

**Per-class metrics** for class $c$, with $\mathrm{tp}_c=C_{cc}$, $\mathrm{fp}_c=\sum_{i\ne c}C_{ic}$, $\mathrm{fn}_c=\sum_{j\ne c}C_{cj}$:

$$\mathrm{precision}_c=\frac{\mathrm{tp}_c}{\mathrm{tp}_c+\mathrm{fp}_c},\qquad \mathrm{recall}_c=\frac{\mathrm{tp}_c}{\mathrm{tp}_c+\mathrm{fn}_c},\qquad F_{1,c}=\frac{2\,\mathrm{precision}_c\,\mathrm{recall}_c}{\mathrm{precision}_c+\mathrm{recall}_c}.$$

**Averages over $M$ classes**, with support $n_c$ = number of true samples of class $c$:
- Macro: $\frac1M\sum_c m_c$ (equal weight per class). Macro-F1 is the mean of per-class F1, not F1 of the mean precision and recall.
- Weighted: $\sum_c \frac{n_c}{n}\,m_c$ (weight by support).
- Micro: pool tp/fp/fn across classes, then compute once. For single-label multiclass, micro precision = micro recall = accuracy (a standard identity; not stated on the page read).
- Balanced accuracy: macro-averaged recall. It equals accuracy for balanced data.

**Class imbalance and the vote.** With uniform weights the vote share estimates the class posterior *including its prior*. If class $c$ has a fraction $\pi_c$ of the rows, then in a region where the classes are equally dense per-class, roughly a fraction $\pi_c$ of the $k$ neighbours are class $c$. So the majority class wins, increasingly as $k$ grows; at $k=n$ it wins everywhere. `KNeighborsClassifier` has no `class_weight` parameter (its parameters are n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs). Rebalancing therefore means a custom vote, e.g. dividing $S_c$ by $\pi_c$, or resampling.

**Validation curve.** For each $k$, plot the training score and the CV score. scikit-learn's reading: both low means underfitting; training high and validation low means overfitting.

## Geometric meaning
- **Train/test split:** the test points are drawn hollow. Only filled (training) points can be neighbours.
- **Self-match:** query a training point and the neighbourhood ball has radius 0 when $k=1$; the point is its own answer.
- **Imbalance:** in a sparse minority region, the ball for large $k$ must stretch into majority territory to collect $k$ points, so the majority colour floods minority pockets on the map as $k$ grows.
- **Confusion matrix:** each off-diagonal cell corresponds to a set of misclassified points. Linking cells to points shows *where* errors sit, typically along the boundary and inside minority pockets.

## Effect on predictions / metrics
- Training accuracy at $k=1$ is 1.0 by construction. CV accuracy at $k=1$ is usually well below that.
- As $k$ grows: recall of the majority class rises and recall of minority classes falls. Macro-F1 often peaks at a smaller $k$ than accuracy does.
- Distance weighting partially protects minority pockets (close minority points outweigh far majority points) but makes training-set scores useless.
- Stratification matters most for small $n$ or rare classes. Without it, a fold can contain zero examples of a class.

## Pitfalls & edge cases
- **Duplicates across the split:** an exact duplicate of a test row in the training set acts like training-set evaluation for that row.
- **Scaler fitted before splitting** is leakage (note 05).
- **LOOCV with duplicates:** scikit-learn removes the row's own index, not its duplicates.
- **Tiny classes:** `StratifiedKFold` needs at least $F$ members per class, or it warns. Per-class precision is undefined when a class is never predicted ($\mathrm{tp}+\mathrm{fp}=0$); scikit-learn has a `zero_division` setting for this case.
- **Accuracy on imbalanced data:** 95% can be "always predict the majority". Show the majority-class baseline next to it.
- **Choosing $k$ on the test set** turns the test set into a validation set. Use CV on the training part to choose $k$, then report the test set once.

## Visualization ideas
- **Split view:** a "Train/Test" toggle with a slider for test fraction and a stratify checkbox. Test points render hollow. A small bar pair shows the class proportions in train vs test, and they match when stratify is on.
- **The self-match trap:** a "score on training data" button runs the evaluation, then an animation zooms into one point with its own label voting for itself, captioned "distance 0, it found itself". It toggles to `predict(None)` (exclude self) and accuracy drops. Under 1/d weights, show the training-accuracy line pinned at 1.0 for all $k$.
- **Validation curve:** accuracy vs $k$ with training (dashed) and CV (solid) lines, plus an optional ±1 sd band across folds. The current $k$ cursor is synced to the main map. A second panel shows macro-F1 vs $k$ to expose imbalance effects.
- **Linked confusion matrix:** clicking cell $(i,j)$ highlights those points on the scatter and dims the rest. Precision/recall bars per class sit beside the matrix, with the formula terms (tp, fp, fn) colour-linked to the matrix cells being summed.
- **Imbalance slider:** subsample the majority class to vary the ratio and watch the minority region shrink as $k$ grows. Optionally toggle "prior-corrected vote" ($S_c/\pi_c$).

## Sources
- scikit-learn cross-validation guide, https://scikit-learn.org/stable/modules/cross_validation.html. Supports: overfitting quote, k-fold procedure, LOO description, high variance, 5/10-fold preference, StratifiedKFold definition, `train_test_split` stratified splitting.
- scikit-learn model evaluation guide, https://scikit-learn.org/stable/modules/model_evaluation.html. Supports: precision/recall/F formulas, macro/weighted/micro definitions and the macro-F1 note, confusion matrix convention $C_{ij}$, balanced accuracy definition.
- scikit-learn validation-curve guide, https://scikit-learn.org/stable/modules/learning_curve.html. Supports: under/overfitting reading of train vs validation scores.
- KNeighborsClassifier API (1.9.1). Supports: parameter list (no `class_weight`), `X=None` excludes self.
- scikit-learn source `_base.py` (kneighbors with `query_is_train`). Supports: $k+1$ query and self-index removal, duplicate corner case.
- scikit-learn source `_base.py` `_get_weights`. Supports: zero-distance rule behind 100% training accuracy.
- FAU MLISP lecture 3. Supports: training error 0 at $k=1$.
- Local runs of scikit-learn 1.9.1. Supports: training-accuracy numbers stated as "checked"; StratifiedKFold warning "The least populated class in y has only 2 members, which is less than n_splits=5"; `precision_score` has a `zero_division` parameter.
- Cornell CS4780 lecture note 02. Supports: $k=n$ returns the most common label.
- The micro = accuracy identity and the "prior in the vote" reasoning are standard derivations; no specific source fetched for them.
