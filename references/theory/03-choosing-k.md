# 03 · Choosing k

## Summary
$k$ trades variance for bias. Small $k$ gives jagged, noise-chasing boundaries; large $k$ averages over a wide region and blurs real structure. The standard regression decomposition puts this exactly: bias² grows with $k$ while variance falls as $\sigma^2/k$. Pick $k$ by cross-validation, never by training accuracy.

## Math

**k-NN regression error at a point** (ESL §7.3 "The Bias–Variance Decomposition"; the form is checked against two lecture notes, see Sources). Model $Y=f(X)+\varepsilon$, $\mathbb{E}[\varepsilon]=0$, $\mathrm{Var}(\varepsilon)=\sigma_\varepsilon^2$. The training inputs $\mathbf{x}_i$ are treated as fixed, and $\hat f_k(\mathbf{x}_0)=\frac1k\sum_{\ell=1}^{k}y_{(\ell)}$ with $(\ell)$ indexing the neighbours of $\mathbf{x}_0$. Then

$$\mathrm{Err}(\mathbf{x}_0)=\underbrace{\sigma_\varepsilon^2}_{\text{irreducible}}+\underbrace{\Big[f(\mathbf{x}_0)-\frac1k\sum_{\ell=1}^{k}f(\mathbf{x}_{(\ell)})\Big]^2}_{\text{bias}^2}+\underbrace{\frac{\sigma_\varepsilon^2}{k}}_{\text{variance}} .$$

- $\sigma_\varepsilon^2$: noise variance, unaffected by $k$.
- Bias²: the gap between the true value at $\mathbf{x}_0$ and the average true value at the $k$ neighbours. It is usually 0-ish for $k=1$ and grows as the neighbours spread out.
- Variance: averaging $k$ independent noise terms divides their variance by $k$.

ESL's equation number for this is **unverified** (I could not open the ESL PDF; see papers/README).

**Classification analogue.** The vote share $\hat p(c\mid\mathbf{x})$ (note 01) is an average of $k$ indicators, so the same logic applies to the *probability estimate*: its variance shrinks like $1/k$ and its bias grows as the neighbourhood widens. 0–1 loss does not decompose additively like squared error, so treat this as intuition, not a formula.

**Effective degrees of freedom.** For k-NN regression smoothing with fixed inputs,

$$\mathrm{df}(\hat f)=\frac{n}{k},$$

which runs from $n$ at $k=1$ (one free parameter per row) to $1$ at $k=n$ (a constant). ESL frames this in §2.3.2 as "effective number of parameters". I verified the $n/k$ value via CMU and FAU lecture notes, but **not** ESL's own wording.

**Extremes**
- $k=1$: zero training error, since each training row is its own nearest neighbour at distance 0 (FAU notes; see the duplicate caveat below).
- $k=n$, uniform weights: every query gets the training majority class (Cornell CS4780).

**Large-sample theory**
- Cover & Hart: with $M=2$, odd $k$ fixed and $n\to\infty$, the bound on the k-NN risk falls monotonically in $k$ towards $R^*$ (note 01).
- For consistency, $k\to\infty$ with $k/n\to0$ (Fix & Hodges, as reported by Cover & Hart).
- Wikipedia (citing Samworth) gives a rate-optimal choice $k^*=\lfloor B\,n^{4/(d+4)}\rfloor$ under regularity conditions. $B$ is an unknown constant, so this sets the growth rate only, not a usable number.

**Heuristics (label as heuristics in the app)**
- $k\approx\sqrt n$: a "rule of thumb" in some course notes (UWO CS4442). It has no derivation in the sources read.
- Odd $k$ for binary problems avoids tied votes (Wikipedia).

**Validation curve.** For each candidate $k\in\mathcal{K}$, compute the training score and a cross-validated score:

$$\mathrm{CV}(k)=\frac1F\sum_{f=1}^{F}\mathrm{Acc}\big(\text{model}_k\text{ fit on folds}\ne f,\ \text{fold } f\big),\qquad \hat k=\arg\max_{k\in\mathcal{K}}\mathrm{CV}(k).$$

LOOCV is $F=n$.

## Geometric meaning
- **2D:** the neighbourhood ball's radius $\delta_{(k)}$ grows with $k$. At $k=1$ each training point "owns" its Voronoi cell (note 06), so islands appear around every mislabelled point. As $k$ grows, islands vanish and the boundary straightens. At $k=n$ the map is one colour.
- **Bias picture:** with a curved true boundary, a large-$k$ ball centred near it reaches across to the other side and pulls in the wrong class. This is visible as the boundary "cutting corners".
- **Variance picture:** at $k=1$, resampling the training set (bootstrap) moves the boundary a lot. At $k=25$ it barely moves.

## Effect on predictions
- Increasing $k$ smooths the boundary and makes probabilities finer-grained (steps of $1/k$). It also pulls predictions toward the majority class (note 08).
- Training accuracy almost always falls as $k$ grows from 1. CV accuracy typically rises then falls (inverted U), though it can be flat.
- With `weights='distance'`, larger $k$ matters less, because far neighbours get small weights. Training accuracy stays at 100% for every $k$ (note 08).

## Pitfalls & edge cases
- **Choosing $k$ by training accuracy always picks $k=1$.**
- **Duplicates:** if two identical rows have different labels, $k=1$ training error is not zero and depends on row order (note 04).
- **Odd $k$ helps only for $M=2$.** With $M\ge3$, ties can occur at any $k\ge2$ (e.g. $k=3$ with votes 1-1-1).
- **$k$ larger than the smallest class** makes that class unable to win a uniform vote in regions where it is surrounded by others.
- **Scaling, encoding and metric change the best $k$.** Tune $k$ *after* fixing preprocessing, or jointly.
- **CV folds must refit the scaler inside each fold** to avoid leakage (note 05).
- **Ties at the $k$-th neighbour** make "$k$" fuzzy. Some implementations include all tied rows; scikit-learn takes exactly $k$ in training-data order (note 04).
- **$\sqrt n$** has no theoretical backing in the sources read. Show it as a starting guess only.

## Visualization ideas
- **k slider + live map:** drag $k$ from 1 to $n$ and watch the boundary map update. A mini-chart underneath shows training accuracy and CV accuracy vs $k$, with a vertical cursor at the current $k$. Add markers for "√n heuristic" and "best CV k".
- **Bias–variance bars (regression demo on a 1D toy):** for a chosen $\mathbf{x}_0$, three stacked bars ($\sigma^2$, bias², $\sigma^2/k$) update with $k$, each colour-linked to its term in the formula. In the 1D plot, the $k$ neighbours of $\mathbf{x}_0$ are highlighted, the bias gap is drawn as a vertical segment between $f(\mathbf{x}_0)$ and the neighbours' mean $f$, and the variance as an error bar of half-width $\sigma/\sqrt k$.
- **Bootstrap flicker:** a toggle that redraws the boundary for 20 bootstrap resamples, overlaid at low opacity. Small $k$ gives a wide fuzzy band; large $k$ gives a tight band. This shows variance directly.
- **Degrees-of-freedom readout:** "$n/k$ = 37.5 effective parameters", next to the slider.
- **Tie warning:** if the current $k$ is even and $M=2$, show how many grid cells or test points ended in a tie, and highlight them on the map.

## Sources
- ESL table of contents (2nd ed.), https://sisis.rz.htw-berlin.de/inh2012/12427576.pdf. Supports: section numbers and titles §2.3.2 "Nearest-Neighbor Methods", §7.3 "The Bias-Variance Decomposition", §7.10 "Cross-Validation".
- CMU (R. Tibshirani) review notes, https://www.stat.cmu.edu/~ryantibs/statml/review/modelbasics.pdf. Supports: $\sigma^2+\big(f(x)-\frac1k\sum_{i\in N_k(x)}f(x_i)\big)^2+\sigma^2/k$ with fixed inputs; $\mathrm{df}=n/k$.
- Toronto STA450 slides (N. Reid), https://utstat.utoronto.ca/reid/sta450/feb25.pdf. Supports: §7.3 framing, $\mathrm{Err}(x_0)=\sigma^2+\mathrm{Bias}^2+\mathrm{var}$, k-NN $\mathrm{var}=\sigma^2/k$ with fixed $x_i$.
- FAU MLISP lecture 3, https://machinelearning.tf.fau.de/pdfs/mlisp/Lecture3.pdf. Supports: effective DoF $n/k$; training error "always be 0 for k = 1".
- Cornell CS4780 lecture note 02, https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html. Supports: $k=n$ gives most common label; ties: fall back to smaller $k$.
- Cover & Hart 1967, local PDF. Supports: k-NN section VIII monotone bound in $k$; Fix & Hodges consistency conditions.
- Wikipedia "k-nearest neighbors algorithm" (raw wikitext). Supports: odd $k$ for binary; $k^*=\lfloor Bn^{4/(d+4)}\rfloor$ (Samworth); larger $k$ reduces noise but blurs boundaries.
- UWO CS4442 (O. Veksler) slides, https://www.csd.uwo.ca/~oveksler/Courses/Winter2015/CS4442_9542b/L3-ML-knn.pdf. Supports: "Rule of thumb is k = sqrt(n)"; cross-validation may be used to choose k.
- scikit-learn validation curve docs, https://scikit-learn.org/stable/modules/learning_curve.html. Supports: train vs validation score vs one hyperparameter; under/overfitting reading.
- scikit-learn cross-validation docs, https://scikit-learn.org/stable/modules/cross_validation.html. Supports: k-fold, LOO definitions and drawbacks.
- ESL itself: **not read** (site disallows automated fetch). The ESL equation number is unverified.
