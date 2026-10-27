# KNN Explorer: design spec

- **Date:** 2026-10-27
- **Owner:** Ujjwal Chowdhury
- **Status:** draft for review
- **Repo:** `u77w41/knn-explorer`
- **Target URL:** `https://u77w41.github.io/knn-explorer/`
- **Inputs:** everything under `references/`. Section numbers like "theory/04" point to those notes.

## 1. Purpose

A static web app that shows how k-nearest-neighbours classification works inside, one visible step at a time. The math and the geometry sit side by side, so you can see how every setting changes the result. It starts on a built-in dummy dataset. When someone loads their own labelled tabular data, the same explanations rebuild on their data.

**Audience.** Two groups at once: people who already know ML and want depth (the owner), and learners with less background who need a gentle start. The design is chapters that open simply, plus a "Math" card in every chapter for the rigour.

### 1.1 What the owner asked for

1. Visualise KNN's inner workings in depth.
2. Accept the user's labelled numeric tabular data, binary or multiclass.
3. For non-numeric columns, ask the user to one-hot encode, transform otherwise, or ignore.
4. Show how hyperparameters and distance functions change results, with the math and its geometric meaning side by side.
5. Target visual learners.
6. Host on GitHub Pages.
7. Explain topics with dummy data by default. When the user adds data, swap it in and rebuild the algorithm and the explanations on it.
8. Offer scaling options, and show how scaling changes outcomes inside the algorithm.
9. Collect reference materials into the repo before building (done: `references/`).

### 1.2 Choices made during brainstorming

- Approach A: a lean rebuild in place (not a patch of the Lovable code, not Pyodide).
- Chaptered lessons, scroll-driven.
- Data size up to about 20k rows × 50 columns.
- All four views for data with more than 2 features: slice, PCA plane, per-feature distance bars, scatter-plot matrix.
- The fruit dataset is the default, and bank is an extra built-in sample.

### 1.3 Assumptions (not stated by the owner, accepted by default)

- **Classification only.** The Lovable regression tab is dropped.
- **All computation runs in the browser.** User data is never uploaded, stored or sent anywhere.
- **English UI only.**

### 1.4 Success criteria

1. For any query point, a learner can follow every step from raw CSV row to predicted label. Every number is visible and linked to the formula term that produced it.
2. Changing k, metric, p, weighting, a tie rule, scaler A/B, an encoding choice or the split updates the affected views within the budgets in §6.8.
3. On the fixture suite (§10.1), predictions, probabilities, neighbour indices and distances match scikit-learn 1.9.x. Indices must match exactly; floating-point values within 1e-9.
4. A user CSV of 20k rows × 50 columns works end to end in a current desktop browser without freezing the page.
5. Every push to `main` deploys the site to the target URL through GitHub Actions, and the deployed site passes the end-to-end smoke test.
6. axe-core reports no serious or critical violations on any chapter.

## 2. Scope

**In scope:** chapters 0–11 (§3); the data pipeline (§5); the engine (§6); the visual blocks (§7); the app shell (§8); deploy and CI (§9); tests (§10).

**Out of scope:**
- Regression
- A different scaler per column (one global scaler, plus manual per-feature weights)
- KD-tree and ball-tree search (brute force only; the theory is in theory/09)
- Metric learning such as NCA or LMNN (references only)
- Gower distance
- t-SNE and UMAP views
- Saving user data between sessions
- Any backend or analytics
- Resampling for class imbalance (e.g. SMOTE)
- Exporting predictions

## 3. Learning structure

### 3.1 Layout

Each chapter is a scrolling story:
- **Left column:** narrative, plus a collapsible "Math" card.
- **Right column:** a visual stage that stays in view while the reader scrolls.
- **Top bar:**
  - an active-dataset chip ("Dummy: Fruit" or "Your data: file.csv"; one click switches back)
  - chapter progress
  - the settings unlocked so far
  - scaler A and an optional "Compare with" scaler B, both unlocked in Chapter 3

**Progressive disclosure.** A chapter unlocks a control by introducing it, and the control then stays in the top bar for every later chapter.

**One query point runs through the whole course.** By default it is a seeded random test row. The reader can instead pick any row, type raw values, or click the plane view.

### 3.2 Chapters

"Needs" lists properties of the data that a demo depends on. If the active dataset lacks one, that demo alone falls back to the fruit dummy with the label "Using the dummy data here because your data has no ⟨property⟩", and the rest of the chapter stays on the user's data.

| # | Chapter | Learning goal | Main visuals | Controls introduced | Scaling impact shown | Needs |
|---|---|---|---|---|---|---|
| 0 | Bring your data | KNN needs numeric vectors and labels | Data table with column-type badges, class-balance bar, missing-value map, animation of one-hot columns expanding | Dataset picker, upload, column wizard, split | none | none |
| 1 | Points in space | A row is a point in d-dimensional space | Plane view (pick 2 features), scatter-plot matrix (≤ 8 features), PCA plane | Axis pickers, view mode | Teaser: notice the axes' units | ≥ 2 numeric features after encoding |
| 2 | Measuring distance | The Minkowski family and its geometry | Query to one training point: L2 as a hypotenuse, L1 as a staircase, L∞ as the longest side. Unit ball morphing with p. Distance bars. Formula panel with numbers plugged in. The p < 1 counterexample. | Metric, p | none | none |
| 3 | Why units matter | Distance depends on units; scaling is choosing feature weights | Raw view at equal aspect, rubber-sheet morph to scaled; distance bars raw vs scaled; gallery of scalers with formulas; dragging an outlier; weight sliders; cost of a one-hot mismatch | Scaler A, Compare B, manual weights, scale-one-hot toggle | This chapter is the scaling story | Features on different scales; an outlier column; a categorical column (each falls back separately) |
| 4 | Finding the k nearest | Brute-force search | Distance ladder (fill, sort, cut at k), neighbourhood ball of radius d_k, neighbour table, rank-tie flag | k, step / play / scrub | Ring diff of neighbours lost and gained, Jaccard overlap | none |
| 5 | The vote | Majority and weighted votes, probabilities, ties | Vote panel, weight curve, tie banner, exact-match demo | Weighting, Gaussian h, tie rules | Vote bars for A and B side by side, prediction flip | none |
| 6 | Choosing k | Bias versus variance | Map animating as k changes, small multiples across k, Voronoi diagram at k = 1, train / test / leave-one-out curves | k range, curve type | One curve per scaler; the best k moves | ≥ 30 training rows |
| 7 | Changing the metric | The metric shapes neighbourhoods and boundaries | Same query under L1, L2, L∞, p, Mahalanobis and cosine; unit balls; boundary small multiples; hatched tie regions | Metric grid | Metric × scaler grid. Mahalanobis gives the same predictions under any per-feature linear scaler; cosine depends on centring. | none |
| 8 | Decision boundaries | What the fitted model "looks like" | Slice view with sliders for the fixed features, PCA-plane map, scatter-plot-matrix backgrounds, note on order-k Voronoi | Slice-position sliders | Small multiples, one per scaler | none |
| 9 | Too many dimensions | The curse of dimensionality | Distance histograms as noise dimensions are added, relative-contrast curve, k-occurrence (hubness) histogram, accuracy vs noise, a feature picker for dropping features | Noise-dimension slider, feature toggles | Z-scoring gives a noise column a full vote. Unscaled, a small-range noise column barely counts. | none |
| 10 | Evaluate honestly | Proper validation | Confusion matrix, per-class precision / recall / F1, majority baseline, the k = 1 training-accuracy trap, leakage toggle, imbalance demo (bank binary 20/80), German Credit trap | Split ratio and seed, leave-one-out toggle | Accuracy per scaler; scaler fitted on all rows vs on training rows only | ≥ 2 rows per class |
| 11 | Lab | Everything together | All blocks, linked; scikit-learn snippet | Everything | Everything | none |

**The Lab's scikit-learn snippet** reproduces the current settings as a `Pipeline` (imputers and encoders in a `ColumnTransformer`, then the scaler, then `KNeighborsClassifier`).
- **Dudani and Gaussian weighting** are emitted as a weight function passed to `weights=`.
- **Tie rules scikit-learn doesn't support** get a comment saying the results will differ.

### 3.3 Narrative templating

Chapter prose lives in MDX files under `src/chapters/`. Values computed from the data are inserted from a `facts` object built from the active dataset and settings. Example:

```mdx
Before scaling, **{facts.dominantFeature.name}** supplies {pct(facts.dominantFeature.shareRaw)} of the distance.
```

Facts are computed in the worker, are pure data, and are covered by unit tests. Every chapter must read correctly on fruit, bank, iris (no categorical columns, no missing values) and penguins (a real dataset with both). §10.3 checks this.

## 4. Data

### 4.1 Default dummy dataset: fruit

Source: `references/datasets/dummy/fruit/`. The generator uses seed 7 and reproduces the CSV byte for byte. The app ships the generated CSV, not the generator.

**Shape:** 200 rows. The classes are apple, orange and pear, split 90 / 70 / 40. A binary variant (apple vs not apple) can be switched on in Chapter 0.

| Column | Role |
|---|---|
| `weight_g` | Headline x-axis, large scale |
| `diameter_cm` | Headline y-axis, small scale |
| `sweetness_brix` | Informative, 4% missing |
| `days_in_store` | Pure noise |
| `stem_length_mm` | Skewed, 5 outliers |
| `skin_colour` | Categorical, 4 levels, 4% missing |
| `fruit` | Target |

**Validated behaviour** (repeated stratified 5-fold, seed 0):

| Check | Result |
|---|---|
| Z-scoring `weight_g` + `diameter_cm` vs no scaling | +23.4 points |
| Best k | 13; k = 1 is 10.1 points lower |
| Robust vs min-max scaling | 21.5% of predictions flip |
| One-hot `skin_colour` | 17.9% of predictions flip |
| Adding 50 noise columns | −15.9 points |
| Raw equal-aspect squash | 41 : 1 |

These numbers are reference values for the facts tests, not UI copy. The UI computes its own.

### 4.2 Built-in samples

All are bundled in `public/datasets/`. Each has preset wizard choices and a licence credit on the About page.

| Sample | Source | Licence |
|---|---|---|
| Fruit (default) | Generated | none |
| Bank | Generated; synthetic, labelled as not real detection rules | none |
| Toy-2D | Generated | none |
| Iris | UCI | CC BY 4.0 |
| Palmer Penguins | palmerpenguins | CC0 |
| Wine | UCI | CC BY 4.0 |
| Breast Cancer Wisconsin | UCI | CC BY 4.0 |
| German Credit | UCI | CC BY 4.0 |

### 4.3 Pipeline

Every stage is a pure function in `src/engine/pipeline/`, run in the worker. Transforms are fitted on training rows only unless the Chapter 10 leakage toggle is on.

| Stage | Behaviour | Defaults and limits |
|---|---|---|
| Load | Papa Parse inside the worker; delimiter detection, quoted fields, header toggle | ≤ 10 MB file |
| Column wizard | Each column is auto-typed: numeric, binary, categorical, ID-like (unique per row), constant, or date/text. The user confirms the target and keeps or ignores each column. | Target defaults to the last column. 2–10 classes. A target with 11+ classes is offered "group the rarest into *other*" (up to 10), otherwise blocked. A numeric target with more than 20 distinct values warns "looks continuous". |
| Encode | Per categorical column: one-hot (default), ordinal (drag to order the levels), or drop | One-hot keeps every level (no drop-first: dropping a level makes mismatches involving it cost half as much as other mismatches, theory/05). More than 20 levels: "top 10 + other", or drop. A level seen only in the test set encodes as all zeros. |
| Transform | Optional per-column log1p | Offered only when min ≥ 0. Skew > 1 is flagged. Off by default. |
| Impute | Numeric: drop rows, mean, or median. Categorical: drop rows, most frequent, or a "missing" level. | Median and most frequent. Counts are reported. |
| Split | Stratified train/test with a visible seed and a reshuffle button | 80/20, seed 7. A class with fewer than 2 rows blocks the split, with a message. |
| Scale | Global scaler A (and optional B): none, min-max, z-score, robust, max-abs, quantile-uniform, quantile-normal, Yeo-Johnson, row L2. Optional manual weight wⱼ per feature. | Scaler A is "none" in chapters 0–2 and switches to z-score when Chapter 3 is reached, with a visible notice. The Lab and direct links use whatever the URL says. B is off. One-hot columns stay 0/1 unless "scale one-hot too" is on. A column with zero spread or zero IQR keeps scale 1 and is flagged. Parameters follow scikit-learn's defaults (StandardScaler ddof 0, QuantileTransformer n_quantiles = min(1000, n_train), RobustScaler 25–75). |
| Size guard | Above 20k rows: a seeded stratified sample of 20k, with a banner | Above 50 features after encoding: a warning and a feature picker. Above 200: blocked. |

Chapters 1–2 run unscaled on purpose: the per-feature distance bars there show one feature dominating, which sets up Chapter 3.

### 4.4 `PreparedData` contract

The interface sketch below is part of the spec, not implementation.

```ts
interface FeatureMeta {
  id: string;            // stable, e.g. "skin_colour=red"
  name: string;          // display name
  source: string;        // original CSV column
  kind: 'numeric' | 'onehot' | 'ordinal';
  transform: 'none' | 'log1p';
  scale: { method: ScalerId; center: number; scale: number; flagged?: 'zero-spread' } | { method: 'quantile' | 'yeo-johnson'; params: unknown };
  weight: number;        // manual weight, default 1
}
interface PreparedData {
  version: number;                       // bumps on any upstream change
  features: FeatureMeta[];
  classes: string[];                     // ordered as np.unique would order them
  train: { X: Float64Array; y: Int32Array; rowIds: Int32Array; raw: RawRow[] };
  test:  { X: Float64Array; y: Int32Array; rowIds: Int32Array; raw: RawRow[] };
  n: number; d: number;
  notes: PipelineNote[];                 // imputed counts, dropped rows, flags, subsampling
}
```

Class order follows scikit-learn: numeric sort if every label parses as a number, otherwise lexicographic. This is what makes vote-tie results match (theory/04).

## 5. Query points

- **Default:** a seeded random test row.
- **Alternatives:**
  - any row, picked from a table
  - raw values typed into a form, which go through the same fitted pipeline
  - a click on the plane view, which sets the two axis features and keeps the rest
- The query's raw values, processed values and source (test row #, custom, or click) are always shown.

## 6. Engine

Plain TypeScript in `src/engine/` with no DOM, hosted in one worker (`src/worker/`) via Comlink. Large arrays cross the thread boundary as Transferables.

### 6.1 Metrics

Each metric is a specialised kernel: L1, L2 (squared for ranking, square-rooted for display), L∞, Minkowski p ∈ [0.25, 10], Mahalanobis, and cosine. Per-feature weights enter as wⱼ|Δⱼ|ᵖ.
- **p < 1** carries a "not a metric" flag and shows the counterexample from theory/02.
- **Mahalanobis:** the covariance comes from the training rows. The data is whitened once by the Cholesky factor of Σ⁻¹, after which L2 applies. A singular Σ gets a ridge of 1e-6 · trace(Σ)/d, with a flag.
- **Cosine** is defined as 1 − cos. A zero vector gets distance 1, with a flag.

### 6.2 Weighting

| Scheme | Weight |
|---|---|
| Uniform | 1 |
| Inverse distance | 1/d |
| Inverse square | 1/d² |
| Dudani | (d_k − d_i)/(d_k − d_1), or 1 when d_k = d_1 |
| Gaussian | exp(−(d/h)²/2), with h = d_k by default or a fixed h |

**Exact matches** follow scikit-learn: if any neighbour is at distance 0, the zero-distance neighbours share all the weight. This applies to 1/d and 1/d². For Dudani and Gaussian, d = 0 is finite and needs no special rule.

### 6.3 Ties

Ties are always detected and reported. Distances within a relative 1e-12 of each other count as equal.

| Tie | Default (matches scikit-learn) | Opt-in alternatives (labelled "not sklearn") |
|---|---|---|
| Rank k vs k+1 | Earlier row in training order | Include all tied rows; seeded random |
| Vote | Smallest class index | Nearest neighbour's class; fall back to k−1; seeded random |

### 6.4 `PredictionTrace`

Every explanation and every step-by-step visual reads from this record.

```ts
interface PredictionTrace {
  query: { raw: RawRow; x: Float64Array; source: QuerySource };
  config: KnnConfig;                       // k, metric, p, weighting, h, tie policies, scaler id
  distances: Float64Array;                 // to every training row, in training order
  order: Int32Array;                       // training indices sorted by distance (stable)
  neighbours: Array<{ idx: number; rowId: number; d: number; w: number; label: number;
                      terms: Float64Array; shares: Float64Array }>; // wⱼ|Δⱼ|ᵖ and their shares
  radius: number;                          // d_k
  rankTie?: { candidates: number[]; chosen: number[]; rule: TieRule };
  scores: Float64Array;                    // S_c per class
  proba: Float64Array;
  voteTie?: { classes: number[]; rule: TieRule };
  predicted: number;
  flags: EngineFlag[];                     // p<1, exact match, ridge applied, etc.
}
```

For L∞, `terms` holds |Δⱼ| and the share is 1 for the arg-max feature (all tied maxima are reported). For Mahalanobis and cosine, the terms are in whitened or product form and labelled as such.

### 6.5 Caching and batch jobs

**Neighbour cache.** Each entry holds the top-K_max neighbours (K_max = 50; indices and distances) for every evaluation row and map cell. Its key is `(data.version, scaler, metric, p, weights)`.
- A change to k, weighting, h or tie rules only re-runs the vote over the cached lists.
- Any other change invalidates the entry.
- Scaler A and B each keep their own entry.

**Slice map.** Features outside the two axes are held at the query's values. For each training row i, their contribution cᵢ is computed once (a sum for Minkowski, a max for L∞), so each grid cell only adds the 2-D part.
- The grid renders at 40×40 first, then refines to 120×120. A "high resolution" option goes up to 240.
- Each cell stores its class, its vote margin, and a tie flag (rank or vote).

**PCA plane.** PCA (ml-pca) is fitted on the processed training rows. The map is an exact slice through the query along the PC1–PC2 plane: each grid point is inverse-projected with the other components set to the query's. Training points are drawn as their projections onto the plane, with the caption "shadows of points that may lie off this plane". The true neighbour links are drawn regardless of how far apart they look.

**Scatter-plot matrix.** Up to 8 features: the user's choice, or by default the 8 with the largest mean |Δⱼ|ᵖ share. Cell backgrounds use 40×40 slice maps, computed lazily when the matrix is open.

**Dimensions chapter.**
- Seeded N(0, 1) noise columns are appended to the processed matrix, 0–100 of them.
- For each noise level it measures, on 200 sampled queries:
  - relative contrast (d_max − d_min)/d_min
  - the distance histogram
  - the k-occurrence distribution and its skew (hubness)
  - test accuracy

**Evaluation.**
- Test-set predictions, the confusion matrix, per-class precision / recall / F1, macro and weighted averages, accuracy, and the majority-class baseline.
- Validation curves for k = 1–50: training accuracy (each row counts as its own neighbour), test accuracy, and leave-one-out.
- Leave-one-out uses one neighbour search per row with that row excluded, then votes for every k. It runs on all rows up to 5k and on a seeded stratified 5k sample above that, with a banner.
- Leave-one-out fits the scaler once on all training rows instead of once per left-out row, and labels the result "approximate (scaler fitted once)". theory/10 measured the difference at no more than 1 prediction.

**Scaler comparison.** For the current query: the A and B neighbour sets, their Jaccard overlap, the lost and gained rows, the predictions and probabilities under each. For evaluation: accuracy under A and B.

### 6.6 Jobs

- Every request carries a generation number, and results from older generations are dropped.
- Long jobs run in chunks, report progress, and check a cancel flag between chunks.
- If the worker crashes, it is restarted and the dataset is re-sent.

### 6.7 Numerical rules

- All maths in float64.
- Sorts are stable, so training order breaks rank ties as in scikit-learn.
- Squared L2 is used for ranking, and the square root only when values are shown.

### 6.8 Performance budgets

Measured at 20k training rows × 50 features on a 2023-class laptop.

| Action | Budget |
|---|---|
| Explain one query (full trace) | < 50 ms |
| First coarse map | < 150 ms |
| Full 120² map | < 1 s |
| Re-vote the map after k / weighting / tie changes | < 16 ms |
| Test-set evaluation (4k × 16k × 50) | < 3 s, with progress |
| Leave-one-out curve on a 5k sample | < 5 s, with progress |
| First load | ≤ 250 KB gzipped JS; KaTeX fonts and chapters lazy-loaded |

## 7. Visual system

### 7.1 Blocks (`src/viz/`)

| Block | Rendering | Notes |
|---|---|---|
| PlaneView | Canvas 2D for the map and points; SVG on top | The SVG layer draws the query, neighbour links (thickness ∝ wᵢ), the radius-d_k ball in the current metric's shape (a superellipse, or an ellipse for Mahalanobis), Voronoi edges (d3-delaunay, L2 and k = 1 only), and tie hatching. Modes: slice, PCA plane, raw at equal aspect. Animates the rubber-sheet morph between raw and scaled. Raw-unit axes with secondary scaled ticks. Click sets the query. |
| DistanceBars | SVG | Stacked wⱼ\|Δⱼ\|ᵖ per neighbour or one pair; raw vs scaled, or A vs B |
| DistanceLadder | Canvas + SVG | All n distances; animated fill, sort and cut at k; tie marks |
| VotePanel | SVG | Bars grow neighbour by neighbour; probabilities; weight curve; tie banner |
| FormulaPanel | KaTeX with `\htmlData` (trust limited to that command) | Symbolic and plugged-in forms, with terms linked to the visuals |
| Curves | SVG with d3-scale | Validation curves, accuracy per scaler, the dimensions-chapter plots |
| Matrices | Canvas + SVG | Scatter-plot matrix, confusion matrix |
| NeighbourTable | HTML table | Rank, row #, class, distance, weight, contribution of the top feature |
| PlaybackBar | HTML | Step back / forward, play, scrub; the button text names the next step |
| ImpactStrip | HTML + SVG | A vs B: Jaccard, lost / gained, prediction A → B, accuracy A vs B |

### 7.2 Linked highlighting

A single store holds `{ rowId?, featureId?, classId?, termId? }`. Every block subscribes and publishes, so hovering a formula term highlights its bar and its geometric element, and the reverse.

### 7.3 Colour and marks

**Classes.**
- Up to 7 classes: the 7 Okabe–Ito hues. Black is reserved for the query and for outlines.
- 8–10 classes: Tableau 10.
- Every class also gets its own shape (circle, triangle, square, diamond, …), so colour is never the only cue.

**Features** use a separate muted palette and are always labelled. **Ties** are grey 45° hatching. Light and dark themes come from CSS variables. Hex values are in `references/visual-exemplars/design-patterns.md`.

### 7.4 Motion and accessibility

- Transitions are at most 600 ms. With `prefers-reduced-motion` on, steps jump to their end state.
- Every control is keyboard operable, and focus is always visible.
- An ARIA live region narrates each step, e.g. "Neighbour 3: row 42, apple, distance 0.81, weight 1.23".
- Every chart has a text summary built from the same facts templates as the narrative.
- Contrast meets WCAG AA.

### 7.5 Responsive layout

| Width | Layout |
|---|---|
| ≥ 1100 px | Two columns, stage in view while scrolling |
| 700–1100 px | Narrower stage |
| < 700 px | Stacked, stage pinned to the top and collapsible |

## 8. App shell and state

- **No router.** The URL hash holds the chapter and settings, e.g. `#ch=4&ds=fruit&k=5&m=l2&p=2&w=uniform&sa=z&sb=none&oh=0&seed=7&fx=weight_g&fy=diameter_cm&view=slice`.
  - `ds` is a built-in sample id or `upload`. User data is never put in the URL.
  - For `ds=upload`, opening the link restores the settings, falls back to the dummy data, and shows "load your file again".
- **Stores:** Zustand for settings, the highlight state and the active dataset descriptor.
- **Chapters** are lazy-loaded MDX modules, compiled with `@mdx-js/rollup`.
- **UI components:** the shadcn/Radix components kept from the template (button, card, badge, slider, select, switch, tabs, tooltip, alert). Others (table, toggle-group, progress, dialog, collapsible) are added with the shadcn CLI only when a chapter uses them.

## 9. Repo, build, deploy

### 9.1 Layout

```
src/engine/          pure TS: pipeline/, metrics/, knn/, ties/, eval/, maps/, pca/, curse/, facts/
src/worker/          Comlink API, caches, job runner
src/state/           Zustand stores, hash sync
src/viz/             the blocks in §7.1
src/chapters/        00-data.mdx … 11-lab.mdx
src/ui/              kept shadcn components
public/datasets/     sample CSVs, LICENSES.md
tools/fixtures/      Python (scikit-learn 1.9.x) fixture generator, requirements.txt
tests/e2e/           Playwright
references/          research (not shipped in the build)
docs/superpowers/    specs/, plans/
```

### 9.2 Clean-up of the Lovable template (at the start of M1)

- **Delete:**
  - the Lovable pages and components
  - `bun.lock` and `bun.lockb`
  - `lovable-tagger`
  - every unused shadcn component and dependency (full list in `references/visual-exemplars/tech-stack.md` §10)
  - `react-router-dom`
  - `react-katex` (replaced by `katex` directly)
- **Fix:**
  - `index.html` title and meta tags
  - the header's GitHub link
- **Turn on** `strict` in TypeScript.

### 9.3 Toolchain

Current stable majors at M1 start: Vite, React 19, and TypeScript. Exact versions are pinned in `package-lock.json`. If `typescript-eslint` doesn't yet support the newest TypeScript major, use the latest one it does support.

**Runtime dependencies:**
- katex
- d3-scale, d3-delaunay, d3-contour, d3-scale-chromatic
- papaparse
- comlink
- zustand
- ml-pca (lazy-loaded)
- @mdx-js/rollup (build only)

### 9.4 Deploy

- `vite.config.ts`: `base: '/knn-explorer/'`.
- `.github/workflows/deploy.yml` on Node 24 LTS runs: `npm ci`, lint, typecheck, unit tests, build, Playwright smoke test against `vite preview` at the base path, then `actions/deploy-pages`.
- `index.html` carries `<meta http-equiv="Content-Security-Policy" content="default-src 'self'; connect-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; worker-src 'self' blob:">`.

## 10. Testing

### 10.1 Engine against scikit-learn

`tools/fixtures/generate.py` writes JSON fixtures that are committed to the repo. They cover:
- **Datasets:** fruit, iris, wine, penguins (with imputation and one-hot), and designed tie cases (duplicates, 45° pairs under L1, shared coordinates under L∞, a balanced even-k vote).
- **For each:** every scaler; metrics L1, L2, L∞, p ∈ {0.5, 3} and Mahalanobis; weighting uniform and 1/d; k ∈ {1, 4, 5, 15}.
- **Recorded:** neighbour indices and distances, predictions, probabilities, scaled matrices, and leave-one-out predictions.

Vitest asserts indices exactly and floating-point values within 1e-9. Dudani, Gaussian and the non-scikit-learn tie rules are tested against hand-computed cases taken from theory/04.

### 10.2 Invariance property tests

These mirror theory/10, over random data:
- Centring any feature changes no Minkowski prediction.
- Multiplying every feature by the same c > 0 changes no neighbour set and no 1/d vote proportion.
- Mahalanobis predictions are unchanged under every per-feature linear scaler.
- Scaling feature j by sⱼ equals a weighted Minkowski distance with wⱼ = sⱼ^(−p).

### 10.3 Facts and narrative

- **Snapshot tests** of the facts for fruit, bank, iris and penguins.
- **Rendering test:** every chapter's MDX renders without missing-fact errors on each of those datasets.
- **Fallback test:** when a needed property is missing (e.g. iris has no categorical column), the fallback note appears.

### 10.4 Components

Vitest with Testing Library covers the highlight store, the wizard's auto-typing and decisions, hash sync, and job cancellation and generation handling.

### 10.5 End to end

Playwright runs against the built site at its `/knn-explorer/` path:
1. Chapters 0–5 load on fruit.
2. Upload `tests/e2e/fixtures/penguins.csv` and go through the wizard (one-hot `island` and `sex`, impute median and most frequent).
3. Check that the displayed predictions for 5 test rows equal the fixtures.
4. Switch scaler A/B and check the impact strip appears.
5. Run axe-core on each chapter.

CI runs all of the above on every push. Deploy happens only if everything passes.

## 11. Error handling

User-facing errors appear as a non-blocking banner in plain language, with the details in a disclosure.

| Case | Behaviour |
|---|---|
| CSV parse error | Line number and a sample of the offending line |
| Non-numeric text in a numeric column | Treated as missing; count shown |
| Rows with too many or too few fields | Reported; the user chooses to skip those rows or cancel |
| Empty file, or only a header | Blocked, with a message |
| Single-class target | Blocked |
| 11+ classes | "Group rarest into *other*" or blocked |
| A class with fewer than 2 rows | Split blocked; option to drop that class |
| More than 50 features after encoding | Warning and feature picker; more than 200 blocked |
| More than 20k rows | Stratified 20k sample with a banner |
| Singular covariance for Mahalanobis | Ridge applied and flagged |
| p < 1 | Allowed and flagged |
| Zero-spread or zero-IQR column | Scale 1, flagged |
| Worker crash | Restart, re-send the dataset, banner "recovered" |
| Stale results | Dropped silently by generation number |

## 12. Milestones

Each milestone gets its own implementation plan (writing-plans), ends deployed to GitHub Pages, and is reviewed by the owner before the next one starts.

| Milestone | Contents | Done when |
|---|---|---|
| **M1: foundation and chapters 0–5** | Template clean-up and toolchain; CI and deploy fixed; fixtures generator; engine (pipeline, scalers, metrics, k-NN, weighting, ties, trace, caching); worker; stores and hash sync; PlaneView (slice and raw-equal-aspect modes, rubber-sheet morph), DistanceBars, DistanceLadder, VotePanel, FormulaPanel, NeighbourTable, PlaybackBar, ImpactStrip; chapters 0–5; built-in samples | Site live at the target URL; §10.1–10.5 pass for chapters 0–5; budgets in §6.8 met for query trace and maps |
| **M2: chapters 6–8** | Validation curves, metric grid, Mahalanobis and cosine visuals, slice-position sliders, PCA plane, scatter-plot matrix, small multiples | Chapters 6–8 pass the tests; budgets met for maps and leave-one-out |
| **M3: chapters 9–11** | Noise dimensions and hubness, evaluation views, leakage toggle, Lab, scikit-learn snippet, About and licences page | All success criteria in §1.4 met |

## 13. Decisions log

| # | Decision | Why | Reference |
|---|---|---|---|
| D1 | Rebuild in place, don't patch Lovable | The Lovable code assumes 2-D and binary labels throughout; the build and deploy are broken | references/repo-audit.md |
| D2 | Canvas + SVG + d3 modules; no Plotly | Plotly adds 400–1,476 KB gzipped and fights custom overlays and linked highlighting | tech-stack.md §2 |
| D3 | Default tie rules match scikit-learn; others opt-in | Parity with the tool people actually use; ties made visible rather than hidden | theory/04 |
| D4 | One-hot keeps every level | Dropping a level makes mismatches involving it cost half as much as other mismatches | theory/05 |
| D5 | One-hot columns not scaled by default | Z-scoring inflates the cost of rare-category mismatches (about 10× at 1% frequency) | theory/05, theory/10 |
| D6 | Scaler fitted on training rows only; leakage demo opt-in | Correct practice; the measured leakage is small but nonzero | theory/10 |
| D7 | PCA map is a slice along the PC1–PC2 plane through the query | Exact for points on the plane; avoids drawing a fake boundary on a projection | tech-stack.md §6 |
| D8 | No t-SNE or UMAP | Can't support an honest decision map; adds weight | tech-stack.md §6 |
| D9 | Classes capped at 10 | Colour-blind-safe palettes run out; group the rest as "other" | design-patterns.md |
| D10 | Fruit as default, bank as an extra | Its scale squash can be drawn; intuitive units; more errors to show; avoids implying real AML rules | datasets/dummy/README.md |
| D11 | Brute-force search only | n ≤ 20k fits the budgets; trees add complexity without teaching value here | theory/09 |
| D12 | Prose in MDX, with facts computed from data | The owner can edit wording without touching code; text follows the active dataset | §3.3 |

## 14. Risks and open items

| Risk | Mitigation |
|---|---|
| Leave-one-out at 20k rows is too slow for interaction | 5k stratified sample, run on request with progress |
| Neighbour-cache memory at 120² cells × 50 neighbours × 2 scalers | About 17 MB, acceptable. The 240² high-resolution mode keeps only scaler A's map cache. |
| The dummy data's best-k window depends on the seed | Ship the validated seed (7). The facts compute the best k live, so the text stays true. |
| Prose that reads well on fruit but awkwardly on other data | §10.3 renders every chapter on four datasets; copy review per milestone |
| KaTeX `trust` setting | Allow only `\htmlData` |
| A newer TypeScript major vs lint tooling | Version fallback rule in §9.3 |
| Fixtures drift with scikit-learn releases | Pin scikit-learn 1.9.x in `tools/fixtures/requirements.txt` |
