# Existing interactive explainers

I opened every page listed here on 2026-10-27. Screenshots in `screenshots/` were taken at 1280 px wide with headless Chromium (Playwright).

Our goals, used for the "lacks" lines:
- **U**: upload your own data
- **S**: step through the algorithm's inner workings
- **M**: math linked to geometry
- **C**: compare k, metric/p, weighting and scaling

## Summary

| # | Tool | U | S | M | C | Best idea to borrow |
|---|---|---|---|---|---|---|
| 1 | CS231n kNN demo | – | – | – | k, L1/L2 | Ties shown as white regions |
| 2 | MLU-Explain: Bias–Variance (KNN widget) | – | – | – | k | Hover a point: its neighbour lines on the left, decision map on the right |
| 3 | Kanaries KNN playground | – | – | – | k, L1/L2, weights | Live neighbour vote table (rank, class, distance, weight) |
| 4 | Cornell CS4780 KD-tree demo | – | partial | – | – | Shading shows how many distance evaluations a region needs |
| 5 | TensorFlow Playground | – | step button | – | many | Left-to-right pipeline layout; play/step/reset bar |
| 6 | Setosa: PCA Explained Visually | – | – | – | – | Drag points in the original space, watch the PC space update |
| 7 | Seeing Theory: Regression | – | – | partial | – | Click a table column to explain that statistic |
| 8 | Naftali Harris: Visualizing k-means | – | yes | – | – | One button labelled with the *next* step |
| 9 | Distill: How to Use t-SNE Effectively | – | play/step | – | perplexity | Small multiples of one parameter; honest projection caveats |
| 10 | Distill: Why Momentum Really Works | – | – | yes | α, β | Hero figure with two sliders, annotated geometry |
| 11 | Equations Explained Colorfully | – | – | yes | – | Colour-coded equation terms linked to prose on hover |
| 12 | Galles / Lucet Data Structure Visualizations | – | yes | – | – | Standard playback bar: skip/step back, play/pause, step/skip forward, speed |

None of these accept an uploaded dataset with encoding and scaling, and none link the distance formula to the picture. That combination is the gap our app fills.

---

### 1. Stanford CS231n k-Nearest Neighbors demo
- **URL:** http://vision.stanford.edu/teaching/cs231n-demos/knn/
- **Shows:** a 2-D plane coloured by the class KNN would predict at every location. Locations where the vote ties are white.
- **Interaction:** controls for Metric (L1, L2), Num Neighbors (K) 1–7, Num classes 2–5 and Num points 20–60. Drag points to move them.
- **Borrow:** render ties in a neutral colour, not by an arbitrary tie-break. It is an honest view of even k and multiclass votes. The minimal control set (metric, k, classes, points) is enough for a first screen.
- **Lacks:** U, S, M. Only L1/L2, no weighting, no scaling, and no view of a single query's neighbours.
- **Screenshot:** none. WebFetch reached the page, but headless Chromium got "upstream request failed" twice, so I skipped it.

### 2. MLU-Explain: The Bias Variance Tradeoff (KNN section)
- **URL:** https://mlu-explain.github.io/bias-variance/. Index: https://mlu-explain.github.io/
- **KNN article?** No. The index lists 14 articles and none is about KNN. KNN appears only as a widget inside Bias–Variance. The Cross-Validation article (https://mlu-explain.github.io/cross-validation/) returned only metadata to WebFetch, so I can't describe it.
- **Shows:** two side-by-side panels. The left has training points with a dashed true boundary; the right has hexagon-binned decision regions, with saturation encoding vote confidence. A K slider ("K: 3") and a "Randomize Data" button sit above.
- **Interaction:** hover a point in the left panel to draw grey lines to its K neighbours, while the right panel shows its classification. The surrounding text walks from small K ("islands and jagged boundaries", high variance) to large K (majority vote, high bias). The page credits Scrollama and KaTeX.
- **Borrow:**
  - Two linked panels: one concrete instance (a query and its neighbour lines) next to the abstraction (the decision map).
  - Confidence as saturation.
  - Overlaying the true boundary on toy data, so overfitting is visible.
- **Lacks:** U, S, M (no formulas in the KNN section), C beyond k.
- **Screenshot:** `screenshots/mlu-bias-variance-knn.png`

### 3. Kanaries "KNN visualization" playground
- **URL:** https://ml.kanaries.net/playground/knn
- **Shows:** a cell-grid decision map, a draggable diamond query point with dashed lines to its 5 nearest neighbours, and stat tiles (Prediction, Training points, Nearest distance). Below the map is a "Current neighbor vote" table with columns Rank, Class, Distance and Vote weight.
- **Interaction:**
  - Pick a dataset (blobs, moons, xor).
  - Set k, the distance metric (Euclidean L2 or Manhattan L1) and voting weights (uniform or inverse distance).
  - Click the map to add points to Class 0 or 1. Undo and Clear are available.
- **Other notes:** the page says it runs in the browser ("No upload or sign-in"), using the `@kanaries/ml` KNearestNeighbors model. It notes that Manhattan distance "creates diamond-shaped neighborhoods instead of Euclidean circles", but never draws the diamond.
- **Borrow:** the vote table next to the map (our step 5), the query point as a draggable glyph distinct from training points, and a stat tile row above the plot.
- **Lacks:** U (it explicitly has none), S, M, Minkowski p, scaling, multiclass.
- **Screenshot:** `screenshots/kanaries-knn-vote-table.png`

### 4. Cornell CS4780: KD-Tree demo
- **URL:** https://www.cs.cornell.edu/courses/cs4780/2022sp/notes/kdtree.html
- **Shows:** a 1-NN classifier. The left panel has the data, the k-d tree's splitting lines and the decision boundary; the right panel shows the tree. Shading counts how many distance evaluations each region needs, and greyed tree parts are pruned work.
- **Interaction:** drag points. Double-click to add, remove or relabel. Hover to see how a test point traverses the tree.
- **Borrow:** a "cost of a query" view. That's useful if we add an "under the hood: brute force vs k-d tree" panel.
- **Lacks:** U, M, C. It is not a stepwise animation of the search.
- **Screenshot:** none. Same Chromium proxy failure as the CS231n demo.

### 5. TensorFlow Playground
- **URL:** https://playground.tensorflow.org/
- **Shows:** the whole pipeline left to right in four columns: DATA (dataset thumbnails, train/test ratio, noise, batch size, REGENERATE), FEATURES (x₁, x₂, x₁², …, each a tiny heatmap), hidden layers (+/− neurons and layers), and OUTPUT (test/training loss and a decision heatmap with a −1…1 orange–blue legend). The OUTPUT column has "Show test data" and "Discretize output" checkboxes.
- **Interaction:** a top bar with reset, play and step buttons, an epoch counter, and dropdowns for learning rate, activation, regularization, its rate, and problem type.
- **Borrow:**
  - Pipeline-as-columns layout: Data → Preprocess (scale/encode) → Distance & neighbours → Vote → Output.
  - Every intermediate stage drawn as a small multiple.
  - A global run bar.
  - Toggles to show test data and to discretize the output, which for us means the hard class versus the vote share.
- **Lacks:** U, M. Its step control steps epochs, not the steps of a single prediction.
- **Screenshot:** `screenshots/tf-playground.png`

### 6. Setosa "Explained Visually": Principal Component Analysis
- **URL:** https://setosa.io/ev/principal-component-analysis/. Series index: https://setosa.io/ev/ ("an experiment in making hard ideas intuitive, inspired by Bret Victor's Explorable Explanations")
- **Shows:** an "original data set" panel next to an "output from PCA" panel, with PC axes drawn as coloured arrows. A 3-D cloud can be rotated, with a "Show PCA" button. A 17-D food dataset is projected to PC1 and PC1–PC2.
- **Interaction:** drag points in the original space and watch the PC axes and projected points update.
- **Borrow:** use this exact pairing for our >2-feature projection view: chosen feature pair on one side, PCA projection on the other, colours consistent across both. The "rotate until you find a good 2-D view" idea can explain why a projection loses distance information.
- **Lacks:** not about classification. No U, S or C.
- **Screenshot:** `screenshots/setosa-pca.png`

### 7. Seeing Theory: Regression Analysis
- **URL:** https://seeing-theory.brown.edu/regression-analysis/index.html. Home: https://seeing-theory.brown.edu/ (banner: "This site is archived for reference")
- **Shows:** chapter tiles with circular icons. Each section has an interactive (drag points to change the OLS line or ANOVA), then an interactive table, then term definitions as display equations.
- **Interaction:** drag points. Clicking a column of the regression or ANOVA table explains that parameter, and clicking a correlation-matrix cell shows that pair's relationship.
- **Borrow:** "click a number to learn what it is" for our distance table and vote table. A cell click could open the formula with that cell's values substituted in. The calm, consistent section rhythm is also worth copying.
- **Lacks:** not KNN. No U, S or C.
- **Screenshot:** `screenshots/seeing-theory-regression.png` (chapter index)

### 8. Naftali Harris: Visualizing K-Means Clustering
- **URL:** https://www.naftaliharris.com/blog/visualizing-k-means-clustering/
- **Shows:** points coloured by cluster over a shaded Voronoi diagram of the centroids. Datasets include Uniform Points, Gaussian Mixture, Smiley Face, Density Bars, Packed Circles, Pimpled Smiley and DBSCAN Rings.
- **Interaction:** choose an initialization (I'll Choose, Randomly, Farthest Point, …) and a dataset, then "Add Centroid". A single action button cycles through "GO!", "Update Centroids" and "Reassign Points", each label naming what the next click will do.
- **Borrow:** the self-labelling next-step button. For KNN: "Compute distances", then "Sort", "Take k nearest", "Vote", "Predict". It is the simplest stepper there is and needs no explanation. The Voronoi shading is a direct analogue of KNN at k=1 under L2.
- **Lacks:** k-means, not KNN. No U, M or C. No step back.
- **Screenshot:** `screenshots/naftaliharris-kmeans.png` (after GO → Update → Reassign)

### 9. Distill: How to Use t-SNE Effectively
- **URL:** https://distill.pub/2016/misread-tsne/
- **Shows:** a large live embedding plus an 18-thumbnail dataset picker, play/pause and restart buttons, a "Step" counter, sliders (Points Per Side, Perplexity, Epsilon), and a caption per dataset. Further down are small multiples of each dataset at several perplexities.
- **Key caveats (quoted):** "you cannot see relative sizes of clusters in a t-SNE plot". "distances between well-separated clusters in a t-SNE plot may mean nothing". Random noise can look clustered at low perplexity.
- **Borrow:** small multiples across one parameter: k = 1, 5, 15, 51, or p = 1, 2, ∞. Also the thumbnail dataset picker. If we offer t-SNE or UMAP projections, put these caveats next to the view.
- **Lacks:** not KNN. No U.
- **Screenshot:** `screenshots/distill-misread-tsne.png`

### 10. Distill: Why Momentum Really Works (Goh, 2017)
- **URL:** https://distill.pub/2017/momentum/
- **Shows:** a full-width hero of a loss landscape with the optimisation path, annotated "Starting Point", "Optimum" and "Solution". Below it are two labelled sliders ("Step-size α = 0.02", "Momentum β = 0.99") and a short paragraph. Equations in the body sit next to figures that illustrate them, such as a convergence-rate formula beside a plot of its regions.
- **Borrow:** slider labels written as math ("Momentum β = 0.99"), so the control is the symbol in the equation. Annotate geometry directly instead of using a legend. Lead with one hero interactive that poses the question.
- **Lacks:** not KNN. No U or S.
- **Screenshot:** `screenshots/distill-momentum.png`

### 11. Equations Explained Colorfully (Piotr Migdał)
- **URL:** https://p.migdal.pl/equations-explained-colorfully/. Source: https://github.com/stared/equations-explained-colorfully (MIT)
- **Shows:** about ten famous equations, including the Discrete Fourier Transform and Bayes' theorem. Each term is coloured, and the same colour marks the matching phrase in a one-sentence explanation. Colour modes: Vibrant, Accessible, High Contrast, No color. An editor shows the markdown and LaTeX source (`\mark[class]{…}`).
- **Interaction:** hover a coloured term to reveal its definition. Clicking locks it (per the README).
- **Tech:** KaTeX for rendering and Vue 3 for the app. It exports to HTML, LaTeX, Beamer and Typst. The README credits Stuart Riffle's colour-coded DFT and BetterExplained's colorized equations as inspiration.
- **Borrow:** the whole pattern for our distance and vote formulas, including the Accessible and High-Contrast modes. The `\mark` syntax maps onto KaTeX `\htmlClass` (see tech-stack.md).
- **Lacks:** static equations only. Terms aren't bound to live values or to a plot.
- **Screenshot:** `screenshots/equations-explained-colorfully.png`

### 12. Data Structure Visualizations (David Galles, modified by Yves Lucet)
- **URL:** https://cmps-people.ok.ubc.ca/ylucet/DS/about.html. Original: http://www.cs.usfca.edu/~galles/visualization
- **Shows:** canvas animations of data structures and algorithms. Algorithm-specific controls sit at the top and an identical general animation bar sits at the bottom.
- **Playback controls:**
  - Skip Back: undo the current or previous command.
  - Step Back: only when paused.
  - Play/Pause.
  - Step Forward: only when paused.
  - Skip Forward: jump to the end.
  - Animation Speed slider: the speed is saved in a cookie.
- **Borrow:** keep algorithm-specific controls separate from a fixed, always-identical playback bar. Allow step back as well as forward.
- **Lacks:** not ML. No U, M or C.
- **Screenshot:** `screenshots/galles-lucet-algo-vis.png`

---

## Also opened (minor)
- **UMass CS335 "KNN and Decision Trees" notebook export:** https://people.cs.umass.edu/~sheldon/teaching/cs335/lec/11-demo.html. Static. Contour plots of Euclidean, Manhattan and Chebyshev distance from the origin are, in effect, unit-ball pictures, with Chebyshev described as Minkowski p → ∞. It also shows KNN predictions converging to a true boundary as n grows from 10 to 10,000. Worth copying as a static "distance contour" panel.
- **Philippe Fournier-Viger K-NN demo:** https://philippe-fournier-viger.com/tools/KNN.php. Training data is typed into a textbox (first line is the attribute names, space-separated, numeric or categorical), the target is marked "?" in the query, and you press "Run KNN". It is the only one I found that accepts user data, and the UI is plain text.
- **Jared Wilber (MLU author) Observable notebooks:** https://observablehq.com/@jwilber/k-nearest-neighbors-knn-decision-boundary, …/live-k-nearest-neighbors-classification and …/n-nearest-voronoi-neighbors. WebFetch returned only titles and metadata, and Chromium hit a "Vercel Security Checkpoint" bot check, so I couldn't assess them. No screenshot.

## Blocked / not reachable
- **R2D3, A Visual Introduction to Machine Learning** (http://www.r2d3.us/visual-intro-to-machine-learning-part-1/): WebFetch returned `robots_disallowed`. Not opened, no screenshot.
- **Nature Methods, Wong 2011 "Color blindness"** (palette reference): `client_error`.
- **Observable notebooks:** bot check (above).
- **CS231n and Cornell demos:** reachable by WebFetch, but screenshots failed through the proxy.
