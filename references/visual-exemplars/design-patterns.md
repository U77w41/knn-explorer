# Design patterns for the KNN explorer

Each principle has its source (fetched 2026-10-27) and a concrete **→ KNN app** line.

## 1. Explorable explanations
**Source:** Bret Victor, "Explorable Explanations" (10 Mar 2011): https://worrydream.com/ExplorableExplanations/

Three ideas:
- **Reactive documents:** numbers in the text are live parameters the reader can adjust. "It's like a spreadsheet without the spreadsheet."
- **Explorable examples:** the reader manipulates the system itself, but "the author holds up their end of the conversation".
- **Contextual information:** lookups that don't break the reading flow.

The essay's goal is to turn text from "information to be consumed" into "an environment to think in".

**→ KNN app:**
- Write the explanation with live tokens, e.g. "Using the **5** nearest neighbours under **Minkowski p = 2**, **3** of **5** vote *Gentoo*". Each bold value is draggable or clickable and drives the same state as the controls.
- The reader's uploaded data is the example, not a canned one.
- Hovering a term such as "standardise" or "one-hot" opens a popover with that column's before and after values (contextual information).

## 2. Up and down the ladder of abstraction
**Source:** Bret Victor, "Up and Down the Ladder of Abstraction" (Oct 2011): https://worrydream.com/LadderOfAbstraction/

The recipe is to make a parameter controllable, step **up** by showing the system across all values of that parameter, and step **down** by picking one value from the abstraction to see why. Insight comes from "the transitions between them". Victor overlays trajectories for one parameter in a single space, and uses separate panels (small multiples) when the environment changes.

**→ KNN app:** three rungs, each clickable into the one below.

| Rung | View | Abstracts over |
|---|---|---|
| Concrete | One query point, its k neighbours, the distance table, the vote bars | nothing |
| Up 1 | Decision map (every pixel is a query) | query position |
| Up 2 | Validation accuracy against k (and against p), one curve per weighting | k, p |

- Click a pixel on the map to set the query (step down from rung 2).
- Click a point on the accuracy-against-k curve to set k (step down from rung 3).
- Overlay boundaries for several k in one plot as thin contour lines. Use small multiples only when the data or preprocessing changes, e.g. raw against standardised.

## 3. Linked views and brushing
**Source:** "Brushing and linking", Wikipedia: https://en.wikipedia.org/wiki/Brushing_and_linking. It is "the connection of two or more views of the same data". Becker and Cleveland introduced brushing ("Brushing Scatterplots", *Technometrics*, 1987). Text–visual linking is tested in Zhi, Ottley & Metoyer 2019 (§5): linking "significantly increased user engagement".

**→ KNN app:** one shared `highlight` state (row id, feature j or class c) that every view reads and writes.
- **Rows:** hovering a row in the data table, a scatter point, a neighbour-table row or a projection point highlights the same training row everywhere.
- **Features:** hovering feature j highlights its term in the distance equation, its axis in the plot and its column in the table.
- **Classes:** hovering class c highlights its bar in the vote chart, its points and its region.
- **Brushing:** dragging a box in the projection view filters the data table.
- Keep this state outside React render-heavy trees, in a small store, so the canvas layers can subscribe without re-rendering the whole tree.

## 4. Colour-coded equations linked to the picture
**Sources:**
- Stuart Riffle's one-sentence colour-coded DFT. The R-bloggers repost shows it only as an image: https://www.r-bloggers.com/2014/01/the-fourier-transform-explained-in-one-sentence/
- The working implementation, "Equations Explained Colorfully" (KaTeX, MIT; it credits Riffle and BetterExplained): https://p.migdal.pl/equations-explained-colorfully/ and https://github.com/stared/equations-explained-colorfully
- Distill, "Communicating with Interactive Articles" (Hohman, Conlen, Heer, Chau 2020) on colorized equations and linked hover highlights: https://distill.pub/2020/communicating-with-interactive-articles/
- The enabling KaTeX feature: `\htmlClass`, `\htmlId` and `\htmlData`, which are disabled by default and need the `trust` option plus a relaxed `strict`: https://katex.org/docs/supported

**→ KNN app:**
- Render the distance as a sum over features with the live values substituted:
  `d(q, xᵢ) = ( Σⱼ |q_j − x_ij|^p )^{1/p}`
  Each per-feature term is wrapped in `\htmlData{feat=j}{…}`. Hovering a term highlights the matching leg of the distance path in the plot (in 2-D the |Δx| and |Δy| legs, which is exactly the L1 "taxicab" path). It also highlights the bar showing that term's share of the sum.
- Clicking a term pins it.
- **Unscaled wine:** the `proline` term visibly swallows the sum. After standardising, the terms balance out.
- **Colour budget:** class identity uses the categorical palette (§7). Do **not** also colour features with categorical hues. Use one accent colour and highlight only the hovered or pinned feature.
- Offer "Accessible" and "No colour" modes, as Migdał does, so the linking still works through underline and outline.

## 5. Layout: scrollytelling vs side-by-side workbench
**Sources:**
- Zhi, Ottley & Metoyer, "Linking and Layout" (*CGF* 2019, n=180): https://diglib.eg.org/handle/10.1111/cgf13719. The slideshow layout gave better comprehension than vertical scrolling and was preferred. Two-way text↔vis linking raised engagement and, combined with slideshow, recall.
- Distill 2020 (above) calls slideshow vs scroll "an ongoing debate". It also notes Zhi et al., and McKenna et al. finding no engagement difference.
- Mittenentzwei et al. ran an 85-participant study of slideshow vs scrollytelling: https://visualstories.cs.ovgu.de/studies/genres. The page doesn't state findings.
- MLU-Explain uses scroll-driven layout (it credits Scrollama). R2D3 was blocked (robots), so it is not assessed.

| | Scrollytelling | Side-by-side workbench (+ optional guided steps) |
|---|---|---|
| Pros | Strong narrative. Low cognitive load per screen. Works well on mobile. | Everything stays visible. Free exploration. Natural fit for uploaded data and many controls. Linked views stay on screen together. |
| Cons | Hard with user-uploaded data, since the narrative assumes known data. Scroll position fights direct manipulation. Long pages make comparisons hard. | Can overwhelm. Needs progressive disclosure (§6). Weak storytelling unless a guided mode is added. Harder on phones. |
| Evidence | Lost to slideshow in Zhi et al. 2019 | Slideshow-style stepping won in Zhi et al. 2019 |

**→ KNN app:**
- Use a **workbench**: a fixed canvas (map plus overlays) on the left; on the right, the equation and tables and a stepper, with a pipeline strip on top: Data → Encode → Scale → Distances → Neighbours → Vote → Predict.
- Add an optional **guided tour** as discrete slides that drive the same workbench state (slideshow, not scroll), e.g. "Step 3 of 7: sort by distance".
- On mobile, stack the views and keep the stepper sticky.

## 6. Progressive disclosure and details-on-demand
**Sources:**
- Jakob Nielsen, "Progressive Disclosure" (NN/g, 3 Dec 2006): https://www.nngroup.com/articles/progressive-disclosure/ (video summary: https://www.nngroup.com/videos/progressive-disclosure/). Show "only a few" key options first and the specialised ones on request. Designs beyond two levels usually test poorly. *Staged* disclosure, wizard-style, suits linear steps that are fairly independent.
- Shneiderman's mantra, "Overview first, zoom and filter, then details-on-demand", as cited by Distill 2020 (above).

**→ KNN app:**
- **Staged** (wizard) for data intake: Upload → Pick target → Column types (numeric, categorical → one-hot/ordinal, ID → drop) → Missing values → Scaling → Explore. Each stage is independent and shows before and after.
- **Progressive** (two levels at most) in the workbench. Level 1: k, metric (L1/L2), features for the X/Y axes. Level 2, behind "Advanced": Minkowski p, weighting, tie rule, scaling method, grid resolution, validation split.
- **Details-on-demand:** the "Show the math" panel starts collapsed. Hovering a cell in a table shows the formula with its numbers substituted in.

## 7. Colour-blind-safe categorical palettes (≤ 10 classes)
**Sources:**
- Okabe & Ito, "Color Universal Design" (jfly, Univ. Cologne mirror): https://jfly.uni-koeln.de/color/. The palette there is an image; hex values were taken from the ConceptViz reference, which cites Okabe & Ito 2008 and Wong 2011: https://conceptviz.app/blog/okabe-ito-palette-hex-codes-complete-reference. Nature Methods (Wong 2011) returned an error.
- Tableau 10 hex values verbatim from d3-scale-chromatic source `Tableau10.js`: https://github.com/d3/d3-scale-chromatic/blob/main/src/categorical/Tableau10.js. Design rationale by Maureen Stone, 28 Jul 2016: https://tableau.com/blog/colors-upgrade-tableau-10-56782 (10 hues spread around the hue circle, designed in CIELAB).
- d3 categorical schemes list: https://D3js.org/d3-scale-chromatic/categorical

**Okabe–Ito (8, including black)**

| Name | Hex |
|---|---|
| Orange | `#E69F00` |
| Sky blue | `#56B4E9` |
| Bluish green | `#009E73` |
| Yellow | `#F0E442` |
| Blue | `#0072B2` |
| Vermillion | `#D55E00` |
| Reddish purple | `#CC79A7` |
| Black | `#000000` |

**Tableau 10** (`d3.schemeTableau10`)

| Name | Hex |
|---|---|
| Blue | `#4e79a7` |
| Orange | `#f28e2c` |
| Red | `#e15759` |
| Teal | `#76b7b2` |
| Green | `#59a14f` |
| Yellow | `#edc949` |
| Purple | `#af7aa1` |
| Pink | `#ff9da7` |
| Brown | `#9c755f` |
| Grey | `#bab0ab` |

- Paul Tol's "muted" scheme (9 colours) is listed as colour-blind safe in the khroma vignette (https://www.stats.bris.ac.uk/R/web/packages/khroma/vignettes/tol.html). I could not verify its hex codes from a fetched page, so check Tol's technical note before using it.

**→ KNN app:**
- **Up to 7 classes:** use the 7 Okabe–Ito hues, dropping black, which is reserved for the query point and outlines. **8–10 classes:** use Tableau 10. **Over 10:** group the rarest classes as "other", or refuse with a message.
- Always add **redundant encoding**: a different marker shape per class and the class name in tooltips and the vote table.
- **Decision regions:** draw a light tint of the class colour, with saturation or alpha for vote share as MLU does. Ties get a neutral grey or white, as in CS231n. Check region tints against both light and dark themes.
- **Yellow** (`#F0E442`, `#edc949`) has poor contrast on white for small points, so give points a dark stroke.

## 8. Animation for algorithm steps (step / play / scrub)
**Sources:**
- Tversky, Morrison & Bétrancourt, "Animation: can it facilitate?" (*IJHCS* 2002): https://hci.stanford.edu/courses/cs448b/papers/Tversky_AnimationFacilitate_IJHCS02.pdf
  - **Congruence:** structure should match the content.
  - **Apprehension:** must be perceivable; animations are often too fast or complex.
  - Most benefits came from interactivity or extra information, not motion itself.
  - Learner control matters: pace, stop/start, replay.
  - Prefer schematic animation with arrows and highlighting.
- Heer & Robertson, "Animated Transitions in Statistical Data Graphics" (InfoVis 2007): https://idl.uw.edu/papers/animated-transitions (PDF: https://homes.cs.washington.edu/~jheer/files/2007-AnimatedTrans-InfoVis.pdf)
  - "Maintain valid data graphics during transitions".
  - "Use staging for complex transitions".
  - "Maximize predictability" (slow-in/slow-out).
  - "Minimize occlusion".
  - "Make transitions as long as needed, but no longer". The experiments used 1.25 s and 2 s, and staged scatter transitions worked at "around a full second" per stage.
- Playback bar: Galles/Lucet visualizations, https://cmps-people.ok.ubc.ca/ylucet/DS/about.html (Skip Back, Step Back, Play/Pause, Step Forward, Skip Forward, Speed).
- Self-labelling next-step button: Naftali Harris k-means, https://www.naftaliharris.com/blog/visualizing-k-means-clustering/
- Play/restart plus step counter: Distill t-SNE, https://distill.pub/2016/misread-tsne/

**→ KNN app:** model one prediction as a deterministic list of states, so step back is just an index decrement.

| # | Stage | Animation (about 1 s, ease in-out) |
|---|---|---|
| 0 | Place query | Query glyph drops in. Equation shows q's values. |
| 1 | Compute distances | Lines grow from q to each training point, staggered. The distance column fills in. With n > 200, show only the nearest 30 lines and a histogram for the rest. |
| 2 | Sort | Table rows slide into rank order (staged: move, then settle). |
| 3 | Take k | The **neighbourhood ball** grows to the k-th distance. Its shape follows the metric: diamond for p=1, circle for p=2, rounded square for p≈4, square for p=∞. Points outside fade. |
| 4 | Vote | Neighbour dots fly into stacked vote bars, sized by weight (1 or 1/d). |
| 5 | Predict | The query takes the winning colour. Ties flash grey with the tie rule shown. |

- **Controls:**
  - A fixed playback bar (⏮ ◀ ▶/⏸ ▶ ⏭) and a speed control.
  - A **scrubber** over stages 0–5.
  - Keyboard: ← → to step, space to play.
  - A next-step button labelled with what it will do ("Sort by distance").
- **Prediction loop:** let the reader guess the class before stage 5 (Distill 2020: prediction prompts aid recall).
- **Reduced motion:** honour `prefers-reduced-motion` by cutting between states instead of tweening.

## 9. Further notes from the Distill survey
Source: Distill 2020, https://distill.pub/2020/communicating-with-interactive-articles/
- Represent the same idea several ways, algebraic and geometric: the equation panel next to the map.
- Prediction prompts ("You Draw It") improve recall. For us: "Click where you think the boundary moves when k goes from 1 to 15", then reveal.
- Pitfalls: interactive work costs much more to build, tooling is immature, accessibility needs alternatives, and "not everything needs to be interactive". Keep static fallbacks: the table view and text summaries of each step.
