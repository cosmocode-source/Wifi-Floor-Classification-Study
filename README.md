# WiFi Floor Classification Benchmark

**A paper-inspired study of PCA–kNN, distance-weighted PCA–kNN, and classical gradient boosting on UJIIndoorLoc.**

This project investigates whether the reference paper's findings about WiFi fingerprint floor classification hold under explicit preprocessing, shared evaluation splits, and reproducible model selection. It also tests distance weighting as a refinement of kNN.

> **Current verified status:** The project plan, architecture, experiment rules, and five-stage Antigravity implementation prompts have been prepared. Implementation, testing, model training, and measured results are not yet verified. This repository must not be presented as a completed experimental study until execution evidence is recorded below.

---

## Part I — Established project plan and completed preparation

This section records the agreed scope and preparation completed before implementation. Keep its factual claims stable. Update it only when the team deliberately changes the design or verifies a correction; do not silently rewrite it to imply experiments have already run.

### What has been completed

- Defined the task: **floor classification within a known building** using WiFi signal-strength fingerprints.
- Selected two model families and three primary configurations:
  1. PCA + uniform-weight kNN.
  2. PCA + distance-weighted kNN.
  3. Classical gradient boosting.
- Planned separate model folders with cleaning, training, testing, processed-data, artifact, and result locations.
- Specified shared data-validation, split, preprocessing, feasibility, and metric contracts.
- Specified a common result schema and a comparison builder that checks evaluation compatibility.
- Prepared five sequential implementation prompts, including persistent progress tracking through `PROJECT_STATE.md`.
- Defined a **code-authoring-only** workflow for Antigravity: the student runs setup, tests, training, and evaluation locally.

These are completed planning deliverables, not evidence that the software exists or that models have been evaluated.

### Research question

> How do standard PCA–kNN, distance-weighted PCA–kNN, and classical gradient boosting compare for known-building floor classification when they use the same evaluation rows and training-only model selection?

The task does **not** include exact coordinate estimation, room identification, live WiFi scanning, or predicting an unknown building. `BUILDINGID` selects a building-specific model; the building is assumed known at inference time.

### Our contribution

Our intended contribution is an **experimental and methodological refinement**, not a newly invented machine-learning algorithm:

1. **Controlled evaluation of distance weighting.** Compare uniform and distance-weighted kNN at matching neighbor counts, PCA dimensions, scaling settings, and splits. Also compare their best training-side-selected pipelines. This distinguishes a weighting effect from the effect of changing other parameters.
2. **Explicit preprocessing.** Document undetected-signal encoding, validate inputs, and audit duplicates and unusual scans rather than silently deleting them.
3. **Leakage-safe dimensionality reduction.** Fit scaling and PCA inside each training fold, with automated checks that reject infeasible component counts.
4. **Reproducible comparisons.** Share split manifests and row identifiers across models; record configurations, seeds, data hashes, predictions, and software versions.
5. **Consistent reporting.** Use one result format and verify that compared outputs refer to identical evaluation rows and protocols.

These contributions become demonstrated outcomes only after the implementation is run and verified. Distance weighting is an established technique; we do not claim to have invented it. Better evaluation does not guarantee higher accuracy, and a well-supported finding of no improvement is valid.

### Relationship to the reference paper

The project is motivated by *Indoor Positioning System Using Wifi Fingerprint*, attributed in the prior supplied-paper analysis to Dan Li, Le Wang, and Shiqi Wu. Verify the original PDF before finalizing bibliographic details.

The paper-inspired starting points are PCA + 1NN and depth-four gradient boosting with 189 iterations. The paper incompletely specifies preprocessing and evaluation, so our project is a **paper-inspired reconstruction**, not an exact numerical reproduction. Its reported scores are not our results and will not be used as guaranteed targets.

### Dataset and predictor contract

Use the official [UJIIndoorLoc dataset](https://archive.ics.uci.edu/dataset/310/ujiindoorloc), [DOI: 10.24432/C5MS59](https://doi.org/10.24432/C5MS59).

| Item | Expected public-dataset specification |
| --- | --- |
| Training/reference file | `trainingData.csv`: 19,937 records |
| Reserved final-evaluation file | `validationData.csv`: 1,111 records |
| Columns | 529, including 520 WiFi features |
| Predictors | `WAP001`–`WAP520` only |
| Target | Original `FLOOR` labels |
| Known-building selector | `BUILDINGID` |
| Observed RSSI | −104 through 0 dBm |
| Undetected access point | Sentinel `100` |

These are expected counts, not an audit of a local copy. Actual file contents and hashes must be recorded after execution. Follow the official dataset's CC BY 4.0 attribution requirements.

Exclude `FLOOR`, coordinates, space identifiers, user/device identifiers, and timestamps from predictors. Metadata may be retained for audits and evaluation grouping.

### Planned preprocessing and models

- Preserve original CSVs unchanged.
- Validate the exact feature schema and signal values.
- Initially encode `100` as `−110` for all three pipelines. **This is our implementation assumption**, not a verified setting from the paper.
- Investigate scaled and unscaled kNN pipelines using training-side validation.
- Use PCA for both kNN variants; no PCA or standard scaling for gradient boosting by default.
- Reject PCA dimensions exceeding `min(current_feature_count, training_rows - 1)` in the relevant training partition; never silently clamp them.
- Preserve floor labels and report building-specific class supports.

| Configuration | Planned starting point | Planned refinement |
| --- | --- | --- |
| Uniform PCA–kNN | `k=1`, 50 PCA components, Euclidean distance | Shared search over feasible PCA dimensions, scaling, and k |
| Weighted PCA–kNN | Euclidean distance, `weights="distance"` | Same candidate settings and folds as uniform kNN |
| Classical gradient boosting | Depth 4, 189 iterations | Bounded learning-rate, iteration, depth, and leaf-size search |

At `k=1`, distance weighting offers no meaningful multi-neighbor voting advantage. All non-paper settings, including the initial boosting learning rate, must be documented as assumptions.

### Evaluation plan

1. Create a seeded internal development holdout using only the official training file.
2. Share training-side CV folds within the remaining development-training data across all models.
3. Select parameters using CV accuracy with a declared macro-F1 tie-breaker.
4. Evaluate selected pipelines on the same internal holdout, keeping it separate from selection.
5. Freeze configurations before explicit final evaluation on the official validation file. Final training on all official training rows must be recorded.
6. Never tune preprocessing or parameters using final-holdout scores.

Report accuracy, error, macro-F1, balanced accuracy, class supports, confusion matrices, fit/inference timing, and per-building results. Calculate pooled accuracy from pooled correct and total counts. Do not mix CV, internal-holdout, and final-holdout scores in an unlabeled leaderboard.

### Planned repository organization

```text
wifi_floor_project/
├── PROJECT_STATE.md
├── README.md
├── pyproject.toml
├── data/raw/
├── shared/
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── feasibility.py
│   ├── create_splits.py
│   ├── metrics.py
│   ├── result_schema.py
│   └── manifests/
├── models/
│   ├── pca_knn/
│   ├── pca_weighted_knn/
│   └── gradient_boosting/
├── comparison/
├── docs/
├── reports/
├── presentation/
└── tests/
```

Each model folder will contain `config.yaml`, `clean_data.py`, `train.py`, `test.py`, `processed_data/`, `artifacts/`, and `results/`. The comparison builder will produce a canonical `combined_results.json` plus derived tables and reports.

This tree is the agreed architecture, not a claim that every file has been created. `PROJECT_STATE.md` should track actual interfaces and progress.

---

<!-- ANTIGRAVITY_EDITABLE_SECTION_START -->

## Part II — Implementation, execution, and results

> **Maintenance instructions for Antigravity:** Update this section as implementation progresses. Read `PROJECT_STATE.md` and relevant files first. Only document commands supported by the actual code. Distinguish authored code, executed commands, and verified results. Do not execute anything unless the user changes the no-execution instruction. Do not mark tests passed or training complete without execution evidence supplied by the user. Replace pending text with verified details; do not invent values. Keep Part I unchanged unless an explicit design correction is needed.

### Current implementation status

| Area | Verified status |
| --- | --- |
| Planning and five-stage prompts | Prepared |
| Shared infrastructure | Pending verification |
| Uniform PCA–kNN | Pending verification |
| Distance-weighted PCA–kNN | Pending verification |
| Classical gradient boosting | Pending verification |
| Comparison/report builder | Pending verification |
| Tests executed | No verified execution evidence |
| Real-data training/evaluation | No verified execution evidence |

### Environment and setup

Pending implementation. Add the supported Python version, dependency-installation commands, working directory, OS notes, and actual environment information.

Planned execution is CPU-based on an ASUS A15 with 16 GB RAM. The RTX 3050 is not required for the selected scikit-learn implementations. Avoid nested parallelism; actual runtime must be measured.

### Data placement and audit

Pending execution. Add actual CSV paths, official-source provenance, file hashes, audited row counts, building/floor supports, diagnostic findings, and any justified exclusions. Do not commit raw datasets by default.

### Supported commands

Pending implementation. Add exact commands in this order:

1. Environment setup.
2. Data audit and shared split generation.
3. Cleaning and training for each model.
4. Testing on the shared internal holdout.
5. Building the combined comparison.
6. Explicitly confirmed frozen final evaluation.
7. Academic report/presentation exports.

Do not publish example flags as runnable instructions until the corresponding interfaces exist.

### Test evidence

Pending user execution. Record the command, date, relevant commit, pass/fail/skip counts, unresolved failures, and log location. Clearly label synthetic-fixture tests separately from real-data experiments.

### Measured results

**No verified measured results are available yet.**

Populate results only from actual saved artifacts. Include evaluation role, row support, run IDs, selected parameters, preprocessing, per-building metrics, pooled metrics, timing scope, and limitations. Link to the canonical combined result and prediction files.

Keep paper-reported scores separate from locally measured scores. Do not claim weighted kNN or boosting is better before evaluation.

### Live demonstration

Pending implementation and rehearsal. Document how to load a saved trusted pipeline and predict from WAP inputs plus a known building, without retraining or an internet connection. Ground truth, when present, is evaluation-only. Record the real rehearsal outcome.

### Team and individual contributions

Pending team details. Add actual names, student identifiers, section, assigned problem serial, and evidence of completed contributions. Do not invent identities, contribution percentages, commits, or work history.

### Submission readiness

- [ ] Exact assigned problem statement/serial confirmed.
- [ ] Two-member team details confirmed.
- [ ] Private GitHub repository verified.
- [ ] Required faculty/TA repository access verified.
- [ ] Implementation and README commands verified.
- [ ] Tests and real-data experiments executed and recorded.
- [ ] Faculty clarified one-page versus two-page write-up requirement.
- [ ] Required PDF write-up prepared and checked.
- [ ] Slide deck prepared and checked.
- [ ] Working-code live demo rehearsed.
- [ ] Q&A and individual contribution evidence prepared.
- [ ] Applicable submission schedule/channel confirmed.

The assignment's original deadline was October 10, 2026, at 11:59 PM. A revised schedule or extension has not been verified. Do not imply that one exists.

### Limitations and conclusions

Pending measured evidence. Discuss the known-building assumption, dataset specificity, encoding assumptions, reproduction gaps, possible dependence between repeated scans, and whether results support the stated research question.

<!-- ANTIGRAVITY_EDITABLE_SECTION_END -->
