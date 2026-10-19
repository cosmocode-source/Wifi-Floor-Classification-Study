# Project Information, Explanation, and Review

This is the **single consolidated explanation file**. `README.md` contains the
short setup instructions. Each model folder has separate `train.py`,
`clean_data.py`, and `test.py` entry points; there is no root execution wrapper.

## 1. Project and scientific scope

**Question:** How do uniform PCA–kNN, distance-weighted PCA–kNN, and classical
gradient boosting compare for floor classification within a known building?

Inputs are the ordered 520 access-point measurements `WAP001`–`WAP520`.
`FLOOR` is the target. `BUILDINGID` selects the building-specific classifier;
it is not a predictor. Coordinates, room IDs, user IDs, devices, and timestamps
are excluded from predictive inputs.

We implement an established-method comparison, not a new localization algorithm.
Weighted kNN is a variant of kNN. XGBoost is not part of this project.

### Dataset

The public UJIIndoorLoc reference files contain 19,937 training rows and 1,111
validation rows, with 529 columns each. There are three buildings, and numerical
floor labels span 0–4 across the whole collection. Individual buildings have
different floor coverage.

Observed RSSI is negative or zero; `100` denotes an undetected access point.
This project replaces exactly `100` with `-110.0`; it preserves genuine `0`
readings. The replacement is our assumption, not a verified author setting.
An all-undetected row consequently becomes a constant vector. Retaining those
rows is transparent but does not create location information that was absent.

The CSVs belong in `Dataset/`. They were absent from the uploaded archive.
No new real-data audit, training, or accuracy verification was possible from
the archive alone.

## 2. Paper interpretation

Reference: Dan Li, Le Wang, Shiqi Wu, *Indoor Positioning System Using Wifi
Fingerprint*, supplied five-page Stanford-affiliated paper.

The paper tests floor classification separately by building, with PCA and kNN,
SVM, decision trees, and gradient boosting. It discusses sample sizes
100/200/500/1000/2000/5000, ten-fold CV, and five rounds per parameter setting.
It does not establish the exact PCA fitting order or complete split/hyperparameter
settings.

Paper-reported kNN errors:

| Components | 500 samples | 5,000 samples |
|---:|---:|---:|
| 5 | 0.1193 | 0.0281 |
| 50 | 0.0340 | 0.0017 |
| 200 | 0.0200 | 0.0012 |

These are reported numbers, **not measured results from this project**.
The best listed error is equivalent to 99.88% accuracy, but its aggregation and
test protocol are not clear enough to guarantee equivalent official-holdout
performance.

The SVM prose says 150 features while the figure caption/legend says 500.
The tree table calls size values depth while the prose calls them split counts.
Gradient boosting is incorrectly described as bagging. Its stated depth is four,
its selected iteration count is 189, and it reports 5% test error and a separate
3% CV error on 100 samples. Multiclass iterations need not equal total tree count.
No evaluated kNN–boosting hybrid is defined.

## 3. Simple explanation of the models

### Uniform kNN

1. Convert undetected values to a finite weak-signal value.
2. Optionally standardize using training means and standard deviations.
3. Fit PCA on the training partition and project fingerprints.
4. Find the `k` nearest training fingerprints using Euclidean distance.
5. Give each neighbor one vote and predict its majority floor.

PCA finds directions of variance, not the physically strongest routers. It creates
combinations of access-point features. The fixed baseline uses unscaled, centered
PCA with 50 components and uniform 1NN.

### Distance-weighted kNN

The pipeline is the same, but closer neighbors receive more voting weight.
At `k=1`, both methods are equivalent; this is not evidence of an improvement.
An interpretable weighting comparison must keep `k`, PCA dimensions, scaling,
data encoding, and splits matched. Independently tuning both models answers a
different question about their best selected complete pipelines.

### Classical gradient boosting

A sequence of trees minimizes classification loss. Each stage adjusts the
current model; boosting is not independent bootstrap averaging.

The default input is raw, finite sentinel-replaced RSSI: **no PCA and no scaler**.
Optional representations are explicit ablations. sklearn's classical
`GradientBoostingClassifier` does not provide the native NaN routing claimed
for some other boosting implementations.

Depth 4 and 189 iterations are paper-inspired. Learning rate 0.1 and other
defaults are implementation assumptions. The bounded search explores tree
depth, learning rate, boosting stages, and minimum leaf support.

## 4. How the code is organized

| File/package | What to explain |
|---|---|
| `models/<model>/train.py` | Independent training command for that model |
| `models/<model>/clean_data.py` | Cleaning plus model-specific scaling/PCA pipeline construction |
| `models/<model>/test.py` | Independent saved-model evaluation command |
| `shared/config.py` | Dataset paths, allowed features, signal constants, labels, split roles |
| `shared/data_validation.py` | Schema/range checks, integer labels, immutable audits, hashes, row IDs |
| `shared/create_splits.py` | Per-building train/holdout split, saved CV folds, index/provenance checks |
| `shared/preprocessing.py` | Stateless sentinel conversion plus fold-local sklearn pipelines |
| `shared/feasibility.py` | Reject impossible PCA, neighbor counts, and class/fold configurations |
| `shared/metrics.py` | Accuracy, macro-F1, balanced accuracy, class metrics, pooled counts |
| `shared/result_schema.py` | Validate saved metrics, predictions, statuses, and confusion counts |
| `shared/artifacts.py` | Save pipelines; verify local paths and checksums on reload |
| `shared/demo.py` | Offline inference using saved building-specific models |
| `models/pca_knn/` | Configuration, optional cleaning, CV selection, refitting, evaluation |
| `models/pca_weighted_knn/` | Reuses the kNN engine with distance weighting |
| `models/gradient_boosting/` | Classical boosting selection and evaluation |
| `comparison/` | Align predictions, recompute results, compare timing/representations, report |
| `tests/` | Synthetic arithmetic, scientific safeguards, persistence, integration, regressions |

The simplest viva summary is:

> Load WiFi features → use shared splits → fit preprocessing and model only on
> training rows → predict unseen rows → compare correct predictions and
> class-balanced performance.

The longer modules also record provenance, model selection, failure states,
timing, and reproducible outputs. They were retained rather than replaced with
an unverified short rewrite. Long duplicated explanatory docstrings and comment
clutter were reduced; substantive runtime behavior and tests were preserved.

## 5. Development and final evaluation

1. Use only `trainingData.csv` to create a 20% internal holdout per building.
2. Within `dev_train`, use the saved 3-fold or 5-fold stratified CV assignments.
3. Fit every scaler/PCA inside each fold. Never fit them globally before CV.
4. Select parameters by mean CV accuracy, then macro-F1 and declared deterministic
   tie-breakers. Best-search CV scores are selection estimates, not unbiased final
   estimates.
5. Refit selected models on each building's development-training rows.
6. Evaluate the frozen pipelines on the internal holdout without refitting.
7. Only at finalization, explicitly refit frozen settings on all official training
   rows and score the untouched official validation file.

This project's 3/5-fold development protocol is not an exact reproduction of the
paper's ten-fold/five-round procedure.

PCA components must not exceed `min(features, smallest_training_partition - 1)`.
The code skips/rejects infeasible settings; it does not label a clamped dimension
as the originally requested one. Rare classes are not silently removed.

### Leakage and duplicate fingerprints

Equal row IDs must not cross split boundaries. However, different rows can have
identical WiFi fingerprints. Random row splits can therefore give optimistic
results for repeated measurements, particularly for 1NN.

Inspect training/validation fingerprint overlap, all-undetected rows, and class
supports. Report limitations and consider a grouped reference-point/fingerprint
evaluation. This review did not invent a grouped-generalization result or silently
change the historical shared split.

### Metrics

- Accuracy = correct predictions / evaluated rows.
- Error = incorrect predictions / evaluated rows.
- Macro-F1 treats the chosen class labels equally.
- Balanced accuracy averages recall over true classes present.
- Confusion matrices show which floors are confused.

Overall pooled accuracy comes from total correct/total rows, **not** an unweighted
average of building accuracies. The global-floor-union macro-F1 and diagnostic
building-averaged F1 are distinct quantities. Fold spreads are descriptive:
overlapping CV folds are not independent repetitions.

## 6. Historical artifacts included in the upload

The archive contains `shared/manifests/dataset_audit_report.json`,
`development_splits.csv`, and its metadata. They were preserved unchanged.
They describe a previous audit and split generation, not a new audit performed
during this review.

Recorded training split: 15,948 development-training rows and 3,989 internal-holdout
rows, totaling 19,937. Recorded seed: 42. Existing manifests are reused only when
their source and file hashes match.

Recorded source digests:

- Training: `45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168`
- Validation: `5f90c536648cd657b2c516d20c4e0968d4003279ea6bd5d5d5322d3f1e8905c0`
- Manifest: `0e0effe365f6568cc712d8dd046c14d5f60f56c32d4ddb403eb51630235a6fed`

The archived audit reports duplicate fingerprints and 76 all-undetected training
rows. The raw CSVs were not provided, so those raw-data diagnostics have not been
independently recomputed here.

An important audit defect was found: a fingerprint shared across buildings was
previously excluded from the check for conflicting floors within each building.
The supplied all-undetected row IDs map to floors 0, 1, and 2 in building 2 in the
saved manifest. The corrected audit checks each building inside every fingerprint
group. Re-running the data-audit command will produce updated diagnostics; the historical JSON
was not silently edited to pretend a new raw-data audit happened.

## 7. Cleanup and correctness fixes

### Removed from the returned copy

- `__pycache__`, compiled `.pyc`, pytest cache, generated egg-info.
- Embedded `.git` administration/history. **Your original local repository was
  not modified.** Apply reviewed source changes to that repository if preserving
  its history matters.
- Root `PROJECT_STATE.md`, extra README files, and scattered explanatory Markdown
  in `docs`, `reports`, and `presentation`.
- Empty placeholder runtime directories and the obsolete `data/raw` scaffold.

Their useful scientific, execution, viva, and assignment information is summarized
in this file. Original source and assignment PDFs are preserved. No uploaded CSV
or trained-model artifact was deleted: none was present.

### Fixed

- Dataset defaults always use `Dataset`, including when files are missing.
- DataFrame inference rejects renamed/reordered features instead of silently
  predicting from a swapped feature order.
- Transformer use before fitting and non-finite replacement settings are rejected.
- Fractional/non-finite floor and building labels are rejected instead of truncated.
- Manifest row indices must match original source order and be finite integers.
- Raw training content is verified against frozen provenance during evaluation,
  not just against the hash stored in a manifest.
- Non-finite candidate selection scores are rejected.
- Prediction labels/model IDs and confusion-matrix counts receive stronger checks.
- Strict metric comparison uses absolute tolerance without an unintended relative
  tolerance.
- Duplicate-vector audits no longer hide within-building conflicting-floor groups
  merely because the same vector also appears in another building.
- An undefined-variable fallback in gradient-boosting metadata was replaced with
  the intended configured replacement value.
- README execution commands use implemented CLI flags, replacing outdated commands.
- Per your requested organization, model-specific pipeline factories now live in
  each model folder's `clean_data.py`; `train.py` imports them. Shared implementation
  stays in `shared/preprocessing.py` to avoid duplicate scaler/PCA logic.
- The proposed single root execution wrapper and wrapper-only tests were removed.
  Regression tests for actual model entry points are retained.
- The weighted-kNN evaluation CLI previously accessed a missing `args.seed`.
  The shared evaluation parser now provides the seed and forwards it consistently.
  This defect was found by actual command-line execution, not by parser-only tests.
- Shared package exports are now lazy, preserving their public interfaces while
  avoiding runpy warnings caused by pre-importing command-line modules.

Joblib loading remains suitable only for models from trusted local runs.
A digest detects changed bytes; it does not authenticate malicious replacement
of both model and sidecar. Cleaned exports carry source provenance but are not
cryptographically equivalent to their original raw files.

## 8. Review execution record

### Tests and static checks

- Original uploaded code: **73 tests passed** in the review environment.
- Final per-model version: **93 tests passed in 35.85 seconds**, with no failed
  or skipped tests. The earlier root-wrapper version had 96 tests; nine
  wrapper-only cases were removed with that script and six independent
  model-entry-point/pipeline cases were added. This is not a failed-test deletion.
- Static checks for undefined names, broken exports, and syntax errors passed
  (`ruff check --select F821,F822,F823,E9 .`). This is a targeted check, not a
  claim that every lint rule or type annotation has been verified.
- Tests cover original model/search/report behavior plus feature ordering,
  integer labels, manifest ordering, provenance, duplicate diagnostics,
  non-finite metrics, actual evaluation seed forwarding, and preprocessing
  ownership in each model's cleaning file.

### Independent module execution

In a separate temporary directory, all three model cleaning commands, all three
smoke-profile training commands, all three internal-holdout evaluation commands,
the comparison command, and the saved-model demo completed successfully using
**195 synthetic training rows** and a separately reserved synthetic validation
fixture. The internal split was 156 development / 39 holdout rows. Official
validation evaluation was not executed.

The uniform fixed PCA-50 reference was correctly marked infeasible/skipped on
these small CV training partitions rather than silently reducing its dimension.
A previous actual CLI run exposed the missing weighted-evaluation seed argument;
it was fixed and the separate-module workflow was rerun successfully. The final
workflow log had no traceback or runpy warning.

Synthetic fixtures, trained fixture models, fixture result tables/plots, and
reviewer logs are **not included** in the returned project. Synthetic scores
are not evidence of real UJIIndoorLoc accuracy.

### Verified environment and retained evidence

The review used Linux with Python **3.13.14**, NumPy **2.5.4**, pandas **2.3.3**,
scikit-learn **1.9.1**, SciPy **1.18.1**, joblib **1.6.0**, PyYAML **6.0.3**, and
pytest **8.4.2**. This does not establish identical behavior on Windows or every
version permitted by the dependency ranges. Windows PowerShell commands are
provided in the README; PowerShell itself was not available for execution here.

The two source PDFs and the three supplied historical audit/split files were
verified byte-for-byte against the uploaded archive. Neither raw dataset CSV nor
real trained artifact was available, so historical audit counts and hashes are
retained evidence, not a freshly verified real-data result.

No guarantee of zero bugs is made. Checks establish specific tested behavior;
they cannot establish every future environment or real-data outcome.

## 9. Execution and results interpretation

The README contains independent commands for each model, not one root script:

- `shared.data_validation`: audit original CSVs without changing them.
- `shared.create_splits`: generate training-only shared folds when needed. Do not
  routinely overwrite the included manifest or use official validation for folds.
- `models.<model>.clean_data`: optional stateless cleaning exports and model-specific
  pipeline factory. Creating a pipeline does not fit scaling/PCA.
- `models.<model>.train`: fit cleaning, scaling/PCA when enabled, and classifier
  inside each training fold, then save the selected model.
- `models.<model>.test`: internal evaluation, or explicitly confirmed official
  evaluation with all-training refit after settings are frozen.
- `comparison.compare`: compare explicitly supplied compatible result paths.
- `shared.demo`: saved-model inference on clearly labeled development examples.

Runtime reports are experiment outputs, not additional permanent explanation
files. Optional academic write-up and presentation generators are retained because
they are functioning code and required by the assignment, but are not called
automatically. Final PDF/presentation formatting still needs inspection.

Use new run IDs to avoid overwrites. If a multi-model action stops partway through,
inspect partial output directories rather than pretending the whole action
succeeded. Retrain with a new run ID if appropriate. Compare only compatible
evaluation roles and refit policies.

When explaining results, separate:

1. Paper-reported scores.
2. Synthetic test outcomes.
3. Actual development-data measurements.
4. Official final benchmark measurements.

Do not infer a real-data improvement from synthetic tests or `k=1` equivalence.

## 10. Assignment, presentation, and viva

The original assignment PDF is preserved. It requires a private GitHub repository
shared with faculty/TAs, setup instructions, a PDF write-up, presentation, live
demonstration, and explanation of individual contributions.

Its heading says “One-Page Write-up” while the body says “two-page summary”;
clarify the page limit with faculty. The source document's original deadline was
October 10, 2026, 11:59 PM; any revised submission schedule is not known here.
Team names, roll numbers, and assigned problem number require confirmation.

Suggested report structure:

1. Problem and known-building assumption.
2. Dataset, sentinel encoding, and actual audit.
3. Three models and fold-local preprocessing.
4. Shared CV selection and held-out evaluation.
5. Actual metrics, computational cost, and limitations.
6. Conclusion and contributions.

Suggested slide outline: problem → paper → dataset → preprocessing/splits →
three models → evaluation → measured results → limitations → demonstration.

For the live demo: run `check`, show one saved comparison, explain a confusion
matrix, run `demo`, and explicitly state that development rehearsal examples are
not independent test evidence.

Common viva answers:

- **Why not GPS?** Indoor reception/propagation can make it unreliable; WiFi offers
  an existing fingerprint source.
- **Why PCA?** Reduce representation size/correlation for kNN; it is unsupervised
  and can discard class-relevant low-variance information.
- **Why not fit PCA once?** Validation information would influence the learned
  representation and bias CV.
- **Why weighted kNN?** Give closer neighbors more influence; at `k=1` it is identical.
- **Why boosting without PCA?** Trees can use raw feature thresholds; dimensionality
  reduction is not universally beneficial.
- **Why not promise 99.88%?** The paper's evaluation details differ and are incomplete.
- **Why accuracy plus macro-F1?** Accuracy can hide poorer performance on rare floors.
- **Why retain duplicates?** Avoid unreported data deletion; disclose overlap and
  use sensitivity evaluations where possible.
- **Is this full positioning?** No: it predicts floor given a known building.
- **What is genuinely improved?** Only claim improvements supported by actual,
  matched local experiments.

## 11. Sources

- Supplied paper: `InfoFiles/Dan Li, Le Wang, Shiqi Wu, Indoor Positioning System Using Wifi Fingerprint.pdf`
- Supplied assignment: `InfoFiles/Guidelines and Instructions_Mini Project Assignment.pdf`
- UCI dataset: https://archive.ics.uci.edu/dataset/310/ujiindoorloc
- Leakage guidance: https://scikit-learn.org/stable/common_pitfalls.html
- Classical boosting: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
- kNN: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

The original uploaded ZIP is the backup for verbatim historical documentation.