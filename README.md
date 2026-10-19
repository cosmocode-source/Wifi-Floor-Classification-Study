# WiFi Fingerprint Floor Classification

**Arin Singh — PES1UG24CS079**  
**Archna Shivkumar — PES1UG24CS078**  
Section B · UE24CS352A: Machine Learning

## 1. Start here: evaluate the supplied trained models

The models are already trained. **`evaluate` loads them and predicts the validation set. It does not train models, refit PCA, select k, or use validation rows for fitting.**

After setup, run this command from the project folder:

```bash
python run.py evaluate
```

Then open **`Results/Comparison.html`** in any browser. It is a standalone offline report; no web server, account, internet connection or JavaScript library is needed to view it.

The report presents:
1. **Validation classification error first:** error percentages and incorrect scan counts for every configuration; lower is better.
2. Accuracy, correct counts and the common evaluation population.
3. Mean training-CV error versus validation error, their gap in percentage points, and five repetition scores.
4. Per-building validation errors and accuracy.
5. All four validation confusion matrices.
6. Model settings, selected k, all evaluated k errors in an expandable table, and interpretation/limitations.

The CSV files in the same Results folder support auditing and re-use; the HTML is the main result to read. No training is needed just to regenerate validation results.

## 2. Setup

### Requirements

- **Python 3.13** (the supplied run and this repackaging were tested with **Python 3.13.14 on Linux**).
- Packages pinned in `requirements.txt`. Use those exact versions for the supplied models; joblib/scikit-learn artifacts are not universally portable across versions.
- Internet is needed for the initial dependency installation, not for evaluation after setup.
- Extract the ZIP before running commands. Keep the project directories together.
- Training is CPU-intensive and much slower than evaluation. There is no GPU requirement.

### Windows (Command Prompt)

Install Python 3.13, open Command Prompt in `WiFi_Floor_Project`, then run:

```bat
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run.py evaluate
```

Open `Results\Comparison.html` after the command completes. Windows commands are provided, but this package's execution checks were run on Linux; Windows was not executed in this environment.

### Linux/macOS

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py evaluate
```

In the remaining examples, `python` means the virtual environment's Python. Either activate it first, or replace `python` with `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on Linux/macOS. An unactivated system Python may not have the dependencies installed.

## 3. Folder structure

```text
WiFi_Floor_Project/
├── Dataset/
│   ├── trainingData.csv
│   └── validationData.csv
├── Cleaned_Dataset/
│   ├── clean_data.py
│   ├── trainingData_cleaned.csv
│   └── validationData_cleaned.csv
├── PCA_KNN/
│   ├── model.py
│   ├── run.py
│   └── Saved_Models/
│       ├── building_0.joblib
│       ├── building_1.joblib
│       ├── building_2.joblib
│       ├── combined.joblib
│       └── metadata.json
├── Random_Forest/
│   ├── model.py
│   ├── run.py
│   └── Saved_Models/
│       ├── building_0.joblib
│       ├── building_1.joblib
│       ├── building_2.joblib
│       ├── combined.joblib
│       └── metadata.json
├── Results/
│   ├── Comparison.html
│   ├── Comparison.csv
│   ├── PCA_KNN/
│   │   ├── CV_Results.csv
│   │   ├── CV_Predictions.csv
│   │   ├── K_Search.csv
│   │   ├── Validation_Results.csv
│   │   ├── Validation_Predictions.csv
│   │   └── Confusion_Matrices.csv
│   └── Random_Forest/
│       ├── CV_Results.csv
│       ├── CV_Predictions.csv
│       ├── Validation_Results.csv
│       ├── Validation_Predictions.csv
│       └── Confusion_Matrices.csv
├── requirements.txt
├── run.py
└── README.md
```

The PDF submission report is a separate deliverable, not part of this runnable package. There are no checkpoint, research, test, documentation or launcher folders in this distribution. Python may create `__pycache__` folders during execution; those are generated caches, not additional project components.

### Responsibilities

- **Dataset:** untouched original CSVs; training has 19,937 rows, validation has 1,111.
- **Cleaned_Dataset:** validated, encoded copies and the only cleaning implementation. No row is removed. Other metadata columns are retained in the cleaned CSVs but are not model predictors.
- **PCA_KNN/model.py:** PCA pipeline and deterministic exact kNN implementation. `PCA_KNN/run.py` is this algorithm's entry point.
- **Random_Forest/model.py:** fixed forest pipeline. `Random_Forest/run.py` is this algorithm's entry point.
- **Saved_Models:** three building specialists plus one combined pipeline per algorithm. `metadata.json` stores selected parameters, training-data hash, runtime versions, model hashes and CV provenance; it is required to load the supplied models safely and associate results with them.
- **Results:** one location for every result. Algorithm subfolders prevent overwriting each other's predictions. The root comparison summarises all four configurations.
- **Root run.py:** shared orchestration for training, saved-model evaluation and HTML generation. The individual runners call it, avoiding duplicated evaluation/training logic.

## 4. Commands

### Evaluate both algorithms (recommended demonstration)

```bash
python run.py evaluate
```

This command:
- Reads and checks **`Dataset/validationData.csv`**.
- Regenerates `Cleaned_Dataset/validationData_cleaned.csv` using the fixed cleaning rule.
- Loads the four saved pipelines from each algorithm folder (eight fitted pipelines total).
- Produces building-wise and combined predictions for each algorithm.
- Updates that algorithm's validation scores, predictions and confusion matrices.
- Automatically creates `Results/Comparison.csv` and **`Results/Comparison.html`**.

The final kNN pipelines already contain their trained encoder, fitted PCA and classifier. The forests already contain their fitted trees. Neither algorithm's `fit` method is called during evaluation.

### Evaluate algorithms separately

```bash
python PCA_KNN/run.py evaluate
python Random_Forest/run.py evaluate
```

Each command recomputes only that algorithm's validation results and refreshes the comparison, retaining the other algorithm's existing compatible evaluation results. If the other algorithm has no results, the report explicitly says it is a partial comparison. Standard evaluation uses the same original validation CSV each time.

### Clean both original datasets

```bash
python Cleaned_Dataset/clean_data.py
```

Equivalent root command:

```bash
python run.py clean
```

Original datasets are never overwritten. Cleaned copies are included for inspection. Evaluation regenerates its standard cleaned input so a manually edited/stale cleaned CSV cannot silently change the benchmark. Training independently regenerates its cleaned training input.

### Rebuild the HTML without evaluating again

```bash
python run.py report
```

Uses recorded CSV results after checking model hashes, CV score-file integrity, prediction totals and confusion matrices. It does not train or predict. A missing evaluation, mixed validation-file hashes or mismatched model/results produces an error instead of a misleading comparison.

### Deliberately retrain

```bash
python PCA_KNN/run.py train --overwrite
python Random_Forest/run.py train --overwrite
```

Or retrain both:

```bash
python run.py train --overwrite
```

**Warning:** `--overwrite` replaces that algorithm's trained models and CV results. The plain `train` command refuses to replace existing models. A complete new run performs five repetitions of ten-fold CV and then final full-training fits. It reads training data only. This simplified package does not retain fold checkpoints or implement resume; allow the training command to finish.

After successful training, generate fresh validation results:

```bash
python run.py evaluate
```

Old comparison/validation results are invalidated before training so they are not presented as results of new models. If training fails, its metadata remains incomplete and evaluation refuses that model set. Restore the original ZIP or complete retraining before evaluating it.

### Evaluate another labelled validation file (optional)

```bash
python run.py evaluate --validation path/to/validation.csv
```

It must have the 520 ordered WAP columns, BUILDINGID and FLOOR; valid building labels are 0–2, floors 0–4. Use the root command so both algorithms evaluate the identical file. This overwrites validation results and the comparison, not the trained models. Custom input does not overwrite the supplied cleaned validation CSV. Row IDs are zero-based positions in the evaluated file, not universal observation identifiers.

## 5. Methodology and reasons

### Paper basis and differences

The project adapts *Indoor Positioning System Using Wifi Fingerprint* by Dan Li, Le Wang and Shiqi Wu. The paper investigated PCA, kNN and tree/other classifiers; building-specific studies; reduced sample sizes and dimensions; and repeated ten-fold CV. This project retains the WiFi floor task, PCA before kNN and five repetitions of ten-fold CV, but uses all supplied training rows, fixed 200-component PCA, explicit training-only k selection, Random Forest instead of Gradient Boosting, and matched building-wise versus combined configurations. It is not an exact reproduction or an identical-protocol comparison to the paper.

### Cleaning and predictors

- Raw **100 means not detected**. Map it to **−110**; it is not a strong signal.
- Preserve genuine zeros. No mean/median imputation or silent row removal.
- Reject missing/nonfinite WiFi signals, invalid ranges, duplicate headers, wrong WAP order and invalid building/floor labels.
- Keep duplicate/full rows, constant features and no-signal observations. This preserves the supplied population; it does not imply duplicates are harmless for generalisation assessment.
- Model predictors are only **WAP001–WAP520**. FLOOR is the target. Coordinates, user, device, space and timestamp metadata are excluded.
- BUILDINGID routes specialists and supports subgroup reporting; it is not part of the combined predictor vector.

Cleaning is fixed/idempotent, not learned from validation statistics. Training replacements: 10,008,477; validation replacements: 559,416. Training retains 137 genuine zero cells, 637 duplicate full rows and 76 no-signal rows; validation has zero of each of those counts. Zero rows are dropped.

### PCA–kNN

- PCA: **200 components**, full SVD, mean-centred, no StandardScaler, no whitening.
- PCA is fitted only on each CV training portion, then used to transform held-out rows. Final PCA is fitted on all applicable training rows.
- 200 is a paper-motivated fixed choice, not a demonstrated universal optimum.
- kNN: exact Euclidean neighbours, equal votes; equal-distance ties follow training-row order; tied floor votes choose the smaller floor label.
- Consider all integer k from 1–25. Extend to 50, 100, 150, 200, 250 unless the best k is at least five below the current upper limit and the last five candidates are strictly worse. Never stop merely at the first upward fluctuation.
- Select the highest mean of 50 fold accuracies (lowest mean error), ties within 1e−12 favour smaller k. Each specialist selects its own k.
- Exact rankings are reused across k candidates to avoid repeated PCA and distance calculations; only the opened search stage is used for selection.
- Supplied selected k: Building 0 = **1**, Building 1 = **1**, Building 2 = **3**, combined = **1**; all searches stopped within 1–25.

### Random Forest

- **200 trees**, `max_features='sqrt'` (about 22 of 520 predictors per split), unrestricted depth, minimum leaf size 1, bootstrap enabled, seed 42, two CPU workers.
- No PCA. Forests use the 520 cleaned WiFi signals.
- Prediction averages tree class probabilities and chooses the highest-probability floor.
- Settings are fixed, not tuned on validation or through an RF parameter search.
- Forests provide a complementary nonlinear ensemble baseline. This is not a claim that they beat untested SVM/boosting methods.

### Architecture and evaluation

Four configurations: building-wise PCA–kNN, combined PCA–kNN, building-wise Random Forest, combined Random Forest. Per algorithm the three specialists train on 5,249 / 5,196 / 9,492 observations; the combined model trains on all 19,937. Known building ID selects one specialist; specialists are not voted together.

Training-only CV uses building–floor strata, seeds 42–46, ten folds per seed. Both algorithms use identical deterministic fold assignments. No location grouping is used. PCA is fold-local. All final models are refitted on the full applicable training population after k selection. The supplied official validation file is evaluated only with those frozen models.

- Accuracy = correct / total.
- Classification error = incorrect / total = 1 − accuracy.
- Mean CV score = arithmetic mean over 50 fold scores.
- Repetition accuracy = pooled held-out accuracy over all training rows once in that repetition.
- SD = sample standard deviation of the five pooled repetition accuracies, reported in percentage points; not a confidence interval.
- CSV accuracy/error fields are fractions in [0,1]; HTML formats them as percentages. `Comparison.csv` explicitly names `repeat_sd_pp` in percentage points.

## 6. Supplied executed results

| Configuration | Mean CV error | Validation error | Incorrect / 1,111 | Validation accuracy |
|---|---:|---:|---:|---:|
| Building-wise PCA–kNN | 0.200% | 11.43% | 127 | 88.57% |
| Combined PCA–kNN | 0.452% | 10.53% | 117 | 89.47% |
| Building-wise Random Forest | 0.086% | 8.91% | 99 | 91.09% |
| Combined Random Forest | 0.337% | 8.91% | 99 | 91.09% |

The two forests tie overall but do not make identical predictions. Building 1 is the weakest validation subgroup; see the HTML breakdown before drawing conclusions from the overall scores.

The supplied fitted pipelines were **migrated from the previously executed run, not retrained during folder reorganisation**. Their module paths were updated for the simpler layout and all validation predictions were checked against the original run. The CV CSVs are converted records of that original run. The reorganised training implementation retains the same evaluation/fitting rules, but a complete full-size retraining was not repeated just to rename folders.

## 7. Result files explained

- `Comparison.html`: primary offline error-first report, automatically refreshed by evaluate.
- `Comparison.csv`: one summary row per evaluated configuration.
- `CV_Results.csv`: `kind` distinguishes summary, fold and repeat records for both architectures; summary values do not invent correct counts for an unweighted mean.
- `CV_Predictions.csv`: row-level held-out predictions (199,370 per algorithm = 19,937 × 5 repeats × 2 architectures). Included to keep the CV scores auditable without a checkpoint folder.
- `K_Search.csv`: each evaluated k's mean training-CV error and selection flag, only for PCA–kNN.
- `Validation_Results.csv`: overall and per-building metrics, plus model/data provenance.
- `Validation_Predictions.csv`: two architecture-labelled predictions per validation observation, with true floor and building ID (2,222 rows per algorithm for the supplied set).
- `Confusion_Matrices.csv`: one count per actual/predicted floor pair and architecture. Rows of the displayed matrix are actual labels; columns are predicted labels.

## 8. Limitations and safety

Random-fold CV allows repeated locations and duplicate fingerprints to cross folds, so it does not measure independent unseen-location generalisation. The same folds select k and report its score (not nested CV). Official validation was inspected in earlier development, so it is a previously seen benchmark, not an untouched final test. Repeated CV does not prove absence of overfitting. No-signal scans receive a prediction without abstention. Unknown-building routing is outside this project's scope.

Load only trusted supplied `.joblib` files: pickle-based model loading can execute code. Hashes detect changed files but do not make arbitrary downloaded pickle files safe. Do not edit `metadata.json` to bypass an integrity/runtime error. Re-extract the package or use the pinned environment instead.

This is a self-contained folder: keep `run.py`, both algorithm folders and `Cleaned_Dataset` together, since saved pipelines refer to the classes in those modules. The individual runners resolve the project root from their file location, not the current working directory. Training outputs overwrite only their own algorithm's results; the original Dataset CSVs remain untouched.
