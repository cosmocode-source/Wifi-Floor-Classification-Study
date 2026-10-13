# WiFi Floor Classification

Compare **PCA + uniform kNN**, **PCA + distance-weighted kNN**, and **classical gradient boosting** on UJIIndoorLoc. Each predicts the floor within a **known building**, not coordinates.

## Separate files for each model

```text
WiFi_Floor_Project/
├── README.md                   # Setup and commands
├── InfoFiles/
│   ├── INFO.md                 # Single consolidated explanation/review file
│   └── *.pdf                   # Original paper and assignment instructions
├── Dataset/                    # Your original CSVs go here
├── pyproject.toml
├── .gitignore
├── models/
│   ├── pca_knn/
│   ├── pca_weighted_knn/
│   └── gradient_boosting/
│       # Each model folder has:
│       # clean_data.py — cleaning + model-specific scaling/PCA pipeline setup
│       # train.py      — training and cross-validation
│       # test.py       — saved-model evaluation
│       # config.yaml  — settings
├── shared/                     # Reusable validation, splits, metrics, preprocessing
│   └── manifests/              # Supplied historical audit and split files
├── comparison/                 # Compare saved results and generate reports
└── tests/                      # Synthetic unit/integration/regression tests
```

There is **no single root training/execution script**. Use the separate model modules below. Run commands from the project root, not from inside a model folder.

**Your ZIP contained neither dataset CSVs nor trained-model/results files.** Keep or copy your unchanged local files into:

```text
Dataset/trainingData.csv
Dataset/validationData.csv
```

Dataset defaults now always use `Dataset/`; do not create `data/raw/`.

## 1. Install and check — PowerShell

Python 3.10+ is declared; Python 3.12 is a sensible choice for your existing Windows setup.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$py = ".\.venv\Scripts\python.exe"
& $py -m pytest -v
```

No activation is needed. On Linux/macOS, use `.venv/bin/python` instead. The `$py` variable is PowerShell syntax; set it again when opening a new terminal.

## 2. Audit data and check shared splits

```powershell
& $py -m shared.data_validation --train-csv Dataset/trainingData.csv --val-csv Dataset/validationData.csv
```

This updates the audit without modifying the raw CSVs. Saved split files are already included. Model cleaning/training verifies their source hash against your training CSV. **Do not overwrite the supplied splits routinely.** If they are missing, or you deliberately start a new split protocol, generate them from training data only:

```powershell
& $py -m shared.create_splits --train-csv Dataset/trainingData.csv --seed 42
```

Investigate a hash mismatch first. Replacing splits invalidates comparisons to older runs.

## 3. Cleaning and feature scaling

Each model has its own `clean_data.py`. Optional cleaned CSV exports:

```powershell
& $py -m models.pca_knn.clean_data
& $py -m models.pca_weighted_knn.clean_data
& $py -m models.gradient_boosting.clean_data
```

Cleaning validates the input and replaces undetected signal `100` with `-110`. Original CSVs are preserved. These exports are optional: the training commands below read the original CSV and clean it inside the pipeline.

**Scaling is configured in each model's cleaning file but fitted during training, separately inside every CV training fold.** Do not fit a scaler/PCA to the full CSV before splitting. kNN searches scaled and unscaled configurations; classical gradient boosting uses neither scaling nor PCA by default.

## 4. Train each model separately

Use the same new run ID for this set of three runs. Train uniform kNN first, so weighted kNN can check matched configurations.

```powershell
& $py -m models.pca_knn.train --profile smoke --run-id experiment_01
& $py -m models.pca_weighted_knn.train --profile smoke --run-id experiment_01 --uniform-reference-run-dir models/pca_knn/artifacts/experiment_01
& $py -m models.gradient_boosting.train --profile smoke --run-id experiment_01
```

`smoke` limits the search candidates, **not** the number of dataset rows. It can still take time. `lightweight` and `full` profiles are available. Choose a new run ID for another experiment; existing runs are not overwritten.

## 5. Evaluate the internal holdout

```powershell
& $py -m models.pca_knn.test --train-run-dir models/pca_knn/artifacts/experiment_01 --eval-mode internal_holdout --variant both --eval-run-id experiment_01_internal
& $py -m models.pca_weighted_knn.test --train-run-dir models/pca_weighted_knn/artifacts/experiment_01 --eval-mode internal_holdout --variant both --eval-run-id experiment_01_internal --uniform-eval-run-dir models/pca_knn/results/experiment_01_internal
& $py -m models.gradient_boosting.test --train-run-dir models/gradient_boosting/artifacts/experiment_01 --eval-mode internal_holdout --variant both --eval-run-id experiment_01_internal
```

Do not repeatedly use holdout scores to choose hyperparameters. Selection uses development CV only.

## 6. Compare results

The following PowerShell block supplies explicit paths for all three models, avoiding ambiguous “latest run” selection:

```powershell
$run = "experiment_01"
$compareArgs = @()
foreach ($model in @("pca_knn", "pca_weighted_knn", "gradient_boosting")) {
    $flag = $model.Replace("_", "-")
    $compareArgs += @(
        "--${flag}-metrics", "models/$model/results/${run}_internal/tuned/metrics.json",
        "--${flag}-predictions", "models/$model/results/${run}_internal/tuned/predictions.csv",
        "--${flag}-cv-metrics", "models/$model/results/$run/cv_tuned_metrics.json",
        "--${flag}-search-csv", "models/$model/results/$run/search_results.csv"
    )
    if ($model -ne "gradient_boosting") {
        $compareArgs += @("--${flag}-validation-curve-csv", "models/$model/results/$run/validation_curve.csv")
    }
}
& $py -m comparison.compare @compareArgs --comparison-id "${run}_internal_comparison"
```

Optional saved-model demonstration:

```powershell
& $py -m shared.demo --model-id pca_weighted_knn --train-run-id experiment_01 --variant tuned --use-dev-rehearsal --samples-per-building 2 --seed 42
```

This is a **development rehearsal**, not independent accuracy evidence.

## 7. Final official evaluation — only after freezing choices

For each model, use its `test.py` module with `--eval-mode official_validation`, `--confirm-final-evaluation`, and `--refit-all-training`. Example for uniform kNN:

```powershell
& $py -m models.pca_knn.test --train-run-dir models/pca_knn/artifacts/experiment_01 --val-csv Dataset/validationData.csv --eval-mode official_validation --variant both --eval-run-id experiment_01_official --confirm-final-evaluation --refit-all-training
```

Repeat for `models.pca_weighted_knn.test` and `models.gradient_boosting.test`, changing each training artifact path. Weighted kNN also needs `--uniform-eval-run-dir models/pca_knn/results/experiment_01_official`; evaluate uniform kNN first. This refits frozen configurations on all official training rows, without retuning. The comparison block above is for **internal** results; do not mix evaluation roles.

## Outputs and limitations

- Cleaned exports: `models/<model>/processed_data/`.
- Training: `models/<model>/results/<run>/` and `artifacts/<run>/`.
- Evaluations: separate `<run>_internal/` or `<run>_official/` result directories.
- Comparison: `comparison/results/<comparison-id>/`.

Generated data/models/reports are ignored by Git. Optional academic report/slide generators are retained but not run automatically. Permanent documentation consists of this README and `InfoFiles/INFO.md` only.

**Final review: 93 tests passed; the independent model workflow passed on synthetic data.**
See INFO for tested versions, fixes, and scientific limitations. Synthetic checks do **not** establish real-data accuracy. Random splits can put duplicate fingerprints on both sides; inspect overlap warnings before claiming generalization to unseen locations. Load joblib artifacts only from trusted local sources; a checksum is not authentication.

Use `--help` on any module for its options.
Dataset source: [UCI UJIIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc).
