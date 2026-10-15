
"""Three building-specific pipelines, with known-building routing.

From the project root:
    python Building_Specific/Random_Forest/model.py train
    python Building_Specific/Random_Forest/model.py evaluate
    python Building_Specific/Random_Forest/model.py demo

Use train --overwrite only to intentionally replace the saved models.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sklearn.pipeline import Pipeline
from shared import MissingSignalEncoder, SEED, run_cli
from sklearn.ensemble import RandomForestClassifier

# Same forest/search settings in both architectures: isolate the training design.
# Two forest workers, one search worker: no nested worker multiplication.
PIPELINE = Pipeline([
    ('encode', MissingSignalEncoder()),
    ('classifier', RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                         random_state=SEED, n_jobs=2)),
])
PARAMETER_GRID = {
    'classifier__max_depth': [None, 20],
    'classifier__min_samples_leaf': [1, 5],
}

if __name__ == '__main__':
    run_cli('Building_Specific', 'Random_Forest', PIPELINE, PARAMETER_GRID)
