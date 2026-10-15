"""One pipeline fitted on all buildings together; no building routing.

From the project root:
    python Combined_Buildings/KNN/model.py train
    python Combined_Buildings/KNN/model.py evaluate
    python Combined_Buildings/KNN/model.py demo

Use train --overwrite only to intentionally replace the saved models.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sklearn.pipeline import Pipeline
from shared import MissingSignalEncoder, SEED, run_cli
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# All learned PCA parameters are fitted inside each training CV fold.
# Float n_components values tell PCA how much variance to retain, not a fixed
# number of components. The final retained count is recorded in metadata.
PIPELINE = Pipeline([
    ('encode', MissingSignalEncoder()),
    ('pca', PCA(svd_solver='full', whiten=False, random_state=SEED)),
    ('classifier', KNeighborsClassifier(weights='uniform', metric='euclidean', n_jobs=1)),
])
PARAMETER_GRID = {
    'pca__n_components': [5, 50, 200, 0.90, 0.95, 0.99],
    'classifier__n_neighbors': list(range(1, 26, 2)),
}

if __name__ == '__main__':
    run_cli('Combined_Buildings', 'KNN', PIPELINE, PARAMETER_GRID)
