"""Validate and clean WiFi data. Originals are never modified."""
from pathlib import Path
import argparse, csv
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parents[1]
WAPS = [f'WAP{i:03d}' for i in range(1, 521)]
BUILDINGS = (0, 1, 2)
FLOORS = (0, 1, 2, 3, 4)

def read_data(path, *, training=False, labels=True):
    """Reject invalid input, without silently dropping or imputing any rows."""
    with open(path, newline='', encoding='utf-8-sig') as stream:
        header = next(csv.reader(stream))
    if len(header) != len(set(header)):
        raise ValueError('Duplicate column headers.')
    if [c for c in header if c.startswith('WAP')] != WAPS:
        raise ValueError('Expected WAP001..WAP520, in that order, with no extra WAP columns.')
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError('Input contains no observations.')
    if 'BUILDINGID' not in frame:
        raise ValueError('BUILDINGID is required for routing and reporting.')
    x = frame[WAPS].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite or missing Wi-Fi values: correct the input; no rows were dropped.')
    if not (((x >= -110) & (x <= 0)) | (x == 100)).all():
        raise ValueError('Wi-Fi values must be in [-110, 0], or raw non-detection code 100.')
    for column, allowed in [('BUILDINGID', BUILDINGS), ('FLOOR', FLOORS)]:
        if column == 'FLOOR' and not labels and column not in frame:
            continue
        if column not in frame:
            raise ValueError(f'Missing {column}.')
        values = pd.to_numeric(frame[column], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isin(values, allowed).all():
            raise ValueError(f'{column} must contain integer labels from {allowed}.')
        frame[column] = values.astype(int)
    frame[WAPS] = x
    if training and set(frame.BUILDINGID) != set(BUILDINGS):
        raise ValueError('Training data must contain all three buildings.')
    return frame


class SignalEncoder(TransformerMixin, BaseEstimator):
    """Fixed, idempotent mapping: undetected=100 -> -110; real zero is unchanged."""
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        x = np.asarray(X, dtype=np.float64).copy()
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError('Unexpected Wi-Fi feature shape.')
        if not np.isfinite(x).all() or not (((x >= -110) & (x <= 0)) | (x == 100)).all():
            raise ValueError('Invalid Wi-Fi values.')
        x[x == 100] = -110
        return x


def clean_file(source, destination, *, training=False):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Do not overwrite an original dataset.')
    frame = read_data(source, training=training)
    replacements = int((frame[WAPS].to_numpy() == 100).sum())
    frame[WAPS] = SignalEncoder().fit_transform(frame[WAPS])
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f'Cleaned {source.name}: {len(frame):,} rows; {replacements:,} replacements; 0 rows removed.', flush=True)
    return frame


def clean_all():
    for name in ('trainingData.csv', 'validationData.csv'):
        clean_file(ROOT/'Dataset'/name, ROOT/'Cleaned_Dataset'/name.replace('.csv', '_cleaned.csv'),
                   training=name.startswith('training'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    clean_all()
